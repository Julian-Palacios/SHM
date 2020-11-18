import numpy as np
from pathlib import Path
import os
import sys

from obspy.core import read, UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.signal.trigger import recursive_sta_lta, delayed_sta_lta, z_detect, carl_sta_trig, trigger_onset
# from Python_tools.processing import *
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas
import pandas as pd
from shapely.geometry import Point
import contextily as ctx
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from scipy import signal
from copy import copy
import pickle

#
pd.set_option("display.max_rows", None, "display.max_columns", None)

station_params = {
    '001':{'Id':'PABUNI', 'Name':'Pabellón Central UNI','Latitude':-12.0236, 'Longitude':-77.0483, 'Location':'Pabellón-UNI, Rímac-Lima', 'Floors':3, 'N Sensors':5, 'Channels':["NS","EW","UD"], 'Skip Header':0, 'Use Columns':[2,3,4],'Scale':1,'Delimiter':','},
    '002':{'Id':'FICUNI', 'Name':'Facultad de Ingeniería Civil UNI','Latitude': -12.0218, 'Longitude': -77.049, 'Location':'FIC-UNI, Rímac-Lima', 'Floors':3, 'N Sensors':5, 'Channels':["EW","NS","UD"], 'Skip Header':4, 'Use Columns':[1,2,3],'Scale':4280,'Delimiter':'	'},
    '003':{'Id':'HERMBA', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '004':{'Id':'CIPTAR', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '005':{'Id':'MLAMAS', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '006':{'Id':'CIIFIC', 'Name':'CIIFIC UNI','Latitude': -12.0215, 'Longitude': -77.0492, 'Location':'CIIFIC-UNI, Rímac-Lima', 'Floors':8, 'N Sensors':4, 'Channels':["NS", "EW", "UD"], 'Skip Header':0, 'Use Columns':[2,3,4],'Scale':1,'Delimiter':','},
    '007':{'Id':'CCEMOS', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '008':{'Id':'LABEST', 'Name':'Laboratorio de Estructuras CISMID','Latitude': 0.0, 'Longitude': 0.0, 'Location':'CISMID FIC-UNI', 'Floors':2, 'N Sensors':2, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    }

def GL(f, fl, n):
    """
    Hace un low cut al array de frecuencias

    inputs:
        f   : array de frecuencias
        fl  : low cut frecuency
        n   : orden de corte
    output:
        GH  : Ganancia de frecuencias recortada (array)
	    GL = ( (f/fl)**(2*n)/(1 + (f/fl)**(2*n)) )**0.5
    """
    return ( (f/fl)**(2*n)/(1 + (f/fl)**(2*n)))**0.5

def GH(f, fh, n):
    """
    Hace un high cut al array de frecuencias

    input:
        f   : array de frecuencias
        fh  : high cut frecuency
        n   : orden de corte
    output:
        GH  : Ganancia de frecuencias recortada (array)
	    GH = ( 1/(1 + (f/fh)**(2*n)) )**0.5
    """
    return ( 1/(1 + (f/fh)**(2*n)) )**0.5

def Butterworth_Bandpass(signal, dt, fl, fh, n):
    """
    Hace un Butterworth Bandpass a las frecuencias de la señal

    inputs:                                         examples:
        signal      : señal (array)                         | array de aceleraciones
        dt          : delta de tiempo de la señal           | para acc = 0.01 seg
        fl          : low cut frecuency                     | fl = 0.10 Hz
        fh          : high cut frecuency                    | hf = 40.0 Hz
        n           : orden de corte                        | n = 15
    output:
        filter      : señal filtrada (array)
    """
    FFT = np.fft.rfft(signal)
    f = np.fft.rfftfreq(len(signal), d = dt)
    FFT_filtered = GL(f, fl, n)*FFT*GH(f, fh, n)

    return np.fft.irfft(FFT_filtered)

def parzem_smoothing(x, len_win_parzen=30):
	"""
	Función que suavisa una señal a traves de una ventana parzen.

	PARÁMETROS:
	x : narray de la señal a suavizar.
	len_win_parzen : ancho de la ventanda parzen, mientras más grande es este parámetro, mayor es el suavizado.
	
	RETORNOS:
	smooth: narray de la neñal suavizada.
	"""

	win = signal.parzen(len_win_parzen)
	smooth = signal.convolve(x, win, mode='same') / sum(win)

	return smooth

class Event:

    def __init__(self):
        self.epicenter = ''
        self.BASE_DIR = str(Path(__file__).resolve(strict=True).parent.parent).replace("\\",'/')
        self.event_waves_dir = ''
        self.station = pd.DataFrame({})
        self.PGA_max = None

    def load_event(self, path_event):
        """
        Método que lee los datos del Evento

        path_event  : ruta del archivo donde se encuentra los datos del evento
        """
        months = {'01':'enero', '02':'febrero', '03':'marzo', '04':'abril', '05':'mayo', '06':'junio',
                '07':'julio', '08':'agosto', '09':'setiembre', '10':'octubre', '11':'noviembre', '12':'diciembre'}

        file = open(path_event, mode='r', encoding='utf-8')
        year, month, day = file.readline().split(':')[1].strip(' ').split('-')
        day = day[1:-1] if day[0] == '0' else day[:-1]
        # fecha
        date = day + ' de ' + months[month] + ' del ' + year
        # hora local
        local_hour = file.readline().split(' ')[-1][:-1]
        # latitud
        latitude = file.readline().split(' ')[-1][:-1]
        # longitud
        longitude = file.readline().split(' ')[-1][:-1]
        # profundidad
        depth = file.readline().split(' ')[-1][:-1]
        # magnitud
        magnitude = file.readline().split(' ')[-1][:-1]
        # lugar de referencia
        venue = file.readline().split(':')[-1]
        for s in venue[:]:
            if s == ' ':
                venue = venue[1:]
            else:
                break  
        # Capeta con los datos de los accs
        self.event_waves_dir = file.readline().split(' ')[-1].strip('\n')
        # Institucion
        inst = 'IGP'
        # lugar
        place = 'Callao'
        # hora utc
        utc_hour = '06:38:02'
        
        self.epicenter = pd.DataFrame({'Latitude':[float(latitude)], 
                                        'Longitude':[float(longitude)], 
                                        'Name':['Epicentro'],
                                        'Date':[date],
                                        'Local Hour':[local_hour],
                                        'Depth':[depth],
                                        'Magnitude':[magnitude],
                                        'Venue':[venue],
                                        'Place':[place],
                                        'Institution':[inst],
                                        'UTC Hour':[utc_hour]                               
                                        })
        print("Event Loaded")

    def add_station(self, station):
        """
        Método que carga las propiedades de la estación

        station   : Objetod de la clases Station
        """
        cod = station.cod
        row = pd.DataFrame({
            'Id':[station_params[cod]['Id']],
            'Cod':[cod],
            'Latitude': [station_params[cod]['Latitude']], 
            'Longitude': [station_params[cod]['Longitude']], 
            'Name': [station_params[cod]['Name']],
            'Location':[station_params[cod]['Location']],
            'PGAs':[station.get_PGA()],
            'Channels':[station_params[cod]['Channels']],
            'Graf Acc_Four':[['No path','No path','No path']]
                            })

        self.station = self.station.append(row, ignore_index=True)
        print("Station {} - {} Added".format(cod, station_params[cod]['Id']))

    def get_max_station(self):
        # self.station["PGAs"][0] = [1,-865246,8]
        z1 = lambda row: np.min(row) if np.absolute(np.min(row)) > np.absolute(np.max(row)) else np.max(row)
        self.station["Max_pga"] = self.station["PGAs"].apply(z1)
        PGA_max = z1(self.station["Max_pga"])

        z2 = lambda x: True if x == PGA_max else False
        self.station["Is_max_station"] = self.station["Max_pga"].apply(z2)

        z3 = lambda a,b,c: ''.join([b[i] if a[i]==c else '' for i in range(3)])
        self.station['Channel_max_pga'] = [ z3(self.station["PGAs"][i], self.station["Channels"][i], self.station["Max_pga"][i]) for i in range(self.station.shape[0])]

        self.max_station = self.station[self.station["Is_max_station"]]

        print("Max Station got")

    def createMap01(self, dpi=300):
        mkdir = self.BASE_DIR + '/Figures'
        if os.path.isdir(mkdir)==False:
            os.makedirs(mkdir)

        total = pd.concat([self.epicenter, self.station], sort=False, axis=0)

        # Creación del geodataframe a partir del total de datos
        gdf = geopandas.GeoDataFrame(total, crs="EPSG:4326", geometry=geopandas.points_from_xy(total["Longitude"], total["Latitude"])) #, crs="EPSG:4326"
        # Creacion del poligono buffer
        geometry = [Point(xy).buffer(0.15) for xy in zip(self.epicenter["Longitude"], self.epicenter["Latitude"])]
        buf = geopandas.GeoDataFrame(self.epicenter, crs="EPSG:4326", geometry=geometry) 

        # Creación del Mapa
        x_size = 15 # in
        y_size = 15 # in
        fig,ax = plt.subplots(figsize=(15, 15))
        gdf[gdf['Name']!="Epicentro"].plot(ax=ax, markersize = 500, color = "yellow", marker = "^", edgecolor="black", linewidth=3, label = "Estación")
        gdf[gdf['Name']=="Epicentro"].plot(ax=ax, markersize = 650, color = "red", marker = "*", edgecolor="black", linewidth=3, label = "Epicentro")
        buf.plot(ax=ax, alpha=0.2, color = "red")

        # Configuracion de Leyenda
        plt.legend(prop={'size': 25},loc='lower left',title = "LEYENDA",title_fontsize=20)
        xmin, xmax, ymin, ymax = ax.axis()
        delta=max(ymax-ymin,xmax-xmin)

        # Asignacion de parametros en los ejes
        ax.axis((xmin, (xmin+delta), ymin, (ymin+delta)))
        ax.tick_params(axis='both', which='major', labelsize=14,width=4,length = 10,direction="inout")
        ax.tick_params(labeltop=True, labelright=True)
        ax.tick_params(top=True, right=True)     

        # Se coloca el norte
        ax.text(x=(xmin+0.9*delta), y=(ymin+0.95*delta), s='N', fontsize=30, fontweight = 'bold')
        ax.arrow((xmin+0.915*delta), (ymin+0.85*delta), 0, 0.18, length_includes_head=True,
                head_width=0.08, head_length=0.2, overhang=.1, facecolor='k')

        # Stations labels
        texts=[]
        for x, y, s in zip(self.station["Longitude"], self.station["Latitude"], self.station["Name"]):
            txt=ax.text(x, y, s, fontsize=16,color="red")
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            texts.append(txt)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        # Se añade el mapa base  
        ctx.add_basemap(ax, crs='epsg:4326', source=ctx.providers.Stamen.Terrain) 
        
        fig.canvas.start_event_loop(sys.float_info.min) # Esta linea oculta la Exception in Tkinter callback
        plt.savefig(mkdir + '/Mapa01.png', dpi=dpi, format='png', bbox_inches='tight', transparent=True) 
        fig.clf()
        plt.close()

        print("Mapa01 Created")

    def createMap02(self, dpi=300):
        mkdir = self.BASE_DIR + '/Figures'
        if os.path.isdir(mkdir)==False:
            os.makedirs(mkdir)

        if len(self.station)>1:
            dx,dy=np.max(self.station["Longitude"])-np.min(self.station["Longitude"]),np.max(self.station["Latitude"])-np.min(self.station["Latitude"])
            marg=16*abs(dy-dx)
            
            if marg>=0.4:
                marg=0.4

            if dy>dx:
                plt.rcParams["axes.xmargin"] = marg # debe ser >=0 y <=1 
                plt.rcParams["axes.ymargin"] = 0.5*marg

            else:
                plt.rcParams["axes.ymargin"] = marg
                plt.rcParams["axes.xmargin"] = 0.5*marg
        else:
            plt.rcParams["axes.xmargin"] = 0.16
            plt.rcParams["axes.ymargin"] = 0.2

        dfstation = copy(self.station)
        dfstation["Name"] = 'Estación'


        # Creación del geodataframe a partir del total de datos
        gdf = geopandas.GeoDataFrame(dfstation, geometry=geopandas.points_from_xy(dfstation["Longitude"], dfstation["Latitude"]))
        # Asignación de proyeccion en la data
        gdf.crs = "EPSG:4326"
        #Creación del Mapa01
        #aspect‘auto’, ‘equal’
        fig, ax = plt.subplots(figsize=(15,15))
        gdf[gdf['Name']=="Estación"].plot(ax=ax, markersize = 700, color = "yellow", marker = "^", edgecolor="black", linewidth=3, label = "Estación",aspect='equal')

        # Configuracion de Leyenda
        #Posible location 'upper left', 'upper right', 'lower left', 'lower right'
        plt.legend(prop={'size': 25},loc='lower left',title = "LEYENDA",title_fontsize=24)
        xmin, xmax, ymin, ymax = ax.axis()
        delta=max(ymax-ymin,xmax-xmin)
        # Asignacion de parametros en los ejes
        ax.axis((xmin, (xmin+delta), ymin, (ymin+delta)))
        ax.tick_params(axis='both', which='major', labelsize=14,width=4,length = 10,direction="inout")
        ax.tick_params(labeltop=True, labelright=True)
        ax.tick_params(top=True, right=True)
        #  Se coloca el norte
        ax.text(x=(xmin+0.9*delta), y=(ymin+0.95*delta), s='N', fontsize=30, fontweight = 'bold')
        ax.arrow((xmin+0.915*delta), (ymin+0.83*delta), 0, 0.1*delta, length_includes_head=True,
                head_width=0.05*delta, head_length=0.106*delta, overhang=0.9*delta, facecolor='k')
        
        #Stations labels
        texts=[]
        for x, y, s in zip(self.station["Longitude"], self.station["Latitude"], self.station["Name"]):
            txt=ax.text(x, y, s, fontsize=23,color="red",fontweight = 'heavy')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            texts.append(txt)
        
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        # #Se añade el mapa base
        ctx.add_basemap(ax, crs='epsg:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        
        fig.canvas.start_event_loop(sys.float_info.min) # Esta linea oculta la Exception in Tkinter callback
        plt.savefig(mkdir + '/Mapa02.png', dpi=dpi, format='png', bbox_inches='tight', transparent=True) 
        fig.clf()
        plt.close()

        print("Mapa02 Created")

    def create_acc_fourier_graf(self, station, dpi=200, transparent=False, smooth_grade=30):
        z = lambda x: np.min(x) if np.absolute(np.min(x)) > np.absolute(np.max(x)) else np.max(x)
        y_limit_acc = self._max_value_channel(station, option='acc')
        y_limit_four = self._max_value_channel(station, option='fourier')
        # print(y_limit_acc, y_limit_four)
        nrows = len(station.acc)
        size_x = 16.0
        size_y = 3.0 + 3.0*nrows
        lw = 0.2
        fs = 8
        offset = 1.25
        
        channels = self.station["Channels"][self.station["Id"]==station.acc[0][0].stats.station]
        index = channels.index
        channels = list(channels)[0]
        path_grafs = []

        for channel in range(len(channels)):
            
            mkdir = self.BASE_DIR + '/Figures/{}'.format(station.acc[0][channel].stats.station)
            if os.path.isdir(mkdir)==False:
                os.makedirs(mkdir)
        
            fig = plt.figure(constrained_layout=False, figsize=(size_x/2.54, size_y/2.54))
            # title = station.acc[0][channel].stats.channel
            # fig.text(0.55, 0.96, title, ha='center', rotation='horizontal')
            fig.text(0.015, 0.45, 'Aceleración ($cm/s^{2}$)', ha='center', rotation='vertical', fontstyle='oblique', fontsize=fs)
            fig.text(0.654, 0.43, 'Amplitud de Fourier (cm/s)', ha='center', rotation='vertical', fontstyle='oblique', fontsize=fs)
            
            widths = [2 , 1]
            heights = [1 for i in range(nrows)]
            spec = fig.add_gridspec(ncols=2, nrows=nrows, width_ratios=widths, height_ratios=heights)

            for i in range(nrows):
                itk = -i + (nrows-1)
                ax = fig.add_subplot(spec[i, 0])
                ax.plot(station.acc[itk][channel].times(), station.acc[itk][channel].data, color='k', lw=lw, 
                            label=station.acc[itk][channel].stats.network + ' pico: ' +  '{:.2f}'.format(z(station.acc[itk][channel].data)) ) 
                ax.legend(loc='upper right', fontsize=fs)
                ax.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                ax.set_xlabel('Tiempo (s)', fontsize=fs) if itk == 0 else ax.set_xticklabels([])
                ax.set_ylim( -y_limit_acc[channel]*offset, offset*y_limit_acc[channel])
                ax.set_xlim(station.acc[itk][channel].times()[0] , station.acc[itk][channel].times()[-1])
                ax.xaxis.set_tick_params(labelsize=fs)
                ax.yaxis.set_tick_params(labelsize=fs)
            
                ax = fig.add_subplot(spec[i, 1])
                ax.plot(station.fourier[itk][channel].times(), parzem_smoothing(station.fourier[itk][channel].data, len_win_parzen=smooth_grade), color='k', lw=lw,
                             label=station.acc[itk][channel].stats.network+ ' pico: ' +  '{:.2f}'.format(z(station.fourier[itk][channel].data)) )
                ax.legend(loc='upper right', fontsize=fs)
                ax.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                ax.set_xlabel('Frecuencia (Hz)', fontsize=fs) if itk == 0 else ax.set_xticklabels([])
                # ax.yaxis.tick_right()
                ax.set_ylim( None, offset*y_limit_four[channel])
                ax.set_xlim(0 , 25.0)
                ax.xaxis.set_tick_params(labelsize=fs)
                ax.yaxis.set_tick_params(labelsize=fs)
            
            fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, hspace=0.0, wspace=0.25)
            path = mkdir + '/Acc_Four_{}.png'.format(station.acc[0][channel].stats.channel)
            plt.savefig(path, dpi=dpi, transparent=transparent) 
            strip_path = path.split('/Figures/')[-1]
            # plt.show()
            path_grafs.append(strip_path)
            print("%s saved" %strip_path)

        self.station.loc[self.station["Id"]==station.acc[0][0].stats.station, "Graf Acc_Four"] = pd.Series([path_grafs], index=index)

    def create_sa_sd_graf(self, station):
        pass

    def _max_value_channel(self, station, option='acc'):
        """
        Calcula el máximo valor por canal o dirección de una estacion.

        station : Estación de la cual se hará el calculo
        option  : En que propiedad se hará. Ejemplo 'acc' -> Aceleraciones
                                                    'fourier' -> Espectros de Fourier
                                                    'sa_sd' -> Espectros de Respuesta  

        max_values_channel : Lista con los valores máximos por canal.                                           
                                                    
        """
        max_values = []

        if option == 'acc':
            data = station.acc
        if option == 'fourier':
            data = station.fourier
        if option == 'sa_sd':
            data = station.sa_sd

        for i in range(3):
            max_value = max([ np.max(np.absolute(itk[i].data)) for itk in data ])
            max_values.append(max_value)

        # print(max_values)
        return max_values

    def save_event_properties(self, path_save):
        """
        Guarda las propiedades "epicenter y station" en la ruta especificada.

        path_save   : ruta donde se guardará las propiedades del evento.
        """
        filename = 'Event_properties.sav'
        pickle.dump(self, open(path_save + '/' + filename, 'wb'))

        print("Event Properties Saved")

    @classmethod
    def load_event_properties(cls, path_load):
        """
        Carga las propiedades "epicenter y stations" de la ruta especificada.

        path_load   : ruta del archivo de donde se cargarán las propiedades del evento.
        event       : objeto de la clase Event que contiene las porpiedades cargadas
        """  
        event = pickle.load(open(path_load, 'rb'))

        return event

class Station:

    def __init__(self, dirName):
        # self.x = []
        # self.y = []
        # self.z = []
        # self.t = []
        self.acc = []
        self.vel = []
        self.desp = []
        self.fourier = []
        self.sa_sd = []
        self.BASE_DIR = str(Path(__file__).resolve(strict=True).parent.parent).replace("\\",'/')
        self.cod = ''

        self._loadStation(dirName)
        
    def _ls(self, path = os.getcwd()):
        """
        Función que ordena los archivos por piso de una carpeta proporcionada para la lectura de datos.

        PARÁMETROS:
        path : Dirección de la carpeta donde se encuentran los archivos con nombre 'itk**' a leer.

        RETORNO:
        titles : Lista con los nombres 'itk**' ordenados de menor a mayor piso.

        EJEMPLO:
        titles = [itk00, itk01, ..., itk**]
        """
        l = [arch.name for arch in os.scandir(path) if arch.is_file()]
        self.extension = '.' + l[0].split('.')[-1]
        n = len(l)
        numbers = np.array([int(i.split('itk')[-1].split('.')[0]) for i in l])
        numbers = np.sort(numbers)
        titles = []

        for i in numbers:
            if i < 10:
                titles.append('itk0' + str(i))
            else:
                titles.append('itk' + str(i))	
        return titles

    def _loadStation(self, dirName):
        self.cod = dirName.split('/')[-1]
        skip_header = station_params[self.cod]['Skip Header']
        usecols = station_params[self.cod]['Use Columns']
        scale = station_params[self.cod]['Scale']
        delimiter = station_params[self.cod]['Delimiter']
        channels = station_params[self.cod]['Channels']
        self.names = self._ls(dirName)
        
        for i in range(len(self.names)):
            wave = np.genfromtxt(fname=dirName + '/' + self.names[i] + self.extension, delimiter=delimiter, usecols=usecols, names=channels, skip_header=skip_header)
            n = wave.shape[0]
            start_time = UTCDateTime()
            st = Stream(traces=[Trace(wave[channels[0]]/scale), Trace(wave[channels[1]]/scale), Trace(wave[channels[2]]/scale)])

            for j in range(3):
                st[j].stats.network = self.names[i]
                st[j].stats.station = station_params[self.cod]['Id']
                st[j].stats._format = None,
                st[j].stats.channel = channels[j]
                st[j].stats.starttime = start_time
                st[j].stats.sampling_rate = 100
                st[j].stats.npts = n 
            self.acc.append(st)

        print("Station {0} - {1} Loaded".format(self.cod, station_params[self.cod]['Id']))

    def passBandButterWorth(self, low_freq=1.0, high_freq=25.0, order=4):
        for acc in self.acc:
            for i in range(3):
                acc[i].data = Butterworth_Bandpass(signal=acc[i].data, dt=acc[i].stats.delta, fl=low_freq, fh=high_freq, n=order)

    def baseLine(self, type='polynomial' , order=2, dspline=1000):
        # for acc in self.acc:
        #     for i in range(3):
        #         acc[i].data= BaseLineCorrection(acc[i].data, dt=acc[i].stats.delta, type=type, order=order, dspline=dspline)
        if type=='polynomial':
            for acc in self.acc:
                acc.detrend(type, order=order)

        if type=='spline':
            for acc in self.acc:
                acc.detrend(type, order=order, dspline=dspline)

    def get_vel(self):
        
        for i in range(len(self.names)):
            v = Stream(traces=[ Trace(integrate.cumtrapz(self.acc[i][0].data, dx=0.01, initial=0.0)), 
                                Trace(integrate.cumtrapz(self.acc[i][1].data, dx=0.01, initial=0.0)),
                                Trace(integrate.cumtrapz(self.acc[i][2].data, dx=0.01, initial=0.0))])

            for j in range(3):
                v[j].stats.network = self.names[i]
                v[j].stats.station = station_params[self.cod]['Id']
                v[j].stats._format = None,
                v[j].stats.channel = self.acc[i][j].stats.channel
                v[j].stats.starttime = self.acc[i][j].stats.starttime
                v[j].stats.sampling_rate = 100
                v[j].stats.npts = len(self.acc[i][j].data)
            self.vel.append(v)

    def get_desp(self):
        
        for i in range(len(self.names)):
            d = Stream(traces=[ Trace(integrate.cumtrapz(self.vel[i][0].data, dx=0.01, initial=0.0)), 
                                Trace(integrate.cumtrapz(self.vel[i][1].data, dx=0.01, initial=0.0)),
                                Trace(integrate.cumtrapz(self.vel[i][2].data, dx=0.01, initial=0.0)) ])

            for j in range(3):
                d[j].stats.network = self.names[i]
                d[j].stats.station = station_params[self.cod]['Id']
                d[j].stats._format = None,
                d[j].stats.channel = self.acc[i][j].stats.channel
                d[j].stats.starttime = self.acc[i][j].stats.starttime
                d[j].stats.sampling_rate = 100
                d[j].stats.npts = len(self.acc[i][j].data)
            self.desp.append(d)

    def get_fourier(self):

        for i in range(len(self.names)):
            tf = self.acc[i][0].times()[-1]
            npts = self.acc[i][0].stats.npts
            f0 = np.abs(np.fft.rfft(self.acc[i][0].data))/tf
            f1 = np.abs(np.fft.rfft(self.acc[i][1].data))/tf
            f2 = np.abs(np.fft.rfft(self.acc[i][2].data))/tf

            fourier = Stream(traces=[   Trace(f0), 
                                        Trace(f1),
                                        Trace(f2) ])

            for j in range(3):
                fourier[j].stats.network = self.names[i]
                fourier[j].stats.station = station_params[self.cod]['Id']
                fourier[j].stats._format = None,
                fourier[j].stats.channel = self.acc[i][j].stats.channel
                fourier[j].stats.starttime = 0
                fourier[j].stats.sampling_rate = npts/100
                fourier[j].stats.npts = len(f0)
            self.fourier.append(fourier)



        # for row in range(3):
        #     for col in range(3):
        #         ax = fig.add_subplot(spec5[row, col])
        #         # label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
        #         # ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')

        # fig5.subplots_adjust(left=0.18, bottom=0.125, right=0.95, top=0.93, hspace=0.0)
        # plt.show()

    def get_PGA(self):
        self.PGAs = []
        for i in range(3):
            mini = np.min(self.acc[0][i].data)
            maxi = np.max(self.acc[0][i].data)
            pga = mini if abs(mini) > abs(maxi) else maxi
            self.PGAs.append(pga)

        return copy(self.PGAs)

    def get_sa_sd(self):
        pass

if __name__ == '__main__':
    import datetime

    event = Event()
    event.load_event('D:/SHM/code-jj/Events/IGP EVENTOS/2020-0675.txt')

    stations = [Station('D:/SHM/code-jj/Events/2020-08-14_18-23-10/006'), Station('D:/SHM/code-jj/Events/2020-08-14_18-23-10/002'), Station('D:/SHM/code-jj/Events/2020-08-14_18-23-10/001')]
  
    stations[2].acc[2].plot()

    # for station in stations:
    #     station.baseLine(type='spline',order=2,dspline=1000)  
    #     station.passBandButterWorth(low_freq=1.0, high_freq=20.0, order=10)
    #     station.get_fourier()
    #     event.add_station(station)

    # # event.createMap01(dpi=50)
    # # event.createMap02(dpi=50)
    # # event.create_acc_fourier_graf(stations[0], dpi=50, transparent=True, smooth_grade=20)
    # # event.create_acc_fourier_graf(stations[1], dpi=50, transparent=True, smooth_grade=20)
    # # event.create_acc_fourier_graf(stations[2], dpi=50, transparent=True, smooth_grade=20)
    
    # event.get_max_station()
    # # event.station.to_excel('D:/SHM/code-jj/Report/stations.xlsx')
    # event.save_event_properties('D:/SHM/code-jj/Report')

    # stations[0].get_fourier()
    # print(station.acc[0][0].times())
    # station.acc[0][0].times = station.acc[0][0].times()*2
    # print(station.acc[0][0].times())

    # print(event.epicenter)
    # print(event.station)

    # print(event.station)

    # CIIFIC.acc[0].plot()
    # FIC.acc[0].plot()
    # PAB.acc[0].plot()
    
    # event.add_station('D:/SHM/code-jj/Stations')
    # CIIFIC.loadWaves_old('D:/SHM/code-jj/2020-11-02_2020-11-02')

    # CIIFIC.baseLine('spline', 1, 100)
    # CIIFIC.passBandButterWorth(0.01,20,10)
    # CIIFIC.createMap01()
    # CIIFIC.createMap02()


    print("Done")

    # df = 100.0
    # lthr = 0.0
    # rthr = -0.15
    # for acc in CIIFIC.acc:
    #     # ax = plt.subplot(111)
    #     for i in range(3):
    #         tr=acc[i]        
    #         # Characteristic function and trigger onsets
    #         # cft = recursive_sta_lta(tr.data, int(1 * df), int(10. * df))
    #         cft = z_detect(tr.data, int(20* df))
    #         # cft = carl_sta_trig(tr.data, int(1 * df), int(10 * df), 0.8, 0.8)
    #         # cft = delayed_sta_lta(tr.data, int(1 * df), int(10 * df))
    #         on_of = trigger_onset(cft, lthr,rthr)
    #         print(on_of)

    #         # Plotting the results
    #         ax = plt.subplot(211)
    #         plt.plot(tr.data, 'k')
    #         ymin, ymax = ax.get_ylim()
    #         # plt.vlines(on_of[:, 0], ymin, ymax, color='r', linewidth=2)
    #         # plt.vlines(on_of[:, 1], ymin, ymax, color='b', linewidth=2)
    #         plt.subplot(212, sharex=ax)
    #         plt.plot(cft, lw=0.5)
    #         # plt.hlines([lthr, rthr], 0, len(cft), color=['r', 'b'], linestyle='--')
    #         plt.axis('tight')
    #         plt.show()