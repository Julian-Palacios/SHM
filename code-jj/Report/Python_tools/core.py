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
import matplotlib.pyplot as plt
import geopandas
import pandas as pd
from shapely.geometry import Point
import contextily as ctx
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from scipy import signal
from copy import copy
import pickle

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
        dt          : delta de tiempo de la señal           | para itk = 0.01 seg
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

class Event:

    def __init__(self):
        self.station = ''
        self.epicenter = ''
        self.BASE_DIR = str(Path(__file__).resolve(strict=True).parent.parent).replace("\\",'/')

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

    def load_stations(self, path_stations):
        """
        Método que carga las estaciones detectadas

        path_stations   : ruta de la carpeta que contiene las subcarpetas de cada estación
        """
        self.station = pd.DataFrame({'Latitude': [-12.14, -12.0976, -12.0605], 
                                    'Longitude': [-76.94, -77.0172, -76.9759], 
                                    'Name': ['CIIFIC', 'FIC-UNI', 'CISMID'], 
                                    'Location':['CIIFIC-FIC-UNI, Rímac, Lima', 'FIC-UNI, Rímac, Lima', 'CISMID-FIC-UNI, Rímac, Lima'],
                                    'PGA':[[3.54, 3.55, 1.52], [-1.04, -1.00, -0.76], [-1.19, 1.43, -1.07]]
                                    })
        
        print(np.max(np.array(self.station["PGA"])))
        print(np.min(self.station["PGA"]))

        print("Stations Loaded")

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
        plt.savefig(mkdir + '/Mapa01.png', dpi=dpi, format='png', bbox_inches='tight') 
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
        plt.savefig(mkdir + '/Mapa02.png', dpi=dpi, format='png', bbox_inches='tight') 
        fig.clf()
        plt.close()

        print("Mapa02 Created")

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

class Waves:

    def __init__(self):
        # self.x = []
        # self.y = []
        # self.z = []
        # self.t = []
        self.itk = []
        self.itk_v = []
        self.itk_d = []
        self.BASE_DIR = str(Path(__file__).resolve(strict=True).parent.parent).replace("\\",'/')
        # self.station = pd.DataFrame({'Latitude': [-12.1740, -12.0976, -12.0605], 
        #                             'Longitude': [-77.0191, -77.0172, -76.9759], 
        #                             'Name': ['CIIFIC', 'FIC-UNI', 'CISMID'], 
        #                             'Location':['CIIFIC-FIC-UNI, Rímac, Lima', 'FIC-UNI, Rímac, Lima', 'CISMID-FIC-UNI, Rímac, Lima'],
        #                             'PGA':[[3.54, 3.55, 1.52], [-1.04, -1.00, -0.76], [-1.19, 1.43, -1.07]]
        #                             })

        # self.epicenter = pd.DataFrame({'Latitude': [-9.95], 'Longitude':[-78.96], 'Name': ['Epicentro']})
        
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

    def loadWaves_old(self, dirName):
        channels = ["N_S", "E_W", "U_D"] # FIC-UNI
        scale = 4280
        self.names = self._ls(dirName)
        
        for i in range(len(self.names)):
            wave = np.genfromtxt(fname=dirName + '/' + self.names[i] + self.extension, delimiter=',', usecols=[2,3,4], names=channels, skip_header=4) #FIC-UNI
            n = wave.shape[0]
            file = open(dirName + '/' + self.names[i] + self.extension, 'r')
            file.readline()
            file.readline()
            file.readline()
            file.readline()
            line = file.readline().split(',')
            date = '20' + line[0].replace('/', '-').split(',')[0]
            hour = line[1]
            start_time = UTCDateTime(date + 'T' + hour)
            st = Stream(traces=[Trace(wave[channels[0]]/scale), Trace(wave[channels[1]]/scale), Trace(wave[channels[2]]/scale)])

            for j in range(3):
                st[j].stats.network = self.names[i]
                st[j].stats.station = 'FIC-UNI'
                st[j].stats._format = None,
                st[j].stats.channel = channels[j]
                st[j].stats.starttime = start_time
                st[j].stats.sampling_rate = 100
                st[j].stats.npts = n 
            self.itk.append(st)

    def loadWaves_new(self, dirName):
        channels = ["N_S", "E_W", "U_D"] # FIC-UNI
        self.names = self._ls(dirName)
        
        for i in range(len(self.names)):
            wave = np.genfromtxt(fname=dirName + '/' + self.names[i] + self.extension, delimiter=',', usecols=[2,3,4], names=channels, skip_header=0) # CCIFIC
            n = wave.shape[0]
            line = open(dirName + '/' + self.names[i] + self.extension, 'r').readline().split(',') # CCIFIC
            date = line[0].replace('/', '-').split(',')[0]
            hour = line[1].split(':') # CCIFIC
            hour = ':'.join(hour[0:3]) + '.' + hour[-1] #CCIFIC
            start_time = UTCDateTime(date + 'T' + hour)
            st = Stream(traces=[Trace(wave[channels[0]]), Trace(wave[channels[1]]), Trace(wave[channels[2]])])

            for j in range(3):
                st[j].stats.network = self.names[i]
                st[j].stats.station = 'CIIFIC'
                st[j].stats._format = None
                st[j].stats.channel = channels[j]
                st[j].stats.starttime = start_time
                st[j].stats.sampling_rate = 100.0
                st[j].stats.npts = n 
            self.itk.append(st)

    def passBandButterWorth(self, low_freq=1.0, high_freq=25.0, order=4):
        for itk in self.itk:
            for i in range(3):
                itk[i].data = Butterworth_Bandpass(signal=itk[i].data, dt=itk[i].stats.delta, fl=low_freq, fh=high_freq, n=order)

    def baseLine(self, type='polynomial' , order=2, dspline=1000):
        # for itk in self.itk:
        #     for i in range(3):
        #         itk[i].data= BaseLineCorrection(itk[i].data, dt=itk[i].stats.delta, type=type, order=order, dspline=dspline)
        if type=='polynomial':
            for itk in self.itk:
                itk.detrend(type, order=order)

        if type=='spline':
            for itk in self.itk:
                itk.detrend(type, order=order, dspline=dspline)

    def get_vel(self):
        
        for i in range(len(self.names)):
            v = Stream(traces=[ Trace(integrate.cumtrapz(self.itk[i][0].data, dx=0.01, initial=0.0)), 
                                Trace(integrate.cumtrapz(self.itk[i][1].data, dx=0.01, initial=0.0)),
                                Trace(integrate.cumtrapz(self.itk[i][2].data, dx=0.01, initial=0.0))])

            for j in range(3):
                v[j].stats.network = self.names[i]
                v[j].stats.station = 'FIC-UNI'
                v[j].stats._format = None,
                v[j].stats.channel = self.itk[i][j].stats.channel
                v[j].stats.starttime = self.itk[i][j].stats.starttime
                v[j].stats.sampling_rate = 100
                v[j].stats.npts = len(self.itk[i][j].data)
            self.itk_v.append(v)

    def get_desp(self):
        
        for i in range(len(self.names)):
            d = Stream(traces=[ Trace(integrate.cumtrapz(self.itk_v[i][0].data, dx=0.01, initial=0.0)), 
                                Trace(integrate.cumtrapz(self.itk_v[i][1].data, dx=0.01, initial=0.0)),
                                Trace(integrate.cumtrapz(self.itk_v[i][2].data, dx=0.01, initial=0.0))])

            for j in range(3):
                d[j].stats.network = self.names[i]
                d[j].stats.station = 'FIC-UNI'
                d[j].stats._format = None,
                d[j].stats.channel = self.itk[i][j].stats.channel
                d[j].stats.starttime = self.itk[i][j].stats.starttime
                d[j].stats.sampling_rate = 100
                d[j].stats.npts = len(self.itk[i][j].data)
            self.itk_d.append(d)

if __name__ == '__main__':

    event = Event()
    event.load_event('D:/SHM/code-jj/Events/2020-0675.txt')
    event.load_stations('D:/SHM/code-jj/Stations')
    # event.save_event_properties('D:/SHM/code-jj/Report')
    # event.createMap01(dpi=100)
    # event.createMap02(dpi=100)
    

    # CIIFIC = Waves()
    # CIIFIC.loadWaves_new('D:/SHM/code-jj/15-01-2020')
    # CIIFIC.loadWaves_old('D:/SHM/code-jj/2020-11-02_2020-11-02')

    # CIIFIC.baseLine('spline', 1, 100)
    # CIIFIC.passBandButterWorth(0.01,20,10)
    # CIIFIC.createMap01()
    # CIIFIC.createMap02()


    print("Done")

    # df = 100.0
    # lthr = 0.0
    # rthr = -0.15
    # for itk in CIIFIC.itk:
    #     # ax = plt.subplot(111)
    #     for i in range(3):
    #         tr=itk[i]        
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