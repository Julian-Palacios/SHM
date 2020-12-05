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
from matplotlib.ticker import MultipleLocator
import geopandas
import pandas as pd
from shapely.geometry import Point
import contextily as ctx
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from scipy import signal
from copy import copy
import pickle
from math import e

#
from scipy.linalg import expm
from numpy.linalg import pinv
#

import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#
pd.set_option("display.max_rows", None, "display.max_columns", None)

station_params = {
    '001':{'Id':'PABUNI', 'Name':'Pabellón Central UNI','Latitude':-12.0236, 'Longitude':-77.0483, 'Location':'Pabellón-UNI, Rímac-Lima', 'Floors':3, 'N Sensors':5, 'Channels':["NS","EW","UD"], 'Skip Header':0, 'Use Columns':[2,3,4],'Scale':1,'Delimiter':','},
    # '001':{'Id':'PABUNI', 'Name':'Pabellón Central UNI','Latitude':-12.03, 'Longitude':-77.08, 'Location':'Pabellón-UNI, Rímac-Lima', 'Floors':3, 'N Sensors':5, 'Channels':["NS","EW","UD"], 'Skip Header':0, 'Use Columns':[2,3,4],'Scale':1,'Delimiter':','},
    '002':{'Id':'FICUNI', 'Name':'Facultad de Ingeniería Civil UNI','Latitude': -12.0218, 'Longitude': -77.049, 'Location':'FIC-UNI, Rímac-Lima', 'Floors':3, 'N Sensors':5, 'Channels':["EW","NS","UD"], 'Skip Header':4, 'Use Columns':[1,2,3],'Scale':4280,'Delimiter':'	'},
    '003':{'Id':'HERMBA', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '004':{'Id':'CIPTAR', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '005':{'Id':'MLAMAS', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '006':{'Id':'CIIFIC', 'Name':'CIIFIC UNI','Latitude': -12.0215, 'Longitude': -77.0492, 'Location':'CIIFIC-UNI, Rímac-Lima', 'Floors':8, 'N Sensors':4, 'Channels':["NS", "EW", "UD"], 'Skip Header':0, 'Use Columns':[2,3,4],'Scale':1,'Delimiter':','},
    '007':{'Id':'CCEMOS', 'Name':'','Latitude': 0.0, 'Longitude': 0.0, 'Location':'', 'Floors':0, 'N Sensors':0, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    '008':{'Id':'LABEST', 'Name':'Laboratorio de Estructuras CISMID','Latitude': 0.0, 'Longitude': 0.0, 'Location':'CISMID FIC-UNI', 'Floors':2, 'N Sensors':2, 'Channels':["NS", "EW", "UD"], 'Skip Header':'', 'Use Columns':'','Scale':'','Delimiter':''},
    }


# St = np.arange(1,202) # 20 seg
# St = np.arange(1,126) # 2 seg

# T = 0.0485246565*e**(0.0299572844*St)
# T = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
#                 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05])
T = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.05])

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

def ins_resp(data, dt, periods, damping = 0.05):
	'''  
	The function generates pseudo-spectral acceleration (PSA), pseudo-spectral velocity (PSV) and spectral displacement (SD) spectra for given damping ratio (xi).
	Spectral ordinates are for linear-elastic single-degree-of-freedom system with unit mass. 


	Reference:
	Wang, L.J. (1996). Processing of near-field earthquake accelerograms: Pasadena, California Institute of Technology.

	This code is converted from Matlab code of Dr. Erol Kalkan, P.E.
	Link:
	https://www.mathworks.com/matlabcentral/fileexchange/57906-pseudo-spectral-acceleration--velocity-and-displacement-spectra?s_tid=prof_contriblnk

	  INPUTS
	  
	data    = numpy array type object (in acceleration (cm/s^2))
	dt      = sampling rate
	periods = spectral periods 
	damping       = damping factor (Default: 0.05)

	  OUTPUTS
	  
	PSA = Pseudo-spectral acceleration ordinates
	PSV = Pseudo-spectral velocity ordinates
	SD  = spectral displacement ordinates

	REQUIREMENTS:
	scipy, numpy, os, matplotlib
	'''

	A = [];Ae = [];AeB = [];  
	displ_max = np.empty((len(periods)))
	veloc_max = np.empty((len(periods)))
	absacc_max = np.empty((len(periods)))
	foverm_max = np.empty((len(periods)))
	pseudo_acc_max = np.empty((len(periods)))
	pseudo_veloc_max = np.empty((len(periods)))
	PSA = np.empty((len(periods)))
	PSV = np.empty((len(periods)))
	SD = np.empty((len(periods)))

	acc = data
	#vel = data[0].integrate(method='cumtrapz')
	#dist = data[0].integrate(method='cumtrapz')

	''' Spectral solution '''

	for num,val in enumerate(periods):
		omegan = 2*np.pi/val # Angular frequency
		C = 2*damping*omegan # Two time of critical damping and angular freq.
		K = omegan**2
		y = np.zeros((2,len(acc)))
		A = np.array([[0, 1], [-K, -C]])
		Ae = expm(A*dt)
		temp_1 = Ae-np.eye(2, dtype=int)
		temp_2 = np.dot(Ae-np.eye(2, dtype=int),pinv(A))
		AeB = np.dot(temp_2,np.array([[0.0],[1.0]]))

		for k in np.arange(1,len(acc)):
		  y[:,k] = np.reshape(np.add(np.reshape(np.dot(Ae,y[:,k-1]),(2,1)), np.dot(AeB,acc[k])),(2))

		displ = np.transpose(y[0,:])	# Relative displacement vector (cm)
		veloc = np.transpose(y[1,:])	# Relative velocity (cm/s)
		foverm = (omegan**2)*displ		# Lateral resisting force over mass (cm/s2)
		absacc = -2*damping*omegan*veloc-foverm	# Absolute acceleration from equilibrium (cm/s2)

		''' Extract peak values '''
		displ_max[num] = max(abs(displ))	# Spectral relative displacement (cm)
		veloc_max[num] = max(abs(veloc))	# Spectral relative velocity (cm/s)
		absacc_max[num] = max(abs(absacc))	# Spectral absolute acceleration (cm/s2)

		foverm_max[num] = max(abs(foverm))			# Spectral value of lateral resisting force over mass (cm/s2)
		pseudo_acc_max[num] = displ_max[num]*omegan**2	# Pseudo spectral acceleration (cm/s2)
		pseudo_veloc_max[num] = displ_max[num]*omegan	# Pseudo spectral velocity (cm/s)

		PSA[num] = pseudo_acc_max[num]	# PSA (cm/s2)
		PSV[num] = pseudo_veloc_max[num]	# PSV (cm/s)
		SD[num] = displ_max[num]		# SD  (cm)

	return PSA, PSV, SD

class Event:

    def __init__(self):
        self.epicenter = ''
        self.BASE_DIR = str(Path(__file__).resolve(strict=True).parent.parent).replace("\\",'/')
        self.event_waves_dir = ''
        self.station = pd.DataFrame({})
        self.PGA_max = None

    def load_event(self,  path_event):
        months = {'01':'enero', '02':'febrero', '03':'marzo', '04':'abril', '05':'mayo', '06':'junio',
                '07':'julio', '08':'agosto', '09':'setiembre', '10':'octubre', '11':'noviembre', '12':'diciembre'}

        d = pickle.load(open(path_event, mode='rb'))

        data = {}
        data['Latitude'] = [d['Latitud']]
        data['Longitude'] = [d['Longitud']]
        data['Name'] = ['Epicentro']
        day, month, year = d['FechaLocal'].split('/')
        day = day[1:-1] if day[0] == '0' else day[:]
        date = day + ' de ' + months[month] + ' del ' + year
        data['Date'] = [date]
        data['Local Hour'] = [d['HoraLocal']]
        data['Depth'] = [d['Profundidad']]
        data['Magnitude'] = [d['Magnitud']]
        data['Venue'] = [d['Referencia']]
        data['Place'] = [d['Referencia'].split('de ')[-1]]
        data['Institution'] = ['IGP']
        hour, minute, second = d['HoraLocal'].split(':')
        time = UTCDateTime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        utc_time = time + 5*3600
        data['UTC Hour'] = [str(utc_time.time)]
        # self.event_waves_dir = data['CarpetaEvento'] 
        # self.event_waves_dir = '2020-08-14_18-23-10'
        self.event_waves_dir = '2020-11-15_08-42-16'

        self.epicenter = pd.DataFrame(data)
        
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
            'Graf Acc_Four':[['No path','No path','No path']],
            'Graf Acc_Sa':['No path']
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

    def createMap01(self, dpi=300, transparent=True):
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
        fig,ax = plt.subplots(figsize=(x_size, y_size))
        gdf[gdf['Name']!="Epicentro"].plot(ax=ax, markersize = 500, color = "yellow", marker = "^", edgecolor="black", linewidth=3, label = "Estación")
        gdf[gdf['Name']=="Epicentro"].plot(ax=ax, markersize = 650, color = "red", marker = "*", edgecolor="black", linewidth=3, label = "Epicentro")
        buf.plot(ax=ax, alpha=0.2, color = "red")

        # Configuracion de Leyenda
        plt.legend(prop={'size': 25},loc='best',title = "LEYENDA",title_fontsize=20)
        xmin, xmax, ymin, ymax = ax.axis()
        delta=max(ymax-ymin,xmax-xmin)

        # Asignacion de parametros en los ejes
        ax.axis((xmin, (xmin+delta), ymin, (ymin+delta)))
        ax.tick_params(axis='both', which='major', labelsize=14,width=4,length = 10,direction="inout")
        ax.tick_params(labeltop=True, labelright=True)
        ax.tick_params(top=True, right=True)     

        # Se coloca el norte
        # ax.text(x=(xmin+0.9*delta), y=(ymin+0.95*delta), s='N', fontsize=30, fontweight = 'bold')
        # ax.arrow((xmin+0.915*delta), (ymin+0.8*delta), 0, 0.1*delta, length_includes_head=True,
        #       head_width=0.025*delta, head_length=0.025*delta, overhang=.025*delta, facecolor='k')

        x, y, arrow_length = 0.9, 0.925, 0.1
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length), fontweight=200,
                    arrowprops=dict(facecolor='black', width=7.5, headwidth=25, headlength=25),
                    ha='center', va='center', fontsize=35, 
                    xycoords=ax.transAxes)


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
        plt.savefig(mkdir + '/Mapa01.png', dpi=dpi, format='png', bbox_inches='tight', transparent=transparent) 
        # plt.show()
        fig.clf()
        plt.close()

        print("Mapa01 Created")

    def createMap02(self, dpi=300, transparent=True):
        mkdir = self.BASE_DIR + '/Figures'
        if os.path.isdir(mkdir)==False:
            os.makedirs(mkdir)

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
        plt.legend(prop={'size': 25},loc='best',title = "LEYENDA",title_fontsize=24)
        xmin, xmax, ymin, ymax = ax.axis()
        delta=max(ymax-ymin,xmax-xmin)
        # Asignacion de parametros en los ejes
        ax.axis((xmin, (xmin+delta), ymin, (ymin+delta)))
        # ax.tick_params(axis='both', which='major', labelsize=14,width=4,length = 10,direction="inout")
        # ax.tick_params(axis='both', labeltop=False, labelright=False)
        # ax.tick_params(top=False, right=False)
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            labeltop=False) # labels along the bottom edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            right=False,      # ticks along the bottom edge are off
            left=False,         # ticks along the top edge are off
            labelleft=False,
            labelright=False) # labels along the bottom edge are off
        #  Se coloca el norte
        # ax.text(x=(xmin+0.9*delta), y=(ymin+0.95*delta), s='N', fontsize=30, fontweight = 'bold')
        # ax.arrow((xmin+0.915*delta), (ymin+0.8*delta), 0, 0.1*delta, length_includes_head=True,
        #       head_width=0.025*delta, head_length=0.025*delta, overhang=.025*delta, facecolor='k')
        x, y, arrow_length = 0.9, 0.925, 0.1
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length), fontweight=200,
                    arrowprops=dict(facecolor='black', width=7.5, headwidth=25, headlength=25),
                    ha='center', va='center', fontsize=35, 
                    xycoords=ax.transAxes)

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
        plt.savefig(mkdir + '/Mapa02.png', dpi=dpi, format='png', bbox_inches='tight', transparent=transparent) 
        # plt.show()
        fig.clf()
        plt.close()

        print("Mapa02 Created")

    def create_acc_fourier_graf(self, station, dpi=200, transparent=False):
        z = lambda x: np.min(x) if np.absolute(np.min(x)) > np.absolute(np.max(x)) else np.max(x)
        y_limit_acc = self._max_value_channel(station, option='acc')
        y_limit_four = self._max_value_channel(station, option='fourier')
        
        # try:
        # acc_step_major, best_y_limit_acc  = self._get_y_limits(y_limit_acc)
        # four_step_major, best_y_limit_four  = self._get_y_limits(y_limit_four)
        # except:
        #     print("Fallo obtencion limites")
        #     pass

        # print(y_limit_acc, y_limit_four)
        nrows = len(station.acc)
        size_x = 16.0
        size_y = 3.0 + 3.0*nrows
        lw = 0.2
        fs = 8
        fs_ticks = 6
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

                # Acceleration
                ax = fig.add_subplot(spec[i, 0])
                ax.plot(station.acc[itk][channel].times(), station.acc[itk][channel].data, color='k', lw=lw, 
                            label=station.acc[itk][channel].stats.network + ' pico: ' +  '{:.2f}'.format(z(station.acc[itk][channel].data)) ) 
                ax.legend(loc='upper right', fontsize=fs)
                ax.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                ax.set_xlabel('Tiempo (s)', fontsize=fs) if itk == 0 else ax.set_xticklabels([])
                # try:
                #     acc_step_major, best_y_limit_acc  = self._get_y_limits(round(y_limit_acc[channel],2))
                #     ax.set_ylim(-best_y_limit_acc, best_y_limit_acc)
                #     print(round(y_limit_acc[channel],2), acc_step_major, best_y_limit_acc)
                #     # ax.yaxis.set_major_locator(MultipleLocator(acc_step_major))
                # except:
                #     print("fallo y axis acc")
                ax.set_ylim( -y_limit_acc[channel]*offset, offset*y_limit_acc[channel])
                
                ax.set_xlim(station.acc[itk][channel].times()[0] , station.acc[itk][channel].times()[-1])
                ax.xaxis.set_tick_params(labelsize=fs_ticks)
                ax.yaxis.set_tick_params(labelsize=fs_ticks)
            
                # Fourier
                ax = fig.add_subplot(spec[i, 1])
                ax.plot(station.fourier[itk][channel].times(), station.fourier[itk][channel].data, color='k', lw=lw,
                             label=station.acc[itk][channel].stats.network+ ' pico: ' +  '{:.2f}'.format(z(station.fourier[itk][channel].data)) )
                ax.legend(loc='upper right', fontsize=fs)
                ax.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
                ax.set_xlabel('Frecuencia (Hz)', fontsize=fs) if itk == 0 else ax.set_xticklabels([])
                # try:
                #     four_step_major, best_y_limit_four  = self._get_y_limits(round(y_limit_four[channel],2))
                #     print(round(y_limit_four[channel],2),four_step_major, best_y_limit_four)
                #     ax.set_ylim(None, best_y_limit_four)
                #     # ax.yaxis.set_major_locator(MultipleLocator(four_step_major))                    
                # except:
                #     print("fallo y axis four")
                ax.set_ylim(None, offset*y_limit_four[channel])

                ax.set_xlim(0 , 25.0)
                ax.xaxis.set_tick_params(labelsize=fs_ticks)
                ax.yaxis.set_tick_params(labelsize=fs_ticks)
            
            fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, hspace=0.0, wspace=0.25)
            path = mkdir + '/Acc_Four_{}.png'.format(station.acc[0][channel].stats.channel)
            plt.savefig(path, dpi=dpi, transparent=transparent) 
            strip_path = path.split('/Figures/')[-1]
            # plt.show()
            path_grafs.append(strip_path)
            print("%s saved" %strip_path)

        self.station.loc[self.station["Id"]==station.acc[0][0].stats.station, "Graf Acc_Four"] = pd.Series([path_grafs], index=index)

    def create_acc_sa_graf(self, station, dpi=200, transparent=False):
        z = lambda x: np.min(x) if np.absolute(np.min(x)) > np.absolute(np.max(x)) else np.max(x)
        y_limit_acc = max([np.max(np.absolute(station.acc[0][i].data)) for i in range(3)])
        y_limit_sa =  max([np.max(np.absolute(station.sa[i])) for i in range(3)])

        nrows = 3
        size_x = 16.0
        size_y = 3.0 + 3.0*nrows
        lw = 0.2
        fs = 8
        fs_ticks = 6
        offset = 1.25

        channels = self.station["Channels"][self.station["Id"]==station.acc[0][0].stats.station]
        index = channels.index

        mkdir = self.BASE_DIR + '/Figures/{}'.format(station.acc[0][0].stats.station)
        if os.path.isdir(mkdir)==False:
            os.makedirs(mkdir)

        fig = plt.figure(constrained_layout=False, figsize=(size_x/2.54, size_y/2.54))
        fig.text(0.015, 0.42, 'Aceleración ($cm/s^{2}$)', ha='center', rotation='vertical', fontstyle='oblique', fontsize=fs)
        fig.text(0.654, 0.37, 'Pseudo Aceleración ($cm/s^{2}$)', ha='center', rotation='vertical', fontstyle='oblique', fontsize=fs)
        
        widths = [2 , 1]
        heights = [1 for i in range(nrows)]
        spec = fig.add_gridspec(ncols=2, nrows=nrows, width_ratios=widths, height_ratios=heights)

        for i in range(3):
            
            # Acceleration
            ax = fig.add_subplot(spec[i, 0])
            ax.plot(station.acc[0][i].times(), station.acc[0][i].data, color='k', lw=lw, 
                        label= 'Dirección: ' + station.acc[0][i].stats.channel + ' pico: ' +  '{:.2f}'.format(z(station.acc[0][i].data)) ) 
            ax.legend(loc='upper right', fontsize=fs)
            ax.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
            ax.set_xlabel('Tiempo (s)', fontsize=fs) if i == 2 else ax.set_xticklabels([])

            ax.set_ylim( -y_limit_acc*offset, offset*y_limit_acc)
            ax.set_xlim(station.acc[0][i].times()[0] , station.acc[0][i].times()[-1])
            ax.xaxis.set_tick_params(labelsize=fs_ticks)
            ax.yaxis.set_tick_params(labelsize=fs_ticks)
            
            # Sa Sd
            ax = fig.add_subplot(spec[i, 1])
            ax.plot(np.insert(T, 0, 0), np.insert(station.sa[i], 0, abs(z(station.acc[0][i].data))), color='k', lw=lw,
                            label='Dirección: ' + station.acc[0][i].stats.channel )
            ax.legend(loc='upper right', fontsize=fs)
            ax.grid(True, color='k', linestyle='-', linewidth=0.4, which='both', alpha = 0.2)
            ax.set_xlabel('Periodo (s)', fontsize=fs) if i == 2 else ax.set_xticklabels([])
            ax.set_xlim(0.0 , 2.0)

            ax.set_ylim(None, offset*y_limit_sa)
            ax.xaxis.set_tick_params(labelsize=fs_ticks)
            ax.yaxis.set_tick_params(labelsize=fs_ticks)

        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, hspace=0.0, wspace=0.25)
        path = mkdir + '/Acc_Sa_{}.png'.format(station.acc[0][0].stats.station)
        plt.savefig(path, dpi=dpi, transparent=transparent) 
        strip_path = path.split('/Figures/')[-1]
        # plt.show()
        print("%s saved" %strip_path)
        self.station.loc[self.station["Id"]==station.acc[0][0].stats.station, "Graf Acc_Sa"] = pd.Series([strip_path], index=index)

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
            for i in range(3):
                max_value = max([ np.max(np.absolute(itk[i].data)) for itk in data ])
                max_values.append(max_value)

        if option == 'fourier':
            data = station.fourier
            for i in range(3):
                max_value = max([ np.max(np.absolute(itk[i].data)) for itk in data ])
                max_values.append(max_value)
            
        if option == 'sa':
            for i in range(3):
                max_value = max([ np.max(np.absolute(sa)) for sa in station.sa ])
                max_values.append(max_value)
   
        if option == 'sd':
            for i in range(3):
                max_value = max([ np.max(np.absolute(sd)) for sd in station.sd ])
                max_values.append(max_value)

        return max_values

    def _get_y_limits(self, x):
        x = float(x)
        s = str(x).replace('.','')
        if x < 1:
            for i in range(len(s)):
                if s[i]!='0':
                    int_part = int(float(s[i:]))
                    break
        else:
            for i in range(len(s)-1,-1,-1):
                if s[i]!='0':
                    int_part = int(float(s[:i+1]))
                    break
                
        n= round(10*int_part/5)*5
        div = round(int_part/x)*10

        major_step = n/2
        y_lim = n + 0.5*major_step

        return (major_step/div, y_lim/div)

    def save_event_properties(self, path_save):
        """
        Guarda las propiedades "epicenter y station" en la ruta especificada.

        path_save   : ruta donde se guardará las propiedades del evento.
        """
        filename = 'Event.sav'
        pickle.dump(self, open(path_save + '/' + filename, 'wb'))

        print("Event Properties Saved")

    def purge_files(self, path_dir):
        list_remove = ["Report.aux", "Report.fdb_latexmk", "Report.fls", "Report.lof", 
                        "Report.log", "Report.lot", "Report.out", "Report.pdf", "Report.pytxcode" ]

        try:
            shutil.rmtree(path_dir + 'pythontex-files-report')
            print('pythontex-files-report Deleted')
        except:
            print("'pythontex-files-report' no Found")

        for file in list_remove:
            try:
                os.remove(path_dir + file)
                print(file + " Deleted.")
            except:
                print(file + " No Found.")        

    def send_email(self, report_path= "D:/SHM/code-jj/Report/Report.pdf", receivers =  ["jjaramillod@uni.pe", "josdaroldplx@gmail.com", "josdarcoldplx@hotmail.com"]):

        subject = "Acelerogramas del Sismo de %s del %s" %(self.epicenter["Place"].iloc[0], self.epicenter["Date"].iloc[0])
        body = "Esta es una prueba de envio automático de Reporte Sismico"
        sender_email = "cismid.remoed@gmail.com"
        receiver_email = receivers
        # receiver_email = receivers
        password = "c15m1dr3m0ed"

        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(receivers)
        message["Subject"] = subject
        message["Bcc"] = ", ".join(receivers) # Recommended for mass emails

        html = """\
        <html>
        <body>
            <p>Saludos Estimados, Este es un mensaje de prueba de envio automático  de Reporte Sísmico generado con python. <br>
            Referencia: 
            <a href="http://www.realpython.com"></a> 
            </p>
            <br>
            <b> Atte: Joseph Jaramillo del Aguila.</b>
        </body>
        </html>
        """


        # Add body to email
        message.attach(MIMEText(body, "plain"))
        message.attach(MIMEText(html, "html"))

        # filename = pdf_path  # In same directory as script
        

        # Open PDF file in binary mode
        with open(report_path, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {report_path}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()

        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            # server.sendmail(sender_email, receiver_email, text)
            server.sendmail(message["From"], receiver_email, text)

        print("#### Email send ####")

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
        self.sa = []
        self.sd = []
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

    def get_fourier(self, smooth_grade=25):

        for i in range(len(self.names)):
            tf = self.acc[i][0].times()[-1]
            npts = self.acc[i][0].stats.npts
            f0 = np.abs(np.fft.rfft(self.acc[i][0].data))/tf
            f1 = np.abs(np.fft.rfft(self.acc[i][1].data))/tf
            f2 = np.abs(np.fft.rfft(self.acc[i][2].data))/tf

            fourier = Stream(traces=[   Trace(parzem_smoothing(f0, len_win_parzen=smooth_grade)), 
                                        Trace(parzem_smoothing(f1, len_win_parzen=smooth_grade)),
                                        Trace(parzem_smoothing(f2, len_win_parzen=smooth_grade)) ])

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
        # s = np.arange(1,202)
        # T = 0.0485246565*e**(0.0299572844*s)

        dt = 0.01 # para itks

        for i in range(3):
            Sa, Sv, Sd = ins_resp(self.acc[0][i], dt, T,  damping=0.05)
            self.sa.append(Sa)
            self.sd.append(Sd)
        
if __name__ == '__main__':
    # step 1) Leer el evento del IGP y crear el objeto de la clase Event.
    event = Event()
    # event.load_event('D:/SHM/code-jj/Events/IGP EVENTOS/2020-0675.txt')
    event.load_event('D:/SHM/code-jj/Events/IGP EVENTOS/2020-0709.sav')
    # print(event.epicenter)

    # step 2) leer la carpeta donde se encuentras los registros de las estaciones.
    path_event = 'D:/SHM/code-jj/Events/%s' %event.event_waves_dir
    with os.scandir(path_event) as f:
        s = [f.name for f in f if f.is_dir()]

    # step 3) Cargar los datos de las estaciones creando los objetos de la clase Station.
    stations = [Station(path_event + '/%s' %i) for i in s]

    for station in stations:
        station.baseLine(type='spline',order=2,dspline=1000) 
        # station.get_fourier(smooth_grade=25)
        station.get_sa_sd()

        event.add_station(station)

        # event.create_acc_fourier_graf(station, dpi=50, transparent=True)

        # station.get_sa_sd()

        event.create_acc_sa_graf(station, dpi=50, transparent=False)
        # station.acc[0][0].plot()