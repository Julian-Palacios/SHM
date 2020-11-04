import numpy as np
from os import getcwd, scandir #, makedirs, 
from obspy.core import read, UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from processing import *
import time
import matplotlib.pyplot as plt
# from obspy.signal.filter import bandpass

class Waves:

    def __init__(self):
        # self.x = []
        # self.y = []
        # self.z = []
        # self.t = []
        self.itk = []

    def _ls(self, path = getcwd()):
        """
        Función que ordena los archivos por piso de una carpeta proporcionada para la lectura de datos.

        PARÁMETROS:
        path : Dirección de la carpeta donde se encuentran los archivos con nombre 'itk**' a leer.

        RETORNO:
        titles : Lista con los nombres 'itk**' ordenados de menor a mayor piso.

        EJEMPLO:
        titles = [itk00, itk01, ..., itk**]
        """
        l = [arch.name for arch in scandir(path) if arch.is_file()]
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
            st = Stream(traces=[Trace(wave[channels[0]]), Trace(wave[channels[1]]), Trace(wave[channels[2]])])

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
        for itk in self.itk:
            for i in range(3):
                itk[i].data= BaseLineCorrection(itk[i].data, dt=itk[i].stats.delta, type=type, order=order, dspline=dspline)


if __name__ == '__main__':

    CIIFIC = Waves()
    # CIIFIC.loadWaves_new('D:/SHM/code-jj/15-01-2020')
    CIIFIC.loadWaves_old('D:/SHM/code-jj/2020-11-02_2020-11-02')

    # plt.plot(CIIFIC.itk[0][0].data, 'r', lw=0.6)
    # CIIFIC.passBandButterWorth(1.0, 5.0, 10)
    # plt.plot(CIIFIC.itk[0][0].data, 'b', lw=0.6)
    # plt.show()

    # CIIFIC.itk[0][0].data = CIIFIC.itk[0][0].data + np.sin(2*np.pi*0.0025*CIIFIC.itk[0][0].times())*0.5 + 5
    # plt.plot(CIIFIC.itk[0][0].data, 'r', lw=0.6)
    # CIIFIC.baseLine('spline', 2, 1000)
    # plt.plot(CIIFIC.itk[0][0].data, 'b', lw=0.6)
    # plt.show()



    # print(CIIFIC.itk[0][0].stats.starttime)
    # print()
    # print(CIIFIC.itk[1][0].stats.starttime)
    # print()
    # print(CIIFIC.itk[2][0].stats.starttime)
    # # for itk in CIIFIC.itk:
    # #     for 



