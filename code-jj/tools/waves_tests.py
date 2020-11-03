import numpy as np
from os import getcwd, scandir #, makedirs, 
from obspy.core import read, UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from processing import *
import time
import matplotlib.pyplot as plt
from obspy.signal.detrend import polynomial
# from obspy.signal.filter import bandpass

class Waves:

    def __init__(self):
        # self.x = []
        # self.y = []
        # self.z = []
        # self.t = []
        self.itk = []
        self.extension = '.csv'

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
            print(wave['N_S'])
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

    def passBandButterWorth_obspy(self):

        t0 = time.time()

        for itk in self.itk:
            # itk.plot()
            for i in range(3):
                itk[i].filter('bandpass', freqmin=1.0, freqmax=5.0, corners=4)
            # itk.plot()
        tf = time.time()
        print("demoró:", tf - t0)

    def passBandButterWorth_own(self):
        t0 = time.time()
        for itk in self.itk:
            # itk.plot()
            for i in range(3):
                itk[i].data = Butterworth_Bandpass(signal=itk[i].data, dt=itk[i].stats.delta, fl=1.0, fh=5.0, n=4)
            # itk.plot()
        tf = time.time()
        print("demoró:", tf - t0)

    def passBandCompara(self):
        for itk in self.itk:
            # itk.plot()
            for i in range(3):
                plt.plot(itk[i].data,  lw=0.8, color = 'black')
                plt.plot(Butterworth_Bandpass(signal=itk[i].data, dt=itk[i].stats.delta, fl=1.0, fh=5.0, n=2), lw=0.8, color = 'blue')
                itk[i].filter('bandpass', freqmin=1.0, freqmax=5.0, corners=2)
                plt.plot(itk[i].data, lw=0.8 , alpha = 0.5, color = 'red')
                plt.show()

    def baseLine_obspy(self, type='polynomial' , order=2, dspline=1000):

        st=time.time()
        if type=='polynomial':
            for itk in self.itk:
                itk.detrend(type, order=order)

        if type=='spline':
            for itk in self.itk:
                itk.detrend(type, order=order, dspline=dspline)
        print(time.time()-st)
        
    def baseLine_compare(self, order=5):
        self.itk[0][0].data -= 0.0001 * self.itk[0][0].times() ** 3 + 0.00001 * self.itk[0][0].times() ** 5


        plt.plot(self.itk[0][0].data, 'black', lw=0.7)
        plt.plot(BaseLineCorrection(self.itk[0][0].data, order), 'b', lw=0.7)
        self.itk[0].detrend('polynomial', order=order)
        plt.plot(self.itk[0][0].data, 'red', lw=0.7)
        plt.show()

        # plt.plot(integrate.cumtrapz(self.itk[0][0].data, dx=0.01, initial=0.0), 'black', lw=0.8)
        # vt = integrate.cumtrapz(BaseLineCorrection(self.itk[0][0].data, order), dx=0.01, initial=0.0)
        # plt.plot(vt, 'b', lw=0.8)
        # self.itk[0].detrend('polynomial', order=order)
        # vt = integrate.cumtrapz(self.itk[0][0].data, dx=0.01, initial=0.0)
        # plt.plot(vt, 'r', lw=0.8)
        # plt.show()


                
  

if __name__ == '__main__':

    CIIFIC = Waves()
    CIIFIC.loadWaves_new('D:/SHM/code-jj/15-01-2020')
    # CCFIC.loadWaves_old('D:/SHM/code-jj/2020-11-02_2020-11-02')
    origi= CIIFIC.itk[0][0].data
    vt_origi = integrate.cumtrapz(origi, dx=0.01, initial=0.0)


    at = CIIFIC.itk[0][0].data + np.sin(2*np.pi*0.0025*CIIFIC.itk[0][0].times())*0.5
    vt_cagado = integrate.cumtrapz(at, dx=0.01, initial=0.0)
    # CIIFIC.itk[0].plot()
    plt.plot(at, 'r', lw=0.7)
    at= BaseLineCorrection(CIIFIC.itk[0][0].data, dt=CIIFIC.itk[0][0].stats.delta, type='spline', order=2, dspline=1000)
    vt_numpy = integrate.cumtrapz(at, dx=0.01, initial=0.0)
    plt.plot(at, 'b', lw=0.7)
    plt.show()

    # plt.plot(vt_numpy, 'g', lw=0.7)
    # plt.show()

    CIIFIC.itk[0][0].data = CIIFIC.itk[0][0].data + np.sin(2*np.pi*0.0025*CIIFIC.itk[0][0].times())*0.5
    plt.plot(CIIFIC.itk[0][0].data, 'r', lw=0.7)
    CIIFIC.baseLine_obspy(type='spline' , order=2, dspline=1000)
    plt.plot(CIIFIC.itk[0][0].data, 'b', lw=0.7)
    plt.show()
    
    vt_obspy = integrate.cumtrapz(CIIFIC.itk[0][0].data, dx=0.01, initial=0.0)
    # plt.plot(vt_obspy, 'g', lw=0.7)
    # plt.show()

    plt.plot(vt_numpy, 'b', lw=0.7)
    plt.plot(vt_obspy, 'r', lw=0.7)
    plt.plot(vt_origi, 'g', lw=0.7)
    plt.show()
    
    plt.plot(at, 'b', lw=0.7)
    plt.plot(CIIFIC.itk[0][0].data, 'r', lw=0.7)
    plt.plot(origi, 'g', lw=0.7)
    plt.show()




    # CIIFIC.baseLine('spline', 2, 1000)
    # CIIFIC.itk[0].plot()


    # CIIFIC.passBandButterWorth_obspy()
    # CIIFIC.passBandButterWorth_own()
    # CIIFIC.passBandCompara()

    # CIIFIC.baseLine_obspy()
    # CIIFIC.baseLine_compare()



    # for itk in CIIFIC.itk:

    #     print("################")
    #     print(itk[0].data)
    #     itk[0].data = itk[0].data*0

    # for itk in CIIFIC.itk:

    #     print("################")
    #     print(itk[0].data)