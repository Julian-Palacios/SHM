import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

def GB(f, fl, fh, n):
    """
    Hace un Butterworth al array de frecuencias

    inputs:
        f   : array de frecuencias
        fl  : low cut frecuency
        fh  : high cut frecuency
        n   : orden de corte
    output:
        GB  : Ganancia de frecuencias recortada (array)
    """
    return GL(f, fl, n)*GH(f, fh, n)

def Butterworth_Bandpass(signal, samplig_rate, fl, fh, n):
    """
    Hace un Butterworth Bandpass a las frecuencias de la señal

    inputs:                                         examples:
        signal      : señal (array)                         | array de aceleraciones
        samplig_rate: delta de tiempo de la señal           | para itk = 0.01 seg]
        fl          : low cut frecuency                     | fl = 0.10 Hz
        fh          : high cut frecuency                    | hf = 40.0 Hz
        n           : orden de corte                        | n = 15
    output:
        filter      : señal filtrada (array)
    """
    FFT = np.fft.rfft(signal)
    f = np.fft.rfftfreq(len(signal), d = samplig_rate)
    FFT_filtered = GL(f, fl, n)*FFT*GH(f, fh, n)

    return np.fft.irfft(FFT_filtered)

if __name__ == '__main__':
    paso=0.01 # paso del tiempo en s
    tf = 100.0
    # tiempo final iniciando desde 0 en s
    t = np.arange(0.0,tf,paso)

    # WHITE NOISE
    mean = 0
    std = 1
    num_samples = int(tf/paso)
    wn = 0.1*np.random.normal(mean, std, size = num_samples)/9.81
    dt = paso

    n = len(t)
    FFT = np.fft.rfft(wn)
    f = np.fft.rfftfreq(n, d = dt)

    fl = 1.0
    fh = 1.0

    x_cm = 13.0
    y_cm = 6.0
    lw= 0.8
    fs = 8
    ls = 6
    spw = 0.2
    alpha = 0.9

    # plt.plot(t, wn, 'r', lw=0.8)
    plt.plot(f, abs(FFT),  'r', lw=0.8)
    plt.plot(f, abs(FFT)*GB(f, fl, fh, 20), 'b', lw=0.8)
    plt.show()
    plt.close()

    plt.plot(t, wn,  'r', lw=0.8)
    plt.plot(t, Butterworth_Bandpass(wn, dt, fl, fh, 20), 'b', lw=0.8)
    plt.show()
    plt.close()
