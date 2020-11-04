import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from numpy.linalg import inv
from scipy.interpolate import LSQUnivariateSpline

def cut(t, x, t0, tf):
    """
    Recorta la señal función del tiempo x(t) en un rango determinado.

    PARÁMETROS:
    t       : narray de tiempo
    x       : narray señal en funcion del tiempo (t)
    t0      : valor de tiempo de corte inferior en segundos
    tf      : valor de tiempo de corte superios en segundos

    NOTA: t0 y tf deben ser un multiplo de dt donde
            dt la frecuencia de muestreo de la señal 
            o paso de tiempo t, por ejemplo para los itk dt = 0.01 seg

    RETORNOS:
    x       : señal recortada

    """
    index_t0 = np.where(t == t0)[0][0]
    index_tf = np.where(t == tf)[0][0]

    return x[index_t0:index_tf+1]

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

def LeastSquares(x, y , orden):
    """
    Realiza un regresión polinomial por el método de mínimos cuadrados

    PARÁMETROS:
    x       : narray referente a la variable independiente. Para este caso el tiempo en segundos
    y       : narray referente a la variable dependiente. Para este caso una señal.
    orden   : grado del polinomio

    RETORNOS:
    rect : narray de la señal ajustada a un polinomio de grado n (orden)
    """
    n = orden + 1
    m = len(x)
    A = np.zeros((m, n))

    for i in range(n):
        A[:,i:i+1] = np.array([x**i]).T
    a = inv(A.T@A)@A.T@y
    poly_n = A@a

    return poly_n.T

def BaseLineCorrection(at, dt=0.01, type='polynomial', order=2, dspline=1000):
    """
    Realiza una corrección por Línea Base a un array de aceleraciones

    PARÁMETROS:
    at      : narray de aceleraciones 
    dt      : delta de tiempo en seguntos. para itk=0.01s
    type    : método de ajuste ('polynomial', 'spline')
    order   : orden del polinomio de aproximación para la línea base
    dspline : en caso de ser el método spline, define cada cuantos puntos se debe hacer el ajuste

    RETORNOS:
    at  : señal de aceleraciones corregida
    """
    vt = integrate.cumtrapz(at, dx=dt, initial=0.0)
    x = np.arange(len(vt))
    
    if type=='polynomial':
        fit_vt = np.polyval(np.polyfit(x, vt, deg=order), x)
        
    if type =='spline':
        splknots = np.arange(dspline / 2.0, len(vt) - dspline / 2.0 + 2, dspline)
        spl = LSQUnivariateSpline(x=x, y=vt, t=splknots, k=order)
        fit_vt = spl(x)

    return at - np.gradient(fit_vt, dt)
    

if __name__ == '__main__':
    from copy import copy

    lw = 1.0

    itk = 'D:/GitLab/automata/workspace/Centro_OK.txt'
    data = np.genfromtxt(itk, delimiter=',', skip_header=1)

    t, at = data[:,0] , data[:,1]
    dt = t[1] - t[0]


    plt.plot(t, BaseLineCorrection(at,0.01, 'spline', 2,1000), 'green', lw=lw)
    plt.plot(t, at, 'red', lw=lw)
    plt.show()    





    # # vo = -0.029529
    # # uo = -0.005465
    # v0 = 0
    # u0 = 0

    # #Data Original
    # vt = integrate.cumtrapz(at, dx=dt, initial=0) + v0
    # plt.plot(t, v_original, 'red', lw=lw)
    # plt.plot(t, vt, 'blue', lw=lw)
    # plt.show()

    # ut = integrate.cumtrapz(vt, dx=dt, initial=0) + u0
    # plt.plot(t, u_original, 'red', lw=lw)
    # plt.plot(t, ut, 'blue', lw=lw)
    # plt.show()


    # ### TEST ###
    # ## Paso 1
    # w = 1
    # vt = integrate.cumtrapz(at, dx=dt, initial=0) + v0
    # #Pvt = parzem_smoothing(vt, w) ####################################################################################
    # Pvt = linear(t, vt)
    # #Pvt = least_squares(t, vt, w)
    # # Pvt = smooth(vt, w)

    # plt.title('Vt vs Smooth')
    # plt.plot(t, vt, 'blue', lw=lw)
    # plt.plot(t, Pvt, 'blue', lw=lw)
    # plt.plot(t, v_original, 'green', lw=lw)
    # plt.plot(t, vt - Pvt, 'orange', lw=lw)
    # plt.show()

    # at = at - np.gradient(Pvt, dt)  # 1er correcion


    # # Comparacion Aceleracion corregida por velocidad
    # plt.title('1era correcion')
    # plt.plot(t, a_original, 'red', lw=lw, label='Original')
    # plt.plot(t, at, 'blue', lw=lw, label='Corregida 1')
    # plt.legend()
    # plt.show()