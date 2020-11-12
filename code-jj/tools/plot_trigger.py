import matplotlib.pylab as plt
import numpy as np
from obspy.signal.trigger import trigger_onset #triggerOnset
import threading


def plot_trigger(tr, cft, thr1, thr2):
    df, npts = tr.stats.sampling_rate, tr.stats.npts
    t = np.arange(npts,dtype='float32')/df
    fig = plt.figure(1)
    #fig = plt.figure(1, figsize=(8, 4))
    fig.clf()
    ax = fig.add_subplot(211)
    ax.plot(t, tr.data, 'black')
    ax2 = fig.add_subplot(212,sharex=ax)
    ax2.plot(t, cft, 'black')
    onof = np.array(trigger_onset(cft, thr1, thr2)) #np.array(triggerOnset(cft, thr1, thr2))
    i,j = ax.get_ylim()
    try:
        ax.vlines(onof[:,0]/df, i, j, color='red', lw = 2)
        ax.vlines(onof[:,1]/df, i, j, color='blue', lw = 2)
    except IndexError:
        pass
    ax2.axhline(thr1, color='red', lw = 1, ls = '--')
    ax2.axhline(thr2, color='blue', lw = 1, ls = '--')
    fig.canvas.draw()
    plt.show()

def plot_threaded(tr, cft, thr1, thr2):
    thread = threading.Thread(target=plot_trigger, args=(tr, cft, thr1, thr2))
    thread.start()

if __name__ == '__main__':
    from obspy.core import read
    from obspy.signal.trigger import plot_trigger
    from obspy.signal.trigger import classic_sta_lta
    import wave


    CIIFIC = wave.Waves()
    # CIIFIC.loadWaves_new('D:/SHM/code-jj/15-01-2020')
    CIIFIC.loadWaves_old('D:/SHM/code-jj/2020-11-02_2020-11-02')

    CIIFIC.baseLine('spline', 1, 100)





    # trace = read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
    # df = trace.stats.sampling_rate
    # cft = classic_sta_lta(trace.data, int(5 * df), int(10 * df))
    # plot_trigger(trace, cft, 1.5, 0.5)