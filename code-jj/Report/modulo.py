from ctypes import *
import numpy as np
test = CDLL('./mod.so')

def resSpe (dt, dr, ag, ts):
    """
    desc
    """
    dt_c = c_float(dt)
    dr_c = c_float(dr)
    ag_c = ag.astype(c_float)
    na_c = c_int(len(ag))
    ts_c = ts.astype(c_float)
    nt_c = c_int(len(ts))
    sr_c = np.zeros((3,len(ts)), dtype=c_float)
    test.res_esp_(byref(dt_c), byref(dr_c),ag_c.ctypes.data_as(POINTER(c_float)), byref(na_c),ts_c.ctypes.data_as(POINTER(c_float)), byref(nt_c), sr_c.ctypes.data_as(POINTER(c_float)))
    sr = sr_c.astype(float)
    sr = np.transpose(sr)
    return sr