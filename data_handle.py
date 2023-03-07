#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from archv2 import *


# In[ ]:
def gen_data_arr(tarch, arr,carr):
    data_arr = {'N':tarch.N,'Nlevel':tarch.Nlevel,'Ej':tarch.Ej, 'Ec':tarch.Ec,'spectrum':arr,'cspectrum':carr}
    return data_arr

def dump_data(data_arr,file_name="transmondata"):
    pickling_on = open(file_name + ".pickle","wb")
    pickle.dump(data_arr, pickling_on)
    pickling_on.close()

def load_data(file_name):
    pickle_off = open(file_name, 'rb')
    arr = pickle.load(pickle_off)
    return arr


# In[ ]:


def unpack_data(filename):
    #format [t,eval,evec,label]
    data_arr = load_data(filename)
    tarch = tmon_fn(data_arr['N'],data_arr['Nlevel'])
    tarch.Ej = data_arr['Ej']
    tarch.Ec = data_arr['Ec']
    tlist = []
    for spec in data_arr['spectrum']:
        tlist.append(spec[0])
    arr= data_arr['spectrum']
    carr = data_arr['cspectrum']
    
    return tarch, tlist, arr, carr

