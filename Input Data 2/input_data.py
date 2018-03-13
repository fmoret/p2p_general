# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:10:47 2017

@author: fmoret
"""

import os
#import math
import numpy as np
#import csv
path =  'C:/Users/fabio/OneDrive - Danmarks Tekniske Universitet/PhD/Peer2Peer/Input Data/'
#os.chdir(path)
# import csv data as dataframe
import pandas as pd
#data = pd.DataFrame.from_csv('Data.csv',index_col=[4,0,3],parse_dates =False)
data = pd.read_csv(path+'Data.csv',index_col=[4,0,3],parse_dates =True,dayfirst=True)
el_price_12 = pd.DataFrame.from_csv('stem-summary-2012.csv',index_col=[0,2])
el_price_13 = pd.DataFrame.from_csv('stem-summary-2013.csv',index_col=[0,2])
full_el_price = pd.concat([el_price_12, el_price_13])

date = np.array(data.index.get_level_values('date').date)
min_day = min(date)
max_day = max(date)

tot_el_price = full_el_price[(full_el_price.index.get_level_values('Trading Date').date >= min_day) & (full_el_price.index.get_level_values('Trading Date').date <= max_day)]
h_id = np.array(tot_el_price.index.get_level_values('Trading Interval').hour)

#%%
cl_id = np.array([13,14,20,33,35,38,39,56,
                  69,73,74,75,82,87,88,101,104,
                  106,109,110,119,124,130,137,141,144,
                  152,153,157,169,176,184,188,189,
                  193,201,202,204,206,207,210,211,212,
                  214,218,244,246,253,256,273,276,297])

PostCode = np.empty(52, dtype=int)
Load = np.empty((17520,52))
Flex_load = np.empty((17520,52))
PV = np.empty((17520,52))
for i in range(52):
    ddd = data[data.index.get_level_values('Customer')==cl_id[i]]
    PostCode[i] = ddd['Postcode'][[0]]
    lll = np.array(ddd[ddd.index.get_level_values('Consumption Category')=='GC'])[:,range(2,50)]
    Load[:,i] = lll.reshape(17520)
    fff = np.array(ddd[ddd.index.get_level_values('Consumption Category')=='CL'])[:,range(2,50)]
    if fff.size:
        fff2 = np.copy(fff)
        for t in range(365):
            fff2[t,:] = sum(fff[t,:])/48*np.ones(48)
        Flex_load[:,i] = fff2.reshape(17520)
    else:
        Flex_load[:,i] = np.zeros(17520)
    ppp = np.array(ddd[ddd.index.get_level_values('Consumption Category')=='GG'])[:,range(2,50)]
    PV[:,i] = ppp.reshape(17520)

#%%
test_id = np.array([0,8,46,50,51,13,14,19,28,39,10,12,18,37,40])
test_PostCode = PostCode[test_id]
test_Load = Load[range(0, 17520, 2),:][:,test_id]
test_Flex_load = Flex_load[range(0, 17520, 2),:][:,test_id]
test_PV = PV[range(0, 17520, 2),:][:,test_id]
    
#%%
scale_price = tot_el_price.as_matrix()[:,5]/1000               #from €/MWh to €/kWh
price_night =  scale_price[(h_id < 6)]                         #night => h < 06:00
price_morning =  scale_price[(h_id >= 6) & (h_id < 12)]        #morning => 06:00 <= h < 12:00
price_afternoon =  scale_price[(h_id >= 12) & (h_id < 18)]     #afternoon => h < 12:00 <= h < 18:00
price_evening =  scale_price[(h_id >= 18)]                     #evening => h >= 18:00

mu_price = np.array([price_night.mean(), price_morning.mean(), price_afternoon.mean(), price_evening.mean()])
sd_price = np.array([price_night.std(), price_morning.std(), price_afternoon.std(), price_evening.std()])

#%%  Prosumer data
n = 15

b1 = np.zeros((n,24))
c1 = np.zeros((n,24))

b1[[0,5,10],0:6] = np.random.normal(0.5*mu_price[0], sd_price[0]/3, ((3,6)))
c1[[0,5,10],0:6] = abs(np.random.normal(0, sd_price[0]/3, ((3,6))))
b1[[2,7,12],0:6] = np.random.normal(mu_price[0], sd_price[0]/3, ((3,6)))
c1[[2,7,12],0:6] = abs(np.random.normal(0, sd_price[0]/3, ((3,6))))

b1[[0,5,10],6:12] = np.random.normal(0.5*mu_price[1], sd_price[1]/3, ((3,6)))
c1[[0,5,10],6:12] = abs(np.random.normal(0, sd_price[1]/3, ((3,6))))
b1[[2,7,12],6:12] = np.random.normal(1.5*mu_price[1], sd_price[1]/3, ((3,6)))
c1[[2,7,12],6:12] = abs(np.random.normal(0, sd_price[1]/3, ((3,6))))

b1[[0,5,10],12:18] = np.random.normal(0.5*mu_price[2], sd_price[2]/3, ((3,6)))
c1[[0,5,10],12:18] = abs(np.random.normal(0, sd_price[2]/3, ((3,6))))
b1[[2,7,12],12:18] = np.random.normal(1.5*mu_price[2], sd_price[2]/3, ((3,6)))
c1[[2,7,12],12:18] = abs(np.random.normal(0, sd_price[2]/3, ((3,6))))

b1[[0,5,10],18:24] = np.random.normal(0.5*mu_price[3], sd_price[3]/3, ((3,6)))
c1[[0,5,10],18:24] = abs(np.random.normal(0, sd_price[3]/3, ((3,6))))
b1[[2,7,12],18:24] = np.random.normal(1.5*mu_price[3], sd_price[3]/3, ((3,6)))
c1[[2,7,12],18:24] = abs(np.random.normal(0, sd_price[3]/3, ((3,6))))

Pmin = np.zeros(n)
Pmin[[0,5,10]] = np.random.uniform(low=0, high=1, size=3) 
Pmax = np.zeros(n)
Pmax[[0,5,10]] = Pmin[[0,5,10]] + np.random.uniform(low=1, high=2, size=3)
Pmax[[2,7,12]] = np.random.uniform(low=0, high=1, size=2)

b2 = np.zeros((n,24))
c2 = np.zeros((n,24))

b2[:,0:6] = np.random.normal(mu_price[0], sd_price[0]/3, ((n,6)))
c2[:,0:6] = abs(np.random.normal(0, sd_price[0]/3, ((n,6))))

b2[:,6:12] = np.random.normal(mu_price[1], sd_price[1]/3, ((n,6)))
c2[:,6:12] = abs(np.random.normal(0, sd_price[1]/3, ((n,6))))

b2[:,12:18] = np.random.normal(mu_price[2], sd_price[2]/3, ((n,6)))
c2[:,12:18] = abs(np.random.normal(0, sd_price[2]/3, ((n,6))))

b2[:,18:24] = np.random.normal(mu_price[3], sd_price[3]/3, ((n,6)))
c2[:,18:24] = abs(np.random.normal(0, sd_price[3]/3, ((n,6))))

#%%
bb2 = np.empty(n)
cc2 = np.empty(n)
bb1 = np.empty(n)
cc1 = np.empty(n)


date_list = np.unique(date)
for day in date_list:
    bb2 = np.row_stack((bb2,np.transpose(b2)))
    cc2 = np.row_stack((cc2,np.transpose(c2)))
    bb1 = np.row_stack((bb1,np.transpose(b1)))
    cc1 = np.row_stack((cc1,np.transpose(c1)))
    
b2 = bb2[range(1,8761),:]
c2 = cc2[range(1,8761),:]

b1 = bb1[range(1,8761),:]
c1 = cc1[range(1,8761),:]

#%%  Save variables for further use
import shelve
filename='input_data_2.out'
d = shelve.open(filename, 'n')

d['data'] = data
d['date'] = date
d['tot_el_price'] = tot_el_price

d['b1'] = b1
d['c1'] = c1
d['Pmin'] = Pmin
d['Pmax'] = Pmax

d['b2'] = b2
d['c2'] = c2

d['PostCode'] = test_PostCode
d['Load'] = test_Load
d['Flex_load'] = test_Flex_load
d['PV'] = test_PV

##%%
#import scipy.io
#mdict = {'b1': b1, 'c1': c1, 'b2': b2, 'c2': c2, 'Pmin': Pmin, 'Pmax': Pmax, 'el_price': tot_el_price.as_matrix()[:,5][0::2]/1000}
#scipy.io.savemat('mdat.mat', mdict)