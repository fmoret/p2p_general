# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:40:18 2016

@author: fmoret
"""
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from Prosumer import Prosumer

dd1 = os.getcwd()
data_path = dd1+r'\Input Data 2'

#%%
filename0=data_path+r'\price.csv'
el_price = pd.read_csv(filename0,index_col=[0])['Market price ($/MWh)']

filename1=data_path+r'\info.csv'
info = pd.read_csv(filename1,index_col=[0])
info.at[info.index[-1],'Pmin'] = -np.inf
info.at[info.index[-1],'Pmax'] = np.inf

filename2=data_path+r'\load_min.csv'
load_min = pd.read_csv(filename2,index_col=[0], skiprows = [1])

filename3=data_path+r'\load_max.csv'
load_max = pd.read_csv(filename3,index_col=[0], skiprows = [1])

filename4=data_path+r'\ren.csv'
ren = pd.read_csv(filename4,index_col=[0], skiprows = [1])

filename5=data_path+r'\network.csv'
network = pd.read_csv(filename5,index_col=[0])

time = el_price.index
#%%
num_agents = info.shape[0]
num_prod = sum(info['Type']=='Producer')
num_cons = sum(info['Type']=='Consumer')
TMST = load_max.shape[0]

cons = info[info['Type']=='Consumer'].index
prod = info[info['Type']=='Producer'].index
prod_ren = info[(info['Type']=='Producer') & ((info['Energy']=='Wind') | (info['Energy']=='Solar'))].index
prod_conv = info[(info['Type']=='Producer') & ~((info['Energy']=='Wind') | (info['Energy']=='Solar'))].index
g = len(prod_conv)
n = num_agents
h = 48

el_price_e = el_price
tau = 10

a = info['a']
b = info['b']

Pmin = info['Pmin']
Pmax = info['Pmax']


#%%
comm = info['Community'].dropna().unique()
p2p = info['p2p'].dropna().unique()

n=0
comm_ag = []
for i in comm:
    comm_ag += [info[info['Community']==i].index.unique()]
    n += 1+len(info[info['Community']==i].index.unique())
    
p2p_ag = []
for i in p2p:
    p2p_ag += [info[info['p2p']==i].index.unique()]
    n += len(info[info['p2p']==i].index.unique())

n += 1 # +1 is for accounting grid
    
ID_CM = []
ID_comm = []
ID_p2p = []

n_comm = len(comm_ag)
n_p2p = len(p2p_ag)

inc = np.zeros([n,n,n_comm+n_p2p])

idx_ag = 0
idx_mrk = 0
for c in [len(comm_ag[i]) for i in range(n_comm)]:
    inc[idx_ag,idx_ag+1:idx_ag+c+1,idx_mrk] = 0.01
    inc[idx_ag+1:idx_ag+c+1,idx_ag,idx_mrk] = 0.01
    ID_CM += [idx_ag]
    ID_comm += [np.arange(idx_ag+1,idx_ag+c+1)]
    idx_ag += c+1
    idx_mrk += 1

for c in [len(p2p_ag[i]) for i in range(n_p2p)]:    
    inc[idx_ag:idx_ag+c,idx_ag:idx_ag+c,idx_mrk] = np.ones([c,c]) - np.eye(c)
    ID_p2p += [np.arange(idx_ag,idx_ag+c)]
    idx_ag += c
    idx_mrk +=1
    
#Assigning all community managers to the last p2p market
for idx in ID_CM:
    inc[idx,ID_p2p[-1],idx_mrk-1] = 1
    inc[ID_p2p[-1],idx,idx_mrk-1] = 1

#Assigning grid to last p2p market
ID_grid = n-1
grid_ag = info.index[-1]
for idx in ID_CM:
    inc[-1,idx,idx_mrk-1] = 10
    inc[idx,-1,idx_mrk-1] = 10
inc[-1,ID_p2p[-1],idx_mrk-1] = 10
inc[ID_p2p[-1],-1,idx_mrk-1] = 10


#%%
rho = 10
#Init Prosumer classes 
pros = {}
# Community
for c in range(n_comm):
    pros[ID_CM[c]] = Prosumer(inc[ID_CM[c],:,:], rho = rho)
    j = 0
    for i in ID_comm[c]:
        pros[i] = Prosumer(inc[i,:,:], data = info[info.index==comm_ag[c][j]], rho = rho)
        j += 1
# p2p
for c in range(n_p2p):
    j = 0
    for i in ID_p2p[c]:
        pros[i] = Prosumer(inc[i,:,:], data = info[info.index==p2p_ag[c][j]], rho = rho)
        j += 1
# Grid
pros[ID_grid] = Prosumer(inc[ID_grid,:,:], data = info[info.index==grid_ag], rho = rho)

display = True
T = np.zeros([n,n,len(comm_ag)+len(p2p_ag)])
T_sum = 100*np.ones([n,n,len(comm_ag)+len(p2p_ag)])


k = 0
while abs(T_sum).max() > 1e-3:
    k += 1
    temp = np.copy(T)
    for i in range(n):
        temp[i,:,:] = pros[i].optimize(T[:,i,:])
    
    T = np.copy(temp)
    for m in range(n_comm+n_p2p):
        T_sum[:,:,m] = T[:,:,m] + T[:,:,m].T

    
if display:
    lim = 100
    plt.figure()
    ax = plt.axes(projection='3d')
    trades = inc.nonzero()
    points = ax.scatter3D(trades[0], trades[2], trades[1], c=T[trades], edgecolors = None, cmap='seismic', vmin=-lim, vmax=lim)



SW = 0
for i in range(n):
    SW += pros[i].model.objVal

print('Social Welfare with rho =', rho, ' is SW =', SW)
