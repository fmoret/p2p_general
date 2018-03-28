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
import networkx as nx
from Prosumer import Prosumer
dd1 = os.getcwd()

#%% Custom functions
def create_comm(G, size, num, data=None):
    comm = 'C'+str(num)+'_'
    CM = comm+'cm'
    G.add_node(CM, ID=CM, a=0, b=0, Pmin=0, Pmax=0, num_ass=0)
    if data is not None:    
        comm_ag = data[data['Community']==num].index.unique()
        size = len(comm_ag)
        
    idx = [CM]
    for i in range(size):
        node = comm+str(i)
        idx += [node]
        if data is not None:
            G.add_node(node, ID = comm_ag[i],
                       a = data[data.index==comm_ag[i]]['a'].values,
                       b = data[data.index==comm_ag[i]]['b'].values,
                       Pmin = data[data.index==comm_ag[i]]['Pmin'].values,
                       Pmax = data[data.index==comm_ag[i]]['Pmax'].values,
                       num_ass = len(data[data.index==comm_ag[i]]['a'].values) )
        else:
            G.add_node(node)
            
        G.add_edge(CM, node, key='comm', pref=0.001)
        G.add_edge(CM, node, key='imp', pref=0.1)
        G.add_edge(CM, node, key='exp', pref=0.1)
        
    return idx  #G.subgraph(idx), 
        

def create_p2p(G, size, num, data=None):
    p2p = 'P'+str(num)+'_'
    if data is not None:    
        p2p_ag = info[info['p2p']==num].index.unique()
        size = len(p2p_ag)
        
    idx = []
    for i in range(size):
        node = p2p+str(i)
        idx += [node]
        if data is not None:
            G.add_node(node, ID = p2p_ag[i],
                       a = data[data.index==p2p_ag[i]]['a'].values,
                       b = data[data.index==p2p_ag[i]]['b'].values,
                       Pmin = data[data.index==p2p_ag[i]]['Pmin'].values,
                       Pmax = data[data.index==p2p_ag[i]]['Pmax'].values,
                       num_ass = len(data[data.index==p2p_ag[i]]['a'].values) )
        else:
            G.add_node(node)
            
    for i in range(len(idx)):
        for j in idx[i+1:]:
            G.add_edge(idx[i], j, key='p2p', pref=1)
            
    return idx #G.subgraph(idx), 

def create_grid(G, nodes=[], data=None):
    if data is not None:
        G.add_node('grid', ID='grid',
                   a = [0],
                   b = data[data['Type']=='Grid']['b'].values,
                   Pmin = [-np.inf],
                   Pmax = [np.inf],
                   num_ass = 1)
    else:
        G.add_node('grid')
        
    for i in nodes:
        G.add_edge('grid', i, key='imp', pref=10)
        G.add_edge('grid', i, key='exp', pref=-10)
            
    return ['grid']


def connect_to_node(G, nodes_from, node_to):
    for n in nodes_from:
        G.add_edge(node_to, n, key='p2p', pref=1)
        

#%% Load data
data_path = dd1+r'\Input Data 2'
filename1=data_path+r'\info.csv'
info = pd.read_csv(filename1,index_col=[0])
info['Pmax'].fillna(np.inf, inplace = True)
info['Pmin'].fillna(-np.inf, inplace = True)

#%%
a = info['a']
b = info['b']

Pmin = info['Pmin']
Pmax = info['Pmax']

imp = info['Type'].str.contains('Import', case=False).any()
exp = info['Type'].str.contains('Export', case=False).any()
comm = np.array(info['Community'].dropna().unique(), dtype=int)
p2p = np.array(info['p2p'].dropna().unique(), dtype=int)

#Initialize graph
try:
    del G
    G = nx.MultiGraph()
except NameError:
    G = nx.MultiGraph()
    
#Create community and p2p layers
G_comm = [create_comm(G, size = 0, num = i, data = info) for i in comm]
G_p2p = [create_p2p(G, size = 0, num = i, data = info) for i in p2p]

#Connect community managers to p2p layer
[connect_to_node(G, G_p2p[0], G_comm[i][0]) for i in range(len(G_comm))]

#Create grid node
to_grid = [item[0] for item in G_comm] + [item for sublist in G_p2p for item in sublist]
G_grid = create_grid(G, to_grid, data=info)

#Draw graph
plt.figure()
nx.draw(G)

nodes = [item for sublist in G_comm for item in sublist] + [item for sublist in G_p2p for item in sublist] + G_grid

#%%
inc = np.zeros([len(nodes),len(nodes),4])
G_0 = G.copy()
for u,v,key,data in G.edges(data=True,keys=True):
    if key != 'comm':
        G_0.remove_edge(u,v,key=key)
G_1 = G.copy()
for u,v,key,data in G.edges(data=True,keys=True):
    if key != 'p2p':
        G_1.remove_edge(u,v,key=key)
G_2 = G.copy()
for u,v,key,data in G.edges(data=True,keys=True):
    if key != 'imp':
        G_2.remove_edge(u,v,key=key)
G_3 = G.copy()
for u,v,key,data in G.edges(data=True,keys=True):
    if key != 'exp':
        G_3.remove_edge(u,v,key=key)

inc[:,:,0] = nx.to_numpy_array(G_0, weight='pref')
inc[:,:,1] = nx.to_numpy_array(G_1, weight='pref')
inc[:,:,2] = nx.to_numpy_array(G_2, weight='pref')
inc[:,:,3] = nx.to_numpy_array(G_3, weight='pref')
#%%
rho = 100

#Init Prosumer classes 
pros = {}
# Community
for i in range(len(nodes)):
    pros[nodes[i]] = Prosumer(inc[i,:,:], data = G.node[nodes[i]], rho = rho)

display = True
T = np.zeros(inc.shape)
T_sum = 100*np.ones(inc.shape)


k = 0
while abs(T_sum).max() > 1e-3 and k<1000:
    k += 1
    temp = np.copy(T)
    for i in range(len(nodes)):
        temp[i,:,:] = pros[nodes[i]].optimize(T[:,i,:])
    
    T = np.copy(temp)
    for m in range(T.shape[2]):
        T_sum[:,:,m] = T[:,:,m] + T[:,:,m].T

    
if display:
    lim = abs(T).max()
    plt.figure()
    ax = plt.axes(projection='3d')
    trades = inc.nonzero()
    points = ax.scatter3D(trades[0], trades[2], trades[1], c=T[trades], edgecolors = None, cmap='coolwarm', vmin=-lim, vmax=lim)
    plt.colorbar(points)


SW = 0
for n in nodes:
    SW += pros[n].model.objVal

print('Social Welfare with rho =', rho, ' is SW =', SW)
