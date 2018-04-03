# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:40:18 2016

@author: fmoret
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from Prosumer import Prosumer
dd1 = os.getcwd()

#%% Custom functions
def create_comm(G, size, num, num_comm, data=None):
    comm = 'C'+str(num)+'_'
    CM = comm+'cm'
    IE = comm+'ie'
    x_center = (100/num_comm)*(num - 1 + 0.5)
    G.add_node(CM, ID='C'+str(num), ntype = 'CM', a=0, b=0, Pmin=0, Pmax=0, num_ass=0, pos = (x_center, 10))
    G.add_node(IE, ID='C'+str(num), ntype = 'IE', a=0, b=0, Pmin=0, Pmax=0, num_ass=0, pos = (x_center, 30))
    if data is not None:    
        comm_ag = data[data['Community']==num].index.unique()
        size = len(comm_ag)
        
    idx = [IE, CM]
    x_comm = 100/num_comm/(size+1)*np.arange(1,size+1) + (num-1)*100/num_comm
    for i in range(size):
        node = comm+str(i)
        idx += [node]
        if data is not None:
            G.add_node(node, ID = comm_ag[i], ntype = 'C_node',
                       a = data[data.index==comm_ag[i]]['a'].values,
                       b = data[data.index==comm_ag[i]]['b'].values,
                       Pmin = data[data.index==comm_ag[i]]['Pmin'].values,
                       Pmax = data[data.index==comm_ag[i]]['Pmax'].values,
                       num_ass = len(data[data.index==comm_ag[i]]['a'].values),
                       pos = (x_comm[i],20) )
        else:
            G.add_node(node)
            
        G.add_edge(CM, node, key='comm', pref=0.01)
        G.add_edge(IE, node, key='ie', pref=1)
        
    return idx  #G.subgraph(idx), 
        

def create_p2p(G, size, num, num_p2p, data=None):
    p2p = 'P'+str(num)+'_'
    if data is not None:    
        p2p_ag = info[info['p2p']==num].index.unique()
        size = len(p2p_ag)
        
    idx = []
    for i in range(size):
        node = p2p+str(i)
        idx += [node]
        if data is not None:
            G.add_node(node, ID = p2p_ag[i], ntype = 'P_node',
                       a = data[data.index==p2p_ag[i]]['a'].values,
                       b = data[data.index==p2p_ag[i]]['b'].values,
                       Pmin = data[data.index==p2p_ag[i]]['Pmin'].values,
                       Pmax = data[data.index==p2p_ag[i]]['Pmax'].values,
                       num_ass = len(data[data.index==p2p_ag[i]]['a'].values), )
        else:
            G.add_node(node)

    x_center = (100/num_p2p)*(num - 1 + 0.5) 
    pos = nx.spring_layout(G.subgraph(idx), scale=100/num_p2p/3, center=(x_center,60)) #y = 40-80

    y_max = np.array(list(pos.values()))[:,1].max()
    y_min = np.array(list(pos.values()))[:,1].min()
    
    for p in pos.keys():
        y_new = 40 + (80 - 40)*(pos[p][1] - y_min)/(y_max - y_min)
        x_new = pos[p][0]
        G.node[p]['pos'] = (x_new, y_new)
    
            
    for i in range(len(idx)):
        for j in idx[i+1:]:
            G.add_edge(idx[i], j, key='p2p', pref=1)
            
    return idx #G.subgraph(idx), 

def create_grid(G, nodes=[], data=None):
    if data is not None:
        G.add_node('grid', ID='G', ntype = 'G',
                   a = [0],
                   b = data[data['Type']=='Grid']['b'].values,
                   Pmin = [-np.inf],
                   Pmax = [np.inf],
                   num_ass = 1,
                   pos = (50,90) )
    else:
        G.add_node('grid')
        
    for i in nodes:
        G.add_edge('grid', i, key='ie', pref=10)
            
    return ['grid']


def connect_to_node(G, nodes_from, node_to):
    for n in nodes_from:
        G.add_edge(node_to, n, key='p2p', pref=1)
        
def node_colors(G):
    dic = {}
    for n in G.nodes:
        idx = G.node[n]['ntype']
        if idx == 'CM':
            dic[n] = 'b'
        elif idx == 'IE':
            dic[n] = 'g'
        elif idx == 'C_node':
            dic[n] = 'c'
        elif idx == 'P_node':
            dic[n] = 'g'
        elif idx == 'G':
            dic[n] = 'r'
    return dic

def edge_trades(G, T, nodelist):
    nodelist = np.array(nodelist)
    e_lab = {}
    for e in G.edges:
        idx = list(e)
        i = np.where(nodelist==idx[0])[0][0]
        j = np.where(nodelist==idx[1])[0][0]
        G.edges[e]['data'] = T[i,j]
        e_lab[e] = round(T[i,j], 3)
    return G, nx.get_edge_attributes(G,'data'), e_lab
        
def draw_graph(G_init, T, nodelist, comm, p2p):
    G = nx.DiGraph()
    G.add_nodes_from(G_init.nodes(data=True))
    G.add_edges_from(G_init.edges(data=True))
    pos = nx.get_node_attributes(G,'pos')
    lab = nx.get_node_attributes(G,'ID')
    n_col = node_colors(G)
    G, e_data, e_lab = edge_trades(G, T, nodelist)
    
    plt.figure()
    nx.draw_networkx_nodes(G, pos, node_color=list(n_col.values()), labels=lab, alpha = 0.7)
    nx.draw_networkx_labels(G,pos,lab,font_size=11)
    edges = nx.draw_networkx_edges(G, pos, edge_color=list(e_data.values()), width=1, edge_cmap=plt.cm.coolwarm)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=e_lab)
    plt.colorbar(edges)
    plt.axis('off')
    
    return G
    
#    jet = plt.get_cmap('coolwarm') 
#    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
#    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#    
#    nx.draw_networkx(G, pos=pos, labels=lab, node_color=list(n_col.values()), edge_cmap='coolwarm', edge_vmin=T.min(), edge_vmax=T.max())
    

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
    G = nx.Graph()
except NameError:
    G = nx.Graph()
    
#Create community and p2p layers
G_comm = [create_comm(G, size = 0, num = i, num_comm = len(comm), data = info) for i in comm]
G_p2p = [create_p2p(G, size = 0, num = i, num_p2p = len(p2p), data = info) for i in p2p]

#Connect community managers to p2p layer
[connect_to_node(G, G_p2p[0], G_comm[i][0]) for i in range(len(G_comm))]

#Create grid node
to_grid = [item[0] for item in G_comm] + [item for sublist in G_p2p for item in sublist]
G_grid = create_grid(G, nodes=to_grid, data=info)

nodes = [item for sublist in G_comm for item in sublist] + [item for sublist in G_p2p for item in sublist] + G_grid

#%%
inc = np.zeros([len(nodes),len(nodes),4])
G_0 = G.copy()
for u,v,data in G.edges(data=True):
    if data['key'] != 'comm':
        G_0.remove_edge(u,v)
G_1 = G.copy()
for u,v,data in G.edges(data=True):
    if data['key'] != 'p2p':
        G_1.remove_edge(u,v)
G_2 = G.copy()
for u,v,data in G.edges(data=True):
    if data['key'] != 'ie':
        G_2.remove_edge(u,v)

nodelist = list(G.nodes())
inc[:,:,0] = nx.to_numpy_array(G_0, weight='pref', nodelist=nodelist)
inc[:,:,1] = nx.to_numpy_array(G_1, weight='pref', nodelist=nodelist)
inc[:,:,2] = nx.to_numpy_array(G_2, weight='pref', nodelist=nodelist)

#%%
rho = 100

#Init Prosumer classes 
pros = {}
# Community
for i in range(len(nodelist)):
    pros[nodelist[i]] = Prosumer(inc[i,:,:], data = G.node[nodelist[i]], rho = rho)

display = True
T = np.zeros(inc.shape)
T_sum = 100*np.ones(inc.shape)


k = 0
while abs(T_sum).max() > 1e-3 and k<1000:
    k += 1
    temp = np.copy(T)
    for i in range(len(nodelist)):
        temp[i,:,:] = pros[nodelist[i]].optimize(T[:,i,:])
    
    T = np.copy(temp)
    for m in range(T.shape[2]):
        T_sum[:,:,m] = T[:,:,m] + T[:,:,m].T

#%%
    
if display:
#    lim = abs(T).max()
#    plt.figure()
#    ax = plt.axes(projection='3d')
#    trades = inc.nonzero()
#    points = ax.scatter3D(trades[0], trades[2], trades[1], c=T[trades], edgecolors = None, cmap='coolwarm', vmin=-lim, vmax=lim)
#    plt.colorbar(points)

    H = draw_graph(G, T.sum(axis=2), nodelist, comm, p2p)

SW = 0
for n in nodes:
    SW += pros[n].model.objVal

print('Social Welfare with rho =', rho, ' is SW =', SW)
