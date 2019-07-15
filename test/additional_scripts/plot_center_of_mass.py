import networkx as nx
import pylab as plt
import numpy as np
import pickle as pickle
import sys as sys
import os as os
import yaml as yaml

from multiscale_centrality import Multiscale_Centrality
from graph_generator import generate_graph

graph_tpe = sys.argv[-1]

params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]

normalization_tpe = 'combinatorial'  # type of normalisation, combinatorial or normalized
rw_tpe 		  = 'continuous'     # discrete or continuous random walk
#os.chdir(graph_tpe+'_rev_'+normalization_tpe)
os.chdir(graph_tpe+'_'+normalization_tpe)

#random walk parameters
t_min 	  = params['t_min']    # maximum time (in spectral gap units)
t_max 	  = params['t_max']    # maximum time (in spectral gap units)
n_t 	  = params['n_t']  # number of timesteps 

print(t_min,t_max)

#create the object
G, pos  = generate_graph(tpe=graph_tpe, params = params)
mc = Multiscale_Centrality(G, pos, t_min = t_min, t_max = t_max, n_t = n_t, normalization_tpe = normalization_tpe, rw_tpe = rw_tpe)

mc.load_centralities()

mc.figsize=(6,3)

mc.figsize = (5,4)
mc.labels = False
mc.plot_edges =True



A = nx.adjacency_matrix(mc.G).todense()
mass = np.array(A.sum(1).flatten(),dtype=np.float32)
mass /= mass.sum()
pos = np.array(mc.pos)

cm_x = (mass*pos[:,0]).sum()
cm_y = (mass*pos[:,1]).sum()

print(cm_x, cm_y)


mc.plot_multiscale_centrality(6,node_size=100)
plt.plot([cm_x,], [cm_y,], 'om', ms= 10)

plt.savefig('center_of_mass_initial.svg', bbox_inches='tight')

mc.plot_multiscale_centrality(-1,node_size=100)
plt.plot([cm_x,], [cm_y,], 'om', ms= 10)

plt.savefig('center_of_mass_final.svg', bbox_inches='tight')

