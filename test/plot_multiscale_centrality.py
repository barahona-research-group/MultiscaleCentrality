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
rev 		  = False # for directed graph, reverse the original direction of the edges if True

if rev:
    os.chdir(graph_tpe+'_rev_'+normalization_tpe)
else:
    os.chdir(graph_tpe+'_'+normalization_tpe)


#random walk parameters
t_min 	  = params['t_min']    # maximum time (in spectral gap units)
t_max 	  = params['t_max']    # maximum time (in spectral gap units)
n_t 	  = params['n_t']  # number of timesteps 

#create the object
G, pos  = generate_graph(tpe=graph_tpe, params = params)
mc = Multiscale_Centrality(G, pos, t_min = t_min, t_max = t_max, n_t = n_t, normalization_tpe = normalization_tpe, rw_tpe = rw_tpe)

mc.load_centralities()

mc.figsize=(6,3)
mc.compare_centralities(n_compare = n_t, n_top = 2, n_force = 100)
mc.plot_comparisons_spearman()

# plot the multiscale centralities as a function of scale
mc.plot_trajectories()

#make video
n_plot = 50
mc.figsize = (5,3)
mc.labels = False
mc.plot_edges = True 
mc.video_multiscale(n_plot = n_plot, node_size=100)


