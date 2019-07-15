import networkx as nx
import sys as sys
import os as os
import pickle as pickle
import yaml as yaml

from multiscale_centrality import Multiscale_Centrality
from graph_generator import generate_graph

################
## parameters ##
################

#which graph to use:
#graph_tpe = 'miserable' 
graph_tpe = sys.argv[-1]

normalization_tpe = 'combinatorial'  # type of normalisation, combinatorial or normalized
rw_tpe 		  = 'continuous'     # discrete or continuous random walk
rev 		  = False # for directed graph, reverse the original direction of the edges if True

#plotting parameters
n_plot    = 50 #number of plots when scanning the time horizon

#number of cpu to use for parallel computations
n_processes = 2

###############
## load data ## 
###############

params = yaml.load(open('graph_params.yaml','rb'))[graph_tpe]
print('Used parameters:', params)

#random walk parameters
t_min 	  = params['t_min']    # maximum time (in spectral gap units)
t_max 	  = params['t_max']    # maximum time (in spectral gap units)
n_t 	  = params['n_t']  # number of timesteps 


if rev:
    if not os.path.isdir(graph_tpe+'_rev_'+normalization_tpe):
        os.mkdir(graph_tpe+'_rev_'+normalization_tpe)
else:
    if not os.path.isdir(graph_tpe+'_'+normalization_tpe):
        os.mkdir(graph_tpe+'_'+normalization_tpe)


if rev:
    os.chdir(graph_tpe+'_rev_'+normalization_tpe)
else:
    os.chdir(graph_tpe+'_'+normalization_tpe)

pickle.dump([t_min, t_max, n_t, n_plot], open('simu_params.pkl','wb'))


G, pos  = generate_graph(tpe=graph_tpe, params = params)

##################
## run the code ##
##################

#create the object
mc = Multiscale_Centrality(G, pos, t_min = t_min, t_max = t_max, n_t = n_t, n_processes = n_processes, normalization_tpe = normalization_tpe, rw_tpe = rw_tpe, rev = rev)

#compute the measures
print('compute the centrality measures')
mc.compute_multiscale_centralities()
mc.save_centralities()

#plot them as frames in folders
print('plot the results')

#if not os.path.isdir('images_reachability'):
#    os.mkdir('images_reachability')
if not os.path.isdir('images_multiscale'):
    os.mkdir('images_multiscale')

#compare with other centrality measures
print('compare with other centrality measures')
mc.compare_centralities(n_compare = n_t, n_top = 2, n_force = 100)
mc.save_comparisons()

mc.figsize=(6,3)
mc.plot_comparisons_spearman()
 
# plot the multiscale centralities as a function of scale
mc.plot_trajectories()

mc.figsize = (5,4)
mc.video_multiscale(n_plot = n_plot, node_size=50)


