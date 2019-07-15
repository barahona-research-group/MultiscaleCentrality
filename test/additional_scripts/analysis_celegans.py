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

neuron_type = []
for i in mc.G:
    neuron_type.append(mc.G.node[i]['type'])

neuron_type = np.array(neuron_type)

neuron_tpe_0 = np.argwhere(neuron_type==0).flatten()
neuron_tpe_1 = np.argwhere(neuron_type==1).flatten()
neuron_tpe_2 = np.argwhere(neuron_type==2).flatten()
print(neuron_tpe_0)
print(neuron_tpe_1)
print(neuron_tpe_2)

tpe_0_mean = []
tpe_1_mean = []
tpe_2_mean = []

tpe_0_std = []
tpe_1_std = []
tpe_2_std = []

for i, tau in enumerate(mc.Times):

    mc.multiscale[:, i] /= np.max(mc.multiscale[:, i])
    tpe_0_mean.append(np.mean(mc.multiscale[neuron_tpe_0, i]))
    tpe_1_mean.append(np.mean(mc.multiscale[neuron_tpe_1, i]))
    tpe_2_mean.append(np.mean(mc.multiscale[neuron_tpe_2, i]))
    tpe_0_std.append(np.std(mc.multiscale[neuron_tpe_0, i]))
    tpe_1_std.append(np.std(mc.multiscale[neuron_tpe_1, i]))
    tpe_2_std.append(np.std(mc.multiscale[neuron_tpe_2, i]))

tpe_0_mean = np.array(tpe_0_mean)
tpe_1_mean = np.array(tpe_1_mean)
tpe_2_mean = np.array(tpe_2_mean)
tpe_0_std = np.array(tpe_0_std)
tpe_1_std = np.array(tpe_1_std)
tpe_2_std = np.array(tpe_2_std)

plt.figure(figsize=(6,3))
plt.semilogx(mc.Times, tpe_0_mean, label='type M', c='C0', lw=5)
plt.plot(mc.Times, tpe_1_mean, label='type I', c='C1',lw=5)
plt.plot(mc.Times, tpe_2_mean, label='type S', c='C2',lw=5)

plt.plot(mc.Times, tpe_0_mean - tpe_0_std, ls='--', color='C0')
plt.plot(mc.Times, tpe_1_mean - tpe_1_std,  ls='--', color='C1')
plt.plot(mc.Times, tpe_2_mean - tpe_2_std, ls='--', color='C2')
plt.plot(mc.Times, tpe_0_mean + tpe_0_std, ls='--', color='C0')
plt.plot(mc.Times, tpe_1_mean + tpe_1_std,  ls='--', color='C1')
plt.plot(mc.Times, tpe_2_mean + tpe_2_std, ls='--', color='C2')



plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.ylabel('Mean normalized multiscale centrality')

plt.axis([mc.Times[0],mc.Times[-1], -0.1, 0.75])
plt.savefig('neuron_types.svg', bbox_inches="tight" ) 

plt.show()
