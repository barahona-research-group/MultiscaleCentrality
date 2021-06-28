import numpy as np
import scipy as sc
import networkx as nx
import pylab as plt
import pickle as pickle
import time
from tqdm import tqdm
import seaborn as sns


import scipy.stats as st
from fa2 import ForceAtlas2


from multiprocessing import Pool
from functools import partial

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

class Multiscale_Centrality(object):
    """
    class to compute the multiscale centrality of a graph
    """

    def __init__(self, G = [], pos = [], t_min = -2, t_max = 1, n_t = 100, log = True, target_nodes = [], n_processes = 2, precision = 1e-10, normalization_tpe = 'combinatorial', rw_tpe = 'continuous', alpha = 0.35, rev = False, node_labels = False):

        """
        Initialisation function of this class
        """

        if len(G)>0:
            self.load_graph(G, pos)

        self.figsize = None
        self.labels = node_labels
        self.plot_edges = True

        self.n_processes = n_processes # number of cpus to use
        self.precision = precision   #machine precision threshold for the computation of expm
        self.normalization_tpe = normalization_tpe #type of Laplacian, combinatorial or normalized
        self.rw_tpe = rw_tpe #type of random walk, continuous or discrete
        self.alpha = alpha # for discrete random walks
        self.rev = rev #for directed graph, reverse flow or not (False/True)

        self.time_spectral_gap = True #set to false for no time rescaling
        
        #set the Markov time parameters
        self.graph_Laplacian()

        if self.time_spectral_gap:  
            print("Spectral gap = ", self.lamb_2)

        #set time boundaries
        self.t_min = t_min
        self.t_max = t_max
        
        self.disable_tqdm = False #set True to not show the tqdm progression bars
        
        #compute the centrality measures only w.r.t to a subset of nodes in target_nodes
        if len(target_nodes) == 0:
            self.target_nodes = G.nodes()
            self.node_mask = np.ones(len(G))
        else:
            self.target_nodes = target_nodes
            self.node_mask = np.zeros(len(G))
            self.node_mask[target_nodes] = 1
            
        self.n_target = len(self.target_nodes)

        #for discrete random walks, find the number of steps corresponding to the time
        if self.rw_tpe == 'discrete':
            self.n_t = int(self.t_max)+1
            self.transition_matrix()
            print('using', self.n_t, 'steps')
        else:
            self.n_t = n_t

        #use log scaled time samples
        self.log = log
        if self.log:
            self.Times = np.logspace(self.t_min, self.t_max, self.n_t) #time vector
        else:
            self.Times = np.linspace(self.t_min, self.t_max, self.n_t) #time vector

        #which centrality to compare with
        self.centralities_list = ['degree', 'eigenvector',  'closeness', 'betweenness',]#'katz', 'pagerank']#, 'force', ] 

    def load_graph(self, G, pos = []):
        """
        load the network 
        """

        self.G = G
        self.n = len(G.nodes)
        self.m = len(G.edges)
        
        #if no positions given, use force atlas
        if len(pos) == 0:
            forceatlas2 = ForceAtlas2(
                        # Tuning
                        scalingRatio=2,
                        strongGravityMode=False,
                        gravity=1,
                        outboundAttractionDistribution=False,  # Dissuade hubs
                        # Log
                        verbose=False)

            self.pos = forceatlas2.forceatlas2_networkx_layout(self.G, pos=None, iterations=2000)
        else: #else use positions
            self.pos = pos


    def delta(self, i):
        """
        return a delta initial condition
        """

        p0 = np.zeros(self.n)
        p0[i] = 1.

        return p0


    def graph_Laplacian(self):
        """
        Compute the graph Laplacian with spectral gap normalisation 
        """

        if nx.is_directed(self.G):

            if self.normalization_tpe == 'normalized':

                print('Does not work!')

            elif self.normalization_tpe == 'combinatorial':
                L  = sc.sparse.csc_matrix(np.array(directed_combinatorial_laplacian_matrix(self.G, walk_type='pagerank', alpha=0.85, rev = self.rev)))
                v = np.array(abs(sc.sparse.linalg.eigs(L, which='SM', k=1)[1])).flatten() #stationary state
                self.v = v/v.sum()
                L_sub = L

            else:
                print('Not defined')
        else:

            if self.normalization_tpe == 'combinatorial':

                L = sc.sparse.csr_matrix(1.*nx.laplacian_matrix(self.G)) #combinatorial Laplacian
                self.v = np.ones(self.n)/self.n

            elif self.normalization_tpe == 'normalized':
                A = nx.adjacency_matrix(self.G).toarray()
                degree = np.array(A.sum(1)).flatten()
                self.v = degree/degree.sum()

                L = sc.sparse.csr_matrix((np.diag(1./degree)).dot(nx.laplacian_matrix(self.G).toarray())) #combinatorial Laplacian

            elif self.normalization_tpe == 'max_entropy':
                A = nx.adjacency_matrix(self.G)
                eigs = sc.sparse.linalg.eigsh(1.*A, which='LM', k=1)
                lamb_0 = abs(eigs[0][0])
                psi = eigs[1][:,0]
                L = sc.sparse.csr_matrix(np.eye(self.n) - np.diag(psi).dot(A.toarray()).dot(np.diag(1./psi))/lamb_0)
                self.v = psi**2

            else:
                print('Not defined!')
                
            #compute the spectral gap of largest connected component
            graphs = sorted(nx.connected_components(self.G), key=len, reverse=True)
            if len(graphs)>1:
                print('WARNING: graph not connected!')

                L_sub = L[np.ix_(graphs[0].nodes,graphs[0].nodes)]
            else:
                L_sub = L

        if self.time_spectral_gap:
            self.lamb_2 = abs(sc.sparse.linalg.eigs(L_sub, which='SM', k=2)[0][1])
        else:
            self.lamb_2 = 1.

        self.L = sc.sparse.csc_matrix(L)/self.lamb_2


    def solve_continuous_time(self, p0):
        """
        compute the exponential for a p0 initial condition
        """

        if self.log:
            
            #faster to apply exponential incrementally
            p_t = []
            p_t.append(sc.sparse.linalg.expm_multiply(-self.Times[0]*self.L, p0))
            for i in range(len(self.Times)-1):
                p_t.append(sc.sparse.linalg.expm_multiply(-(self.Times[i+1]-self.Times[i])*self.L, p_t[-1]))
        else:
            p_t = sc.sparse.linalg.expm_multiply(-self.L, p0, self.t_min, self.t_max, self.n_t)

        return np.array(p_t)

    def transition_matrix(self):
        """
        compute discrete lazy walk transition matrices for normalized or combinatorial
        """
    

        if nx.is_directed(self.G):
            if self.rev: #again reverse order
                A = sc.sparse.csr_matrix(_transition_matrix(self.G, walk_type='pagerank', alpha=0.85, rev = False))
            else:
                A = sc.sparse.csr_matrix(_transition_matrix(self.G, walk_type='pagerank', alpha=0.85, rev = True))
        else:
            A = nx.adjacency_matrix(self.G)

        if self.normalization_tpe == 'combinatorial':
            eigs = sc.sparse.linalg.eigs(1.*A, which='LM', k=1)
            lamb_0 = abs(eigs[0][0])
            v_0 = eigs[1][:,0]
            T = A/lamb_0
            
        elif self.normalization_tpe == 'min_entropy':
            eigs = sc.sparse.linalg.eigs(1.*A, which='LM', k=1)
            lamb_0 = abs(eigs[0][0])
            psi = eigs[1][:,0]
            T = sc.sparse.csr_matrix(np.diag(psi).dot(A.toarray()).dot(np.diag(1./psi))/lamb_0)
            
        elif self.normalization_tpe == 'max_entropy':
            eigs = sc.sparse.linalg.eigs(1.*A, which='LM', k=1)
            lamb_0 = abs(eigs[0][0])
            psi = eigs[1][:,0]
            T = sc.sparse.csr_matrix(np.diag(1./psi).dot(A.toarray()).dot(np.diag(psi))/lamb_0)

        elif self.normalization_tpe == 'normalized': 
            Dinv = sc.sparse.csr_matrix(np.diag(1./np.sqrt(np.array(A.sum(1).reshape(self.n))[0])))
            T = Dinv.dot(A).dot(Dinv)
        else:
            print('Not defined!')
            
        self.T = sc.sparse.csc_matrix((np.eye(self.n)*self.alpha + (1.-self.alpha)*T))

        #compute the stationary solution
        v = np.array(abs(sc.sparse.linalg.eigs(self.T, which='LM', k=1)[1])).flatten() #stationary state
        self.v = v/v.sum()

    def solve_discrete_time(self):
        """
        compute the exponential for a p0 initial condition
        """

        Ts = [self.T.todense(), ]
        Ts_last = self.T #save sparse matrices for faster computations

        for i in range(self.n_t-1):
            T_new = self.T.dot(Ts_last)
            Ts.append(T_new.toarray()) 
            Ts_last  = T_new.copy() 

        return np.array(Ts)
    
    def compute_trajectories(self, p0):
        """
        Compute the node trajectories from a source p0
        """
        if self.rw_tpe == 'discrete':
             return self.solve_discrete_time().dot(p0)

        if self.rw_tpe == 'continuous':
             return self.solve_continuous_time(p0)


    def compute_peak_distance(self, p_t):
        """
        Compute the multiscale centrality vector of node with diffusion trajectories p_t
        """

        distances = [] #np.zeros([self.n, self.n_t]) #empty distance matrix

        for tau in range(self.n_t): #for each tau
            
            #id_reachable = np.argwhere((p_t[:tau+1]*self.node_mask).max(0) > self.v + self.precision).flatten() #find reachable nodes
            id_reachable = np.argwhere((p_t[:tau+1]).max(0) > self.v + self.precision).flatten() #find reachable nodes
            distance = (self.t_max + 1e8)*np.ones(self.n) #set the distance to unreachable to all: (t_max+1)
            distance[id_reachable] = self.Times[np.argmax(p_t[:tau+1, id_reachable], axis=0)] #set the time for reachable nodes

            distances.append(distance) #collect the distance

        return distances


    def compute_multiscale_centrality(self, pair_distances):
        """
        Compute the multiscale centrality vector of node with diffusion trajectories p_t
        """

        args =  [self.n, self.target_nodes, self.precision]
        compute_triangle_pool_p = partial(compute_triangle_pool, args)
        with Pool(processes = self.n_processes) as p_tri:  #initialise the parallel computation
            out = list(tqdm(p_tri.imap(compute_triangle_pool_p, pair_distances), total = self.n_t, disable=self.disable_tqdm))
        
        triangles = np.zeros([self.n, self.n_t])
        for tau in range(self.n_t): #for each tau
            triangles[:,tau] = out[tau] 

        return triangles


    def compute_centrality_pool(self, i):

        if self.rw_tpe == 'continuous':
            p_t = self.solve_continuous_time(self.delta(i))
        
        if self.rw_tpe == 'discrete':
            p_t = np.array(self.Ts[:, i])
        
        pair_distances = self.compute_peak_distance(p_t) 

        return pair_distances

    def compute_multiscale_centralities(self):
        """
        compute reachability and multiscale centrality
        """

        if self.rw_tpe == 'continuous':
            self.graph_Laplacian() # compute the graph Laplacian first

        if self.rw_tpe == 'discrete':
            self.transition_matrix()
            self.Ts = self.solve_discrete_time()


        with Pool(processes = self.n_processes) as p_uc:  #initialise the parallel computation
            out = list(tqdm(p_uc.imap(self.compute_centrality_pool,
                                      self.target_nodes), 
                                      total = self.n_target,
                                      disable=self.disable_tqdm))
            
        self.out = out
        self.pair_distances = np.zeros([self.n_t, self.n, self.n])

        for i,node in enumerate(self.target_nodes):
            self.pair_distances[:, node, :] = np.array(out[i])
            
        if self.n_target < len(self.G):
            for t in range(self.n_t):
                self.pair_distances[t, :, :] = np.maximum( self.pair_distances[t, :, :], self.pair_distances[t, :, :].T )
                
                
        self.multiscale = self.compute_multiscale_centrality(self.pair_distances)
    

    def plot_multiscale_centrality(self, tau, node_size = 200):
        """
        plot the multiscale centrality for a given tau
        """

        plt.figure(figsize = self.figsize)

        vmin = 0
        vmax = 1

        node_size = self.multiscale[:, tau]/np.max(self.multiscale[:, tau])*node_size
        node_order = np.argsort(node_size)
        for n in node_order:
            nodes = nx.draw_networkx_nodes(self.G, pos = self.pos, nodelist = [n,], node_size = node_size[n], node_color=[self.multiscale[n, tau]/np.max(self.multiscale[:, tau]),], vmin=vmin, vmax=vmax)

        if self.n_target < len(self.G):
            nodes = nx.draw_networkx_nodes(self.G, nodelist=self.target_nodes, pos = self.pos, node_size = node_size/3, node_color='r')

        if self.plot_edges:
            #weights = np.array([self.G[i][j]['weight'] for i,j in self.G.edges])
            nx.draw_networkx_edges(self.G, pos = self.pos, alpha=0.5)# ,width = 2*weights)

        if self.labels:
            old_labels={}
            for i in self.G:
                old_labels[i] = self.G.node[i]['old_label']
            nx.draw_networkx_labels(self.G, pos = self.pos, labels = old_labels)

        limits = plt.axis('off') #turn axis odd


    def video_multiscale(self, n_plot = 10, folder = 'images_multiscale', node_size = 200):
        """
        plot the multiscale centrality for all tau 
        """

        if n_plot > self.n_t-1:
            n_plot = self.n_t-1

        dtau = int((self.n_t)/n_plot)
        for i in tqdm(range(n_plot), disable=self.disable_tqdm):
            tau = i*dtau    


            self.plot_multiscale_centrality(tau, node_size = node_size)
            if self.log:
                plt.title(r'$log_{10}(\tau)=$'+str(np.around(np.log10(self.Times[tau]),2)))
            else:
                plt.title(r'$\tau=$'+str(np.around(self.Times[tau],2)))
            plt.savefig(folder + '/multiscale_' + '%0.3d' % i + '.svg')
            plt.close()


    def plot_trajectories(self):
        """
        Plot the multiscale centrality of each node as a function of scale
        """

        plt.figure(figsize=self.figsize)

        for i in range(np.shape(self.multiscale)[1]):
            self.multiscale[:,i] /= np.max(self.multiscale[:,i])

        for i in range(len(self.multiscale)):
            plt.semilogx(self.Times, self.multiscale[i], lw=0.5, alpha=1.0, c='0.5')

        #highlight central nodes at small and large scales
        for i in range(len(self.multiscale)):
            if i == np.argmax(self.multiscale[:,0],axis=0):
                plt.semilogx(self.Times, self.multiscale[i], lw=3.0, c='b')

            if i == np.argmax(self.multiscale[:,-1],axis=0):
                plt.semilogx(self.Times, self.multiscale[i], lw=3.0, c='r')

        plt.axis([self.Times[0], self.Times[-1], 0,1.05])
        plt.xlabel(r'$\tau$')
        plt.ylabel('Normalized Multiscale centrality')
        plt.axis([self.Times[0], self.Times[-1], -0.02,1.02])
        plt.savefig('multiscale_trajectories.svg', bbox_inches="tight")
        plt.close()

    
    def other_centralities(self, n_force = 20, c = 0):

        C = [] #to collect the centralities

        for centrality in self.centralities_list:
            if centrality == 'force':
                "find node position with force atlas, and distance to the center is the centrality"
                forceatlas2 = ForceAtlas2(
                                # Tuning
                                scalingRatio=2.0,
                                strongGravityMode=False,
                                gravity=1.0,

                                # Log
                                verbose=False)
                pos = forceatlas2.forceatlas2_networkx_layout(self.G, pos=None, iterations=2000)
                c = np.linalg.norm(np.array(list(pos.values())),axis=1)

                for i in range(n_force-1):
                    pos = forceatlas2.forceatlas2_networkx_layout(self.G, pos=None, iterations=2000)
                    c += np.linalg.norm(np.array(list(pos.values())), axis=1)
                    
                c = -c/n_force
                
            elif centrality == 'degree':
                #degree centrality
                c = list(nx.degree_centrality(self.G).values())
            
            elif centrality == 'eigenvector':
                #eigenvector centrality
                try:
                    c = list(nx.eigenvector_centrality_numpy(self.G).values())
                except:
                    print(centrality + 'failed computation')
                    c = np.zeros(self.n)
                
            elif centrality == 'katz':
                #katz centrality
                try:
                    print('alpha (Kac)=', 1./np.max(np.linalg.eigh(nx.adjacency_matrix(self.G).toarray())[0]))
                    alpha = 1./np.max(np.linalg.eigh(nx.adjacency_matrix(self.G).toarray())[0]) - 5e-3 
                    c = list(nx.katz_centrality(self.G, alpha=alpha).values())
                except:
                    c = list(nx.katz_centrality(self.G, max_iter = 1000, tol=1e-3).values())
                    print(centrality + ' failed computation')
                    c = np.zeros(self.n)
                
            elif centrality == 'closeness':
                #closeness centrality
                c = list(nx.closeness_centrality(self.G).values())
                
            elif centrality == 'betweenness':
                #betweenness centrality
                c = list(nx.betweenness_centrality(self.G).values())
             
            elif centrality == 'pagerank':
                #betweenness centrality
                c = list(nx.pagerank(self.G, alpha = 1).values())

            elif centrality == 'other':
                c = c

            else:
                print("I don't know this one!")
           
            C.append(c)

        return C


    def compare_centralities_spearman(self, n_compare, n_force = 10):
        """
        compare the centrality measures using spearman correlation
        """

        plot = False
        disp = False

        C = self.other_centralities()

        if n_compare > self.n_t-1:
            n_compare = self.n_t-1
        n_compare = self.n_t 


        spearman_multiscale = np.zeros([n_compare, len(C)]) #to collect the pearson coefficients for multiscale 

        for i in range(n_compare):

            for ic, centrality in enumerate(self.centralities_list):

                tri = self.multiscale[:, i]
                spearman_multiscale[i,ic] = st.spearmanr(tri, C[ic])[0] 

        spearman_multiscale[np.isnan(spearman_multiscale)] = 0 #set nan values to 0

        self.spearman_multiscale = spearman_multiscale

 
    def compare_centralities(self, n_compare, n_top, n_force = 10):
        """
        compute all the centralities comparisons
        """

        self.compare_centralities_spearman(n_compare, n_force)


    def save_comparisons(self, folder = ''):
        """
        save comparison data
        """

        pickle.dump([self.centralities_list, self.spearman_multiscale], open(folder + 'uc_multiscale_comparisons.pkl','wb'))


    def load_comparisons(self, folder = ''): 
        """
        load comparison data
        """

        self.centralities_list, self.spearman_multiscale = pickle.load( open(folder + 'uc_multiscale_comparisons.pkl','rb'))


    def plot_comparisons_spearman(self, folder = ''):
        """
        plot the comparison between reachability/multiscale centrality with other centrality measures
        """

        plt.figure(figsize = self.figsize)

        for i, centrality in enumerate(self.centralities_list):
            if self.log:
                plt.semilogx(self.Times, self.spearman_multiscale[:,i], label=centrality)
            else:
                plt.plot(self.Times, self.spearman_multiscale[:,i], label=centrality)

        plt.legend(loc='lower right')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\mathrm{Spearman\, correlation}$')
        plt.axis([self.Times[0], self.Times[-1],  np.min(self.spearman_multiscale) , 1])

        plt.savefig(folder+'multiscale_spearman.svg', bbox_inches="tight" ) 

    def save_centralities(self, folder = ''):
        """
        save the results in a pickle
        """

        pickle.dump(self.multiscale, open(folder + 'uc_results.pkl','wb'))

    def load_centralities(self, folder = ''):
        """
        load the results from a pickle
        """

        self.multiscale =  pickle.load(open(folder + 'uc_results.pkl','rb'))


####################################
## functions from latest networkx ##
####################################

def directed_laplacian_matrix(G, nodelist=None, weight='weight',
                              walk_type=None, alpha=0.95, rev = False):
    r"""Returns the directed Laplacian matrix of G.

    The graph directed Laplacian is the matrix

    .. math::

        L = I - (\Phi^{1/2} P \Phi^{-1/2} + \Phi^{-1/2} P^T \Phi^{1/2} ) / 2

    where `I` is the identity matrix, `P` is the transition matrix of the
    graph, and `\Phi` a matrix with the Perron vector of `P` in the diagonal and
    zeros elsewhere.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy array
      Normalized Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    See Also
    --------
    laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    import scipy as sp
    from scipy.sparse import spdiags, linalg

    P = _transition_matrix(G, nodelist=nodelist, weight=weight,
                           walk_type=walk_type, alpha=alpha, rev = rev)


    n, m = P.shape

    evals, evecs = linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = sp.sqrt(p)
    Q = spdiags(sqrtp, [0], n, n) * P * spdiags(1.0 / sqrtp, [0], n, n)
    I = sp.identity(len(G))

    return I - (Q + Q.T) / 2.0



def directed_combinatorial_laplacian_matrix(G, nodelist=None, weight='weight',
                                            walk_type=None, alpha=0.95, rev = False):
    r"""Return the directed combinatorial Laplacian matrix of G.

    The graph directed combinatorial Laplacian is the matrix

    .. math::

        L = \Phi - (\Phi P + P^T \Phi) / 2

    where `P` is the transition matrix of the graph and and `\Phi` a matrix
    with the Perron vector of `P` in the diagonal and zeros elsewhere.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy array
      Combinatorial Laplacian of G.

    Notes
    -----
    Only implemented for DiGraphs

    See Also
    --------
    laplacian_matrix

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    from scipy.sparse import spdiags, linalg

    P = _transition_matrix(G, nodelist=nodelist, weight=weight,
                           walk_type=walk_type, alpha=alpha, rev = rev)

    n, m = P.shape

    evals, evecs = linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    Phi = spdiags(p, [0], n, n)

    Phi = Phi.todense()

    return Phi - (Phi*P + P.T*Phi) / 2.0


def _transition_matrix(G, nodelist=None, weight='weight',
                       walk_type=None, alpha=0.95, rev = False):
    """Returns the transition matrix of G.

    This is a row stochastic giving the transition probabilities while
    performing a random walk on the graph. Depending on the value of walk_type,
    P can be the transition matrix induced by a random walk, a lazy random walk,
    or a random walk with teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    P : NumPy array
      transition matrix of G.

    Raises
    ------
    NetworkXError
        If walk_type not specified or alpha not in valid range
    """

    import scipy as sp
    from scipy.sparse import identity, spdiags
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = "random"
            else:
                walk_type = "lazy"
        else:
            walk_type = "pagerank"

    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)

    if not rev: 
        M = M.T

    n, m = M.shape
    if walk_type in ["random", "lazy"]:
        DI = spdiags(1.0 / sp.array(M.sum(axis=1).flat), [0], n, n)
        if walk_type == "random":
            P = DI * M
        else:
            I = identity(n)
            P = (I + DI * M) / 2.0

    elif walk_type == "pagerank":
        if not (0 < alpha < 1):
            raise nx.NetworkXError('alpha must be between 0 and 1')
        # this is using a dense representation
        M = M.todense()
        # add constant to dangling nodes' row
        dangling = sp.where(M.sum(axis=1) == 0)
        for d in dangling[0]:
            M[d] = 1.0 / n
        # normalize
        M = M / M.sum(axis=1)
        P = alpha * M + (1 - alpha) / n
    else:
        raise nx.NetworkXError("walk_type must be random, lazy, or pagerank")

    return P



########## ###################
## multiprocessing function ##
##############################

def compute_triangle_pool(args, pair_distances):

    """
    Compute the triangle inequalities for multiprocessing
    """
    n, target_nodes, precision = args
    pair_distances = np.array(pair_distances)

    triangles = np.zeros(n)
    for i in range(n):
        dij = np.tile(pair_distances[i, :], n).reshape( (n, n))
        dist =  dij + dij.T - 0.5*(pair_distances + pair_distances) #average on the last term for directed graph (not needed for un-directed)
        dist = dist[np.ix_(target_nodes, target_nodes)]
        triangles[i] = len(np.argwhere(dist < -precision)) / len(target_nodes)**2

    return triangles
