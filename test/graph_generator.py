import networkx as nx
import numpy as np

################################################################
## Gragh generator function, linked with the graph_params.yaml
################################################################

def generate_graph(tpe='SM', params= {}):

    pos = [] 

    if tpe == 'karate':
        """
        Karate club network
        """

        G = nx.karate_club_graph()
        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['club']+' ' + str(i)

        for i,j in G.edges:
            G[i][j]['weight'] = 1. 
    
    elif tpe == 'powergrid':

        edges    = np.genfromtxt('../datasets/UCTE_edges.txt')
        location = np.genfromtxt('../datasets/UCTE_nodes.txt')
        posx = location[:,1]
        posy = location[:,2]
        pos  = {}

        edges = np.array(edges,dtype=np.int32)
        G = nx.Graph() #empty graph
        G.add_edges_from(edges) #add edges

        G = nx.convert_node_labels_to_integers(G, label_attribute = 'old_label' )

        #create the position vector for plotting
        for i in G.nodes():
            pos[i] = [posx[G.nodes[i]['old_label']-1],posy[G.nodes[i]['old_label']-1]]

    elif tpe == 'celegans':
        """
        directed weighted C.elegans network
        """

        from datasets.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../datasets/celegans/')

        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']
            G.nodes[i]['type'] = neuron_type[i]

    elif tpe == 'celegans_undirected':
        """
        undirected weighted C.elegans network
        """

        from datasets.celegans.create_graph import create_celegans 
        G, pos, labels, neuron_type, colors = create_celegans(location = '../datasets/celegans/')

        for i in G:
            G.nodes[i]['old_label'] = G.nodes[i]['labels']

        G = G.to_undirected()

    elif tpe == 'delaunay-grid' or tpe == 'delaunay-grid-noisy_1'or tpe == 'delaunay-grid-noisy_2' or tpe == 'delaunay-grid-noisy_3':
        """
        Delaunay grid with noise parameter eps
        """

        from scipy.spatial import Delaunay

        #np.random.seed(0)
        x = np.linspace(0,1,params['n'])

        points = []
        for i in range(params['n']):
            for j in range(params['n']):
                points.append([x[j]+np.random.normal(0,params['eps']),x[i]+np.random.normal(0,params['eps'])])

        points = np.array(points)

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0],t[1]])
            edge_list.append([t[0],t[2]])
            edge_list.append([t[1],t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()

        for i,j in G.edges:
            G[i][j]['weight'] = 1./np.linalg.norm(points[i]-points[j])


    elif tpe == 'delaunay':
        """
        Delaunay grid with inhomogeneity
        """

        from scipy.spatial import Delaunay
        np.random.seed(0)
        x = np.linspace(0,1,params['n'])

        points = []
        for i in range(params['n']):
            for j in range(params['n']):
                points.append([x[j],x[i]])

        points = np.array(points)

        gauss_pos = [.2, 0.2]
        gauss_pos2 = [0.7, 0.7]
        gauss_var = [.05,.05]
        new_points = np.random.normal(gauss_pos, gauss_var , [50,2])

        for p in new_points:
            if p[0]>0 and p[0]<1. and p[1]>0 and p[1]<1:
                points = np.concatenate( (points, [p,]) )

        tri = Delaunay(points)

        edge_list = []
        for t in tri.simplices:
            edge_list.append([t[0],t[1]])
            edge_list.append([t[0],t[2]])
            edge_list.append([t[1],t[2]])

        G = nx.Graph()
        G.add_nodes_from(np.arange(len(points)))
        G.add_edges_from(edge_list)
        pos = points.copy()

        for i,j in G.edges:
            G[i][j]['weight'] = 1./np.linalg.norm(points[i]-points[j])


    return G, pos

