Test folder
============

This folder contains a test suite for multiscale centrality, it works as follow. 

The script `graph_generator.py` generate the graphs, and needs the parameters set in the yaml file `graph_params.yaml'. 
The folder `datasets` contains some of the datasets used for the graphs. 

Calling these files is done in the script `compute_multiscale_centrality.py` which takes as argument the name of the graph. For example 
```
python compute_multiscale_centrality.py karate
```
will compute the centrality measure for the karate clube network, and save the data in the corresponding subfolder. 

Once the computation is done, plotted and saved by the first script, you can run 
```
python plot_multiscale_centrality.py karate
```
to redo the plots (and modify plotting parameters). 

In addition, the foldef `additional_scripts` contains other scripts related to particular datasets, see the readme in the folder. 

