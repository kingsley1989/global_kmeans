# global_kmeans
The global branch and bound clustering algorithm for minimum sum-of-squares clustering (MSSC, kmeans-like objective) task.

## related paper
[A Scalable Deterministic Global Optimization Algorithm for Clustering Problems](http://proceedings.mlr.press/v139/hua21a/hua21a.pdf) *Accepted ICML2021*

## source file list
bb_functions.jl branch and bound functions.

lb_functions.jl lower bound calculation functions.

ub_functions.jl upper bound calculation functions.

opt_functions.jl local and global optimization functions.

data_process.jl data preprocess and result evaluation and plotting functions.

branch.jl method for branching functions.

Nodes.jl The Node struct of BB procedure.

## test file list
testing_toy.jl testing code for toy data. 

testing_real.jl testing code for real-world data.

## testing data info
glass identification dataset: https://archive.ics.uci.edu/ml/datasets/glass+identification

Hemicellulose: data batch hemicellulose hydrolysis of hardwood. Each of its data sample represents a reaction condition.

pr2392: travel salesman location data, http://www.math.uwaterloo.ca/tsp/history/tspinfo/pr2392_info.html

seeds: https://archive.ics.uci.edu/ml/datasets/seeds

all synthetic datasets are two-dimensional with three Gaussian distributed clusters. Please refer to testing_toy.jl line 46-53 for the detailed generation process. all datasets use the random seed 1.

## installation and testing
Please refer to line 9-12 of testing_*.jl file for the meaning of each input parameter 
```shell
julia test/testing_toy.jl 1 3 50 CF

julia test/testing_real.jl 1 3 iris CF
```