using RDatasets, DataFrames, CSV
using Random, Distributions
#using Plots, StatsPlots
using MLDataUtils, Clustering

# load functions for branch&bound and data preprocess from self-created module
if !("." in LOAD_PATH)
    push!(LOAD_PATH, ".")
end
using data_process, bb_functions, opt_functions



#############################################################
################# Main Process Program Body #################
#############################################################

# real world dataset testing
data, label = data_preprocess("iris") # read iris data from datasets package

label = vec(label)
k = length(unique(label))
Random.seed!(123)


# local optimization for kmeans clustering
centers_l, assign_l, objv_l = local_OPT(data, k)
# branch&bound global optimization for kmeans clustering
@time centers, objv, calcInfo = branch_bound(data, k)
# @time centers_LD, objv_LD, calcInfo_LD = bb_functions.branch_bound_LD(data, k)
@time centers_adp1, objv_adp1, calcInfo_adp1 = bb_functions.branch_bound_adptGp1(data, k) # 237s 11 iterations
# @time centers_adp1_LD, objv_adp1_LD, calcInfo_adp1_LD = bb_functions.branch_bound_adptGp1_LD(data, k) #


# plot branch and bound calculation process
plotResult(calcInfo)
plotResult(calcInfo_adp1)

# Nested evaluation on the clustering results with kmeans
nestedEval(data, label, centers, objv) # evaluation with orignal bb
nestedEval(data, label, centers_adp1, objv_adp1) # evaluation with adpative grouping bb


