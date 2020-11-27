using RDatasets, DataFrames, CSV
using Random, Distributions
using Plots#, StatsPlots
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
@time centers_adp, objv_adp, calcInfo_adp = bb_functions.branch_bound_adptGp(data, k) # 237s 11 iterations
# @time centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = bb_functions.branch_bound_adptGp_LD(data, k) #


# plot branch and bound calculation process
plotResult(calcInfo)
plotResult(calcInfo_adp)

# Nested evaluation on the clustering results with kmeans
nestedEval(data, label, centers, objv) # evaluation with orignal bb
nestedEval(data, label, centers_adp, objv_adp) # evaluation with adpative grouping bb


