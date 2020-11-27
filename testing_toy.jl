using RDatasets, DataFrames, CSV
using Random, Distributions
# using Plots, StatsPlots
using MLDataUtils, Clustering

# load functions for branch&bound and data preprocess from self-created module
if !("." in LOAD_PATH)
    push!(LOAD_PATH, ".")
end
using data_process, bb_functions, opt_functions

#############################################################
################# Main Process Program Body #################
#############################################################

Random.seed!(123)
clst_n = 10 # number of points in a cluster
k = 3
toy_data = Array{Float64}(undef, 2, 0) # empty array 0*2 
label = Array{Float64}(undef, 0, 1) # label is empty vector 1*0
mu = [5 4; 2 1; 10 3]
sig = [[0.7 0; 0 0.7],[1.5 0;0 1.5],[0.2 0;0 0.6]]
for i = 1:k
    clst = rand(MvNormal(mu[i,:], sig[i]), clst_n) # data is 2*clst_n
    toy_data = hcat(toy_data, clst)
    label = vcat(label, repeat([i], clst_n))
end

k = nlabel(vec(label)) #length(unique(label))
label = convertlabel(1:k, vec(label))

# plot the original data
pyplot()
scatter(toy_data[1,:], toy_data[2,:], markercolor=label, legend = false, title = "My Scatter Plot")

# local optimization for kmeans clustering
centers_l, assign_l, objv_l = local_OPT(toy_data, k)
# branch&bound global optimization for kmeans clustering
@time centers, objv, calcInfo = branch_bound(toy_data, k)
# @time centers_LD, objv_LD, calcInfo_LD= bb_functions.branch_bound_LD(toy_data, k)
@time centers_adp1, objv_adp1, calcInfo_adp1 = bb_functions.branch_bound_adptGp1(toy_data, k) # 237s 11 iterations
# @time centers_adp1_LD, objv_adp1_LD, calcInfo_adp1_LD = bb_functions.branch_bound_adptGp1_LD(toy_data, k) # 233s 1 node with 20 updates of lambda


# plot branch and bound calculation process
plotResult(calcInfo)
plotResult(calcInfo_adp1)

# Nested evaluation on the clustering results with kmeans
nestedEval(toy_data, label, centers, objv) # evaluation with orignal bb
nestedEval(toy_data, label, centers_adp1, objv_adp1) # evaluation with adpative grouping bb

