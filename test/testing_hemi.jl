using RDatasets, DataFrames, CSV
using Random, Distributions
using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD

# load functions for branch&bound and data preprocess from self-created module
if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end
using data_process, bb_functions, opt_functions



#############################################################
################# Main Process Program Body #################
#############################################################

# real world dataset testing
if Sys.iswindows()
    data, label = data_preprocess("Hemi_reduced.csv", nothing, joinpath(@__DIR__, "..\\data\\"), nothing, true) # read data in Windows
else
    data, label = data_preprocess("Hemi_reduced.csv", nothing, joinpath(@__DIR__, "../data/"), nothing, true) # read data in Mac
end
label = vec(label)
k = length(unique(label))
Random.seed!(123)


# local optimization for kmeans clustering
centers_l, assign_l, objv_l = local_OPT(data, k)
# branch&bound global optimization for kmeans clustering
t = @elapsed centers, objv, calcInfo = branch_bound(data, k)
t_LD = @elapsed centers_LD, objv_LD, calcInfo_LD = bb_functions.branch_bound_LD(data, k)
t_adp = @elapsed centers_adp, objv_adp, calcInfo_adp = bb_functions.branch_bound_adptGp(data, k) # 237s 11 iterations
t_adp_LD = @elapsed centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = bb_functions.branch_bound_adptGp_LD(data, k) #

# kmeans results for comparison
t_km = @elapsed rlt_km = kmeans(data, k)
nmi_km, vi_km, ari_km = cluster_eval(rlt_km.assignments, label)

# plot branch and bound calculation process
plotResult(calcInfo, "hemi")
#plotResult(calcInfo_LD)
plotResult(calcInfo_adp, "hemi")
#plotResult(calcInfo_adp_LD)


# Nested evaluation on the clustering results with kmeans
eval_orig = nestedEval(data, label, centers, objv, rlt_km) # evaluation with orignal bb
eval_LD = nestedEval(data, label, centers_LD, objv_LD, rlt_km) # evaluation with LD bb
eval_adp = nestedEval(data, label, centers_adp, objv_adp, rlt_km) # evaluation with adpative grouping bb
eval_adp_LD = nestedEval(data, label, centers_adp_LD, objv_adp_LD, rlt_km) # evaluation with LD and adpative grouping bb

# nestRlt save results for comparison plot. Each row represents: time, gap, nmi, vi, ari, final_cost
timeGapRlt = [[t t_LD t_adp t_adp_LD]; [calcInfo[end][end] calcInfo_LD[end][end] calcInfo_adp[end][end] calcInfo_adp_LD[end][end]]]

evalRlt = [eval_orig[:,end] eval_LD[:,end] eval_adp[:,end] eval_adp_LD[:,end] [nmi_km; vi_km; ari_km; rlt_km.totalcost]]

save("testing_hemi.jld", "data", data,  "timeGapRlt", timeGapRlt, "evalRlt", evalRlt)