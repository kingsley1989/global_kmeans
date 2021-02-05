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
    data, label = data_preprocess("seeds_dataset.txt", nothing, joinpath(@__DIR__, "..\\data\\")) # read data in Windows
else
    data, label = data_preprocess("seeds_dataset.txt", nothing, joinpath(@__DIR__, "../data/")) # read data in Mac
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

# global optimization using CPLEX directly objv_lg is the lower bound of current solution
t_g = @elapsed centers_g, objv_g, assign_g, gap_g = global_OPT_base(data, k)

# kmeans results for comparison
Random.seed!(23)
trail = 100
nmi_km_list = []
objv_km_list = []
for i = 1:trail
    t_km = @elapsed rlt_km = kmeans(data, k)
    nmi_km, ~, ~ = cluster_eval(rlt_km.assignments, label)
    push!(nmi_km_list, nmi_km)
    push!(objv_km_list, rlt_km.totalcost)
end
nmi_km_mean = mean(nmi_km_list)
objv_km_mean = mean(objv_km_list)
objv_km_max, nmi_km_max = findmax(objv_km_list)
objv_km_min, nmi_km_min = findmin(objv_km_list)
println([nmi_km_list[nmi_km_max] nmi_km_mean nmi_km_list[nmi_km_min]; objv_km_max objv_km_mean objv_km_min])

# test of km for reference (temporal way)
rlt_km = kmeans(data, k)
~, vi_km, ari_km = cluster_eval(rlt_km.assignments, label)

# plot branch and bound calculation process
plotResult(calcInfo, "seeds")
plotResult(calcInfo_LD, "seeds_LD")
plotResult(calcInfo_adp, "seeds_adp")
plotResult(calcInfo_adp_LD, "seeds_adp_LD")


# Nested evaluation on the clustering results with kmeans
eval_CPLEX = nestedEval(data, label, centers_g, objv_g, rlt_km) # evaluation with CPLEX solver
eval_orig = nestedEval(data, label, centers, objv, rlt_km) # evaluation with orignal bb
eval_LD = nestedEval(data, label, centers_LD, objv_LD, rlt_km) # evaluation with LD bb
eval_adp = nestedEval(data, label, centers_adp, objv_adp, rlt_km) # evaluation with adpative grouping bb
eval_adp_LD = nestedEval(data, label, centers_adp_LD, objv_adp_LD, rlt_km) # evaluation with LD and adpative grouping bb

# nestRlt save results for comparison plot. Each row represents: time, gap, nmi, vi, ari, final_cost
timeGapRlt = [[t_g t t_LD t_adp t_adp_LD]; [gap_g calcInfo[end][end] calcInfo_LD[end][end] calcInfo_adp[end][end] calcInfo_adp_LD[end][end]]]

evalRlt = [eval_CPLEX[:,end] eval_orig[:,end] eval_LD[:,end] eval_adp[:,end] eval_adp_LD[:,end] [nmi_km_mean; vi_km; ari_km; objv_km_mean]]

save("result/testing_seed.jld", "data", data,  "timeGapRlt", timeGapRlt, "evalRlt", evalRlt)