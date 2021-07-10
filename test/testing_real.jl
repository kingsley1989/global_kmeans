using RDatasets, DataFrames, CSV
using Random, Distributions
#using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD

using Distributed, SharedArrays

# arg1=: number of cores
# arg2=: number of clusters to be solved
# arg3=: dataset name
# arg4=: lower bound method: base-BB, CF, SG, LD+SG

if ARGS[1] == "HPC"
    using ClusterManagers
    addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"])-nprocs(), 
            nodes=parse(Int, ENV["SLURM_JOB_NUM_NODES"]))
else # ARGS[1]==core number if ARGS[1]==1, then it is serial computing
    addprocs(parse(Int, ARGS[1])-nprocs())
end

println("Running ",nprocs()," processes")

# load functions for branch&bound and data preprocess from self-created module
@everywhere if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end

using data_process, bb_functions, opt_functions



#############################################################
################# Main Process Program Body #################
#############################################################

# real world dataset testing
dataname = ARGS[3]
if dataname == "iris"
    data, label = data_preprocess("iris") # read iris data from datasets package
else
    if Sys.iswindows()
        data, label = data_preprocess(dataname, nothing, joinpath(@__DIR__, "..\\data\\"), "NA") # read data in Windows
    else
        data, label = data_preprocess(dataname, nothing, joinpath(@__DIR__, "../data/"), "NA") # read data in Mac
    end
end

label = vec(label)
k = parse(Int, ARGS[2]) #length(unique(label))
Random.seed!(123)

if ARGS[4] == "base-BB"
    t_g = @elapsed centers_g, objv_g, iter_g, gap_g = global_OPT_base(data, k)
    println("$dataname:\t", round(objv_g, digits=2), "\t", round(t_g, digits=2), "\t", 
        round(gap_g, digits=4), "%\t", iter_g)
else
    t_adp_LD = @elapsed centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = bb_functions.branch_bound(data, k, ARGS[4], "fixed", "CPLEX") #
    println("$dataname:\t",  
            round(objv_adp_LD, digits=2), "\t",
            round(t_adp_LD, digits=2), "\t", 
            round(calcInfo_adp_LD[end][end]*100, digits=4), "%\t", 
            calcInfo_adp_LD[end][end] <= 0.001 ? length(calcInfo_adp_LD)-1 : length(calcInfo_adp_LD))
end



#=
# kmeans results for comparison
Random.seed!(0)
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
=#

rmprocs(procs()[2:nprocs()])
