using RDatasets, DataFrames, CSV
using Random, Distributions
#using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD

using Distributed, SharedArrays

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



t_adp_LD = @elapsed centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = bb_functions.branch_bound(data, k, "LD+adaGp", "fixed", "SOS1", ARGS[4], ARGS[5]) #
println("$dataname:\t",  round(t_adp_LD, digits=2), "\t", round(calcInfo_adp_LD[end][end]*100, digits=4), "%\t", 
    calcInfo_adp_LD[end][end] <= 0.001 ? length(calcInfo_adp_LD)-1 : length(calcInfo_adp_LD))


rmprocs(procs()[2:nprocs()])