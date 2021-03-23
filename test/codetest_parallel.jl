using RDatasets, DataFrames, CSV
using Random, Distributions
using Plots#, StatPlots
using Clustering, Distances
using MLDataUtils#, MLBase

using Distributed, SharedArrays
addprocs(8-nprocs())
println("Running ",nprocs()," processes")

# load functions for branch&bound and data preprocess from self-created module
@everywhere if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end
@everywhere if !("src_par/" in LOAD_PATH)
    push!(LOAD_PATH, "src_par/")
end
#@everywhere using opt_functions

using data_process
using lb_par, lb_functions, bb_par, opt_functions



#############################################################
################# Main Process Program Body #################
#############################################################

Random.seed!(0)
clst_n = 500 # number of points in a cluster 
k = 3
data = Array{Float64}(undef, 2, clst_n*k) # initial data array (clst_n*k)*2 
label = Array{Float64}(undef, clst_n*k) # label is empty vector 1*(clst_n*k)
mu = reshape(sample(1:20, k*2), k, 2) #[5 4; 2 1; 10 3]
# sig = [[0.7 0; 0 0.7],[1.5 0;0 1.5],[0.2 0;0 0.6]]
# we can not do with a = [a, i] refer to Scope of Variables in julia documentation
for i = 1:k 
    sig = round.(sig_gen(sample(1:5, 2)))
    print(sig)
    clst = rand(MvNormal(mu[i,:], sig), clst_n) # data is 2*clst_n
    data[:,((i-1)*clst_n+1):(i*clst_n)] = clst
    label[((i-1)*clst_n+1):(i*clst_n)] = repeat([i], clst_n)
end

k = nlabel(label) #length(unique(label))
label = convertlabel(1:k, vec(label))


# plot the original data
pyplot()
scatter(data[1,:], data[2,:], markercolor=label, legend = false, title = "My Scatter Plot")


result = kmeans(data, k)

# local optimization for kmeans clustering
centers_l, assign_l, objv_l = local_OPT(data, k)

lb = lb_par.LD_2_par_test(data, k, result.centers)
@time test_ld_pl = lb_par.getLowerBound_LD_par(data, k, result.centers)

# branch&bound global optimization for kmeans clustering
t = @elapsed centers, objv, calcInfo = bb_par.branch_bound_par(data, k, "LD+adaGp")

@time test_ctrl = lb_functions.getLowerBound_Test(data, k, result.centers) # 129s
@time test_ctrl_pl = lb_par.getLowerBound_Test_par(data, k, result.centers) # 30.85s
@time test_ctrl_pl2 = lb_par.getLowerBound_Test_par2(data, k, result.centers) # 31.94s
@time test_ctrl_pl3 = lb_par.getLowerBound_Test_par3(data, k, result.centers) # 30.71s
@time test_ctrl_pl3 = lb_par.getLowerBound_LD_par(data, k, result.centers) # 30.71s
# ~, assign = obj_assign(result.centers, data);
# d, n = size(data);
# ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
# groups = lb_functions.kmeans_group(data, assign, ngroups)
# @time test_adpt = lb_functions.getLowerBound_adptGp(data, k, result.centers, groups, nothing, nothing, test_ctrl)
# @time test_ld = lb_functions.getLowerBound_LD(data, k, result.centers)


rmprocs(procs()[2:nprocs()])