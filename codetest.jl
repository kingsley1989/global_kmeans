using RDatasets, DataFrames, CSV
using Random, Distributions
using Plots, StatPlots
using Clustering, Distances
using MLDataUtils#, MLBase

# load functions for branch&bound and data preprocess from self-created module
if !("." in LOAD_PATH)
    push!(LOAD_PATH, ".")
end
using opt_functions, data_process, lb_functions



#############################################################
################# Main Process Program Body #################
#############################################################

Random.seed!(123)
clst_n = 50 # number of points in a cluster
nclst = 3 # number of cluster generated
data = Array{Float64}(undef, 2, 0) # empty array 0*2 
label = Array{Float64}(undef, 0, 1) # label is empty vector 1*0
mu = [5 4; 2 1; 10 3]
sig = [[0.7 0; 0 0.7],[1.5 0;0 1.5],[0.2 0;0 0.6]]
for i = 1:nclst
    clst = rand(MvNormal(mu[i,:], sig[i]), clst_n) # data is 2*clst_n
    data = hcat(data, clst)
    label = vcat(label, repeat([i], clst_n))
end

k = nlabel(vec(label)) #length(unique(label))
label = convertlabel(1:k, vec(label))
D = pairwise(Euclidean(), data) # calculate the distance matrix of data
D_ult = lb_functions.ultra_dist(D)

# plot the original data
pyplot()
scatter(data[1,:], data[2,:], markercolor=label, legend = false, title = "My Scatter Plot")


result = kmeans(data, k)
@time test_ctrl = lb_functions.getLowerBound_Test(data, k, result.centers)

~, assign = obj_assign(result.centers, data);
d, n = size(data);
ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
groups = lb_functions.kmeans_group(data, assign, ngroups)
@time test_adpt = lb_functions.getLowerBound_adptGp(data, k, result.centers, groups, nothing, nothing, test_ctrl)
@time test_ld = lb_functions.getLowerBound_LD(data, k, result.centers)