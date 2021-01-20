module lb_functions

using Random, Distributions
using LinearAlgebra
using MLDataUtils, Clustering #, MLBase
using opt_functions

#export getLowerBound_adptGp


############## Original lower bound calculation ##############

# first original function for calcualting lower bound
function getLowerBound_Test(X, k, centers, lower=nothing, upper=nothing)
    ~, assign = obj_assign(centers, X); # objv as a qUB
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    
    groups = [[] for i=1:ngroups];
    ng = 0; # heuristic improve grouping 
    for i = 1:k
    	cid = findall(assign.==i);
        for j = 1:length(cid)	
            if ng == ngroups
                ng = 0;
            end	  
            ng += 1;
            push!(groups[ng], cid[j]);
        end
    end

    # println(groups)

    LB = 0
    for i = 1:ngroups
    	centers, assign, objv = global_OPT3(X[:,groups[i]], k, lower, upper, true);
	    LB += objv;
    end
    return LB
end


############## Lower bound calculation with largrangean decomposition ##############

# function for updating the largrange multiplier Lambda
# centers is the set of all centers achieved from each global_OPT3
function updateLambda_subgrad(lambda_vec, qUB, LB, centers_gp, alpha=1)
    d, k, ngroups = size(centers_gp)
    sk = zeros(d, k, ngroups)
    for i in 1:ngroups
        if i == ngroups
            sk[:,:,i] = centers_gp[:,:,i]-centers_gp[:,:,1]
        else
            sk[:,:,i] = centers_gp[:,:,i]-centers_gp[:,:,i+1]
        end
    end
    #println(sk)
    sk_vec = vec(sk)
    step = alpha*(qUB-LB)/(norm(sk_vec)^2)*sk_vec # norm may change to sum of squared sk
    lambda_vec = lambda_vec + step
    return reshape(lambda_vec, d, k, ngroups) # change back with d*k*g
end

# function for largrangean decomposition with lambda updates
function LD(X, d, k, ngroups, groups, qUB, lower=nothing, upper=nothing)
    lambda = zeros(d, k, ngroups) # d*k*g initialize lambda with 0
    # inital calculation for LB with zero lambda
    LB = 0
    # start LB caculation with lambda updating process
    maxLB = -Inf
    i = 0
    alpha = 1
    while  (alpha >= 1.0e-6) & (LB >= maxLB) # norm(lambda, Inf) > 0.1 # (qUB-maxLB)/min(abs(qUB)) >= 0.01 # 
        if LB > qUB
            maxLB = LB
            break
        end
        maxLB = LB
        #println("============lambda============")
        #println(lambda)
        # here lambda input is a vectors but output is the matrix    
        LB = 0
        centers_gp = zeros(d, k, ngroups) # initial var to save centers for each group
        for i = 1:ngroups
            if i == ngroups
                centers, assign, objv = opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                    lambda[:,:,[i, 1]], lower, upper, true); # here when i=ng, i+1 should be 1
            else
                centers, assign, objv = opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                    lambda[:,:,i:(i+1)], lower, upper, true); # here when i=ng, i+1 should be 1
            end
            centers_gp[:,:,i] = centers
            LB += objv;
        end
        # update lambda before the new loop
        lambda = updateLambda_subgrad(vec(lambda), qUB, LB, centers_gp, alpha) 
        alpha = 0.6*alpha
        println(LB)
        i += 1
    end

    return maxLB
end


function updateLambda_subgrad_2(lambda_vec, qUB, LB, centers_gp, alpha = 1)
    d, k, ngroups = size(centers_gp)
    sk = zeros(d, k, ngroups-1) # step size will be d*k*(ngroups-1)
    for i in 1:(ngroups-1)
        sk[:,:,i] = centers_gp[:,:,i]-centers_gp[:,:,i+1]
    end
    #println(sk)
    println("alpha:", alpha)
    sk_vec = vec(sk)
    step = alpha*(qUB-LB)/(norm(sk_vec)^2)*sk_vec # norm may change to sum of squared sk
    lambda_vec = lambda_vec + step
    return reshape(lambda_vec, d, k, ngroups-1) # change back with d*k*(g-1)
end

function LD_2(X, d, k, ngroups, groups, qUB, lower=nothing, upper=nothing)
    lambda = zeros(d, k, ngroups+1) # d*k*(g+1) initialize lambda with 0
    # inital calculation for LB with zero lambda
    LB = 0
    # start LB caculation with lambda updating process
    maxLB = -Inf
    #i = 0
    alpha = 0.5
    while  (alpha >= 1.0e-6) & (LB >= maxLB) # norm(lambda, Inf) > 0.1 # (qUB-maxLB)/min(abs(qUB)) >= 0.01 # 
        if LB > qUB
            maxLB = LB
            break
        end
        maxLB = LB
        println("============lambda============")
        println(lambda)
        # here lambda input is a vectors but output is the matrix    
        LB = 0
        centers_gp = zeros(d, k, ngroups) # initial var to save centers for each group
        
        # lambda dimensions: d*k*(ngroups+1) [0-ngroups]
        # lambda[:,:,1] --> lambda_0, lambda[:,:,ng+1] --> lambda_ng, both are 0 here.
        for i = 1:ngroups
            centers, assign, objv = opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                lambda[:,:,i:(i+1)], lower, upper, true);
            centers_gp[:,:,i] = centers; # i for groups and centers are corresponding to i+1 of lambda
            LB += objv;
        end
        # update lambda before the new loop
        # here we only need to update lambda[:,:,2:ngroups] (actaully is 1:(ngroups-1))
        lambda[:,:,2:ngroups] = updateLambda_subgrad_2(vec(lambda[:,:,2:ngroups]), qUB, LB, centers_gp, alpha)
        alpha = 0.5*alpha 
        println(LB)
        #i += 1
    end

    return maxLB
end

# first original function for calcualting lower bound
function getLowerBound_LD(X, k, centers, lower=nothing, upper=nothing)
    obj_ub, assign = obj_assign(centers, X); # objv as a qUB
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    
    groups = [[] for i=1:ngroups];
    ng = 0; # heuristic improve grouping 
    for i = 1:k
    	cid = findall(assign.==i);
        for j = 1:length(cid)	
            if ng == ngroups
                ng = 0;
            end	  
            ng += 1;
            push!(groups[ng], cid[j]);
        end
    end

    # largrangean decomposition
    LB = LD_2(X, d, k, ngroups, groups, obj_ub, lower, upper)
    
    return LB
end

############## Lower bound calculation with adaptive sub-grouping ##############

#import Base: length

function unique_inverse(A::AbstractArray)
    out = Array{eltype(A)}(undef, 0)
    out_idx = Array{Vector{Int}}(undef, 0)
    seen = Dict{eltype(A), Int}()
    for (idx, x) in enumerate(A)
        if !in(x, keys(seen))
            seen[x] = length(seen) + 1
            push!(out, x)
            push!(out_idx, Int[])
        end
        push!(out_idx[seen[x]], idx)
    end
    out, out_idx
end

# this is the function for grouping a cluster, and can avoid failure when a sub-cluster has size < ngroups
function strGrp_nofill(assign, ngroups)
    l, c = unique_inverse(assign) # get label and cluster index
    groups = [[] for i=1:ngroups]
    ng = 0
    for i in l
        p = c[i][randperm(length(c[i]))] # first shuffling the index to introduce the randomness
        for j = 1:length(p)	
            if ng == ngroups
                ng = 0;
            end	  
            ng += 1;
            push!(groups[ng], p[j]);
        end
    end
    return groups
end

# grouping function that stratified on assign and select data evenly by applying kmeans clustering on each cluster
# the sampling will have a bug in kmeans grouping with sample function when the smallest cluster only has p data points, which p < 2*ngroups
function kmeans_group(X, assign, ngroups)
    clst_label, clst_idx = unique_inverse(assign)
    # number of sub-cluster for each cluster, k_sub <= clst_size/ngroups, which is length(clst_idx[i])/ngroups
    # we have to check if the cluster size p is too small that p < ngroups, just set k_sub as 2, 
    # here we may occur that points this cluster may not fall in all groups, aka, # of points in each sub-cluster < ngroups.
    # if floor(Int, minimum(length.(clst_idx))/ngroups/2) <= 1
    #     k_sub = 2
    # else
    #     k_sub = sample(2:floor(Int, minimum(length.(clst_idx))/ngroups/2)) # this can only guarantee at least one cluster can be grouped into ngroups. Do we need to divid 2?
    # end
    groups = [[] for i=1:ngroups]
    for i in clst_label
        # using floor to guarantee one cluster can assign points to every group
        k_sub = floor(Int, length(clst_idx[i])/ngroups) # get number of sub-cluster for each cluster, the higher the k_sub, the more sparse of the subgroup
        if k_sub <=1
            k_sub = 2
        end
        clst_rlt = kmeans(X[:,clst_idx[i]], k_sub) # get sub-cluster from one cluster
        #println(clst_rlt.assignments)
        clst_groups = strGrp_nofill(clst_rlt.assignments, ngroups) # get the grouping label from one cluster
        for j in 1:length(clst_groups) # we put index of points in sub-clusters from cluster i that belongs to group j
            append!(groups[j], clst_idx[i][clst_groups[j]]);
        end
    end
    return groups
end

# function with kmeans grouping but has a guarantee with accending lower bound
function getLowerBound_adptGp(X, k, centers, parent_groups=nothing, lower=nothing, upper=nothing,  glbLB=-Inf)
    # first generate new grouping based the assignment of current centers
    ~, assign = obj_assign(centers, X);
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    groups = kmeans_group(X, assign, ngroups)
    # calculate the lower bound
    LB = 0
    for i = 1:ngroups
    	centers, assign, objv = global_OPT3(X[:,groups[i]], k, lower, upper, true);
	    LB += objv;
    end

    # check if LB with new grouping lower than the LB of parent node
    if LB < glbLB # if LB is smaller, than adopt the parent grouping
        groups = parent_groups
        # calculate the lower bound
        LB = 0
        for i = 1:ngroups
            centers, assign, objv = global_OPT3(X[:,groups[i]], k, lower, upper, true);
            LB += objv;
        end
    end
    return LB, groups
end

############## Lower bound calculation with adaptive sub-grouping and largrangean decomposition ##############

function getLowerBound_adptGp_LD(X, k, centers, parent_groups=nothing, lower=nothing, upper=nothing,  glbLB=-Inf)
    # first generate new grouping based the assignment of current centers
    obj_ub, assign = obj_assign(centers, X);
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    groups = kmeans_group(X, assign, ngroups)
    
    # calculate the lower bound with largrangean decomposition
    LB = LD_2(X, d, k, ngroups, groups, obj_ub, lower, upper)

    # check if LB with new grouping lower than the LB of parent node
    if (LB < glbLB) || (LB > obj_ub) # if LB is smaller, than adopt the parent grouping
        groups = parent_groups
        # calculate the lower bound with largrangean decomposition
        LB = LD_2(X, d, k, ngroups, groups, obj_ub, lower, upper)
    end
    return LB, groups
end



# end of the module
end