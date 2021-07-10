module lb_functions

using Random, Distributions
using LinearAlgebra, Statistics
using MLDataUtils, Clustering #, MLBase
#using opt_functions

using Distributed, SharedArrays
@everywhere using opt_functions

export getLowerBound_adptGp_LD, getGlobalLowerBound


tol = 1e-6
mingap = 1e-3



# function to calcuate the median value of a vector with 3 elements
function med(a,b,c)
    return a+b+c-max(a,b,c)-min(a,b,c)
end



function getGlobalLowerBound(nodeList) # if LB same, choose the first smallest one
    LB = 1e15
    nodeid = 1
    for (idx,n) in enumerate(nodeList)
    	#println("remaining ", idx,  "   ", n.LB)
        if n.LB < LB
            LB = n.LB
            nodeid = idx
        end
    end
    return LB, nodeid
end



############## Lower bound calculation with closed-form ##############
function getLowerBound_analytic(X, k, lower=nothing, upper=nothing)
    d, n = size(X)
    if lower === nothing
        lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    end
    # get the mid value of each mu for each cluster
    # start calculating the lower bound (distance of x_s to its closest mu)
    LB = 0
    centers_gp = rand(d, k, n)
    for s in 1:n
        # the way median is precalculated is faster 
        x_mat = repeat(X[:,s], 1, k)
        mu = med.(lower[:,:], x_mat[:,:], upper[:,:]) # solution for each scenario
        centers_gp[:,:,s] = mu
        min_dist = Inf
        for i in 1:k
            dist = sum((X[t,s] - mu[t,i])^2 for t in 1:d)
            if dist <= min_dist
                min_dist = dist
            end
        end
        LB += min_dist
    end
    return LB, centers_gp
end



############## auxilary functions for largrangean decomposition ##############

function updateLambda_subgrad_2(lambda_vec, sk_old_vec, qUB, LB, centers_gp, alpha = 1)
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
    hk = sk_vec'*sk_old_vec # covergence rate factor
    return reshape(lambda_vec, d, k, ngroups-1), sk_vec, hk # change back with d*k*(g-1)
end


function LD_2(X, d, k, ngroups, groups, qUB, w_sos=nothing, lambda=nothing, centers_init = nothing, lower=nothing, upper=nothing, solver="CPLEX")
    ncores = nprocs() 	 
    if lambda === nothing # use parent lambda as the initial guess, if not 
        lambda = zeros(d, k, ngroups+1) # d*k*(g+1) initialize lambda with 0
    end
    # start LB caculation with lambda updating process
    maxLB = -Inf
    #i = 0
    alpha = 1
    trial = 0
    trial_no_improve = 0
    maxtrial = 15 # maximum 15 iterations to update lambda
    maxtrial_no_improve = 3 # once we hit a non-improve on lambda, if this no-improve continue for 3 iteration, then stop
    group_centers = nothing # here is the group_centers can be used as the initial guess of the solution and for branching

    # initialize the step vector, step size will be d*k*(ngroups-1)
    sk_old_vec = vec(ones(d, k, ngroups-1)) 
    # initalize LB_old for checking LB increase or decrease
    LB_old = -Inf
    println(norm(lambda, 2))
    # here we can change alpha< 1e-6 to (LB - previous LB) <= mingap*0.1, have to smaller than mingap ? 
    while norm(sk_old_vec, 2) > 0.1 && (trial <= maxtrial) && (trial_no_improve <= maxtrial_no_improve) && (alpha >= 1.0e-6) #  # (qUB-maxLB)/min(abs(qUB)) >= 0.01 #    
        # here lambda input is a vectors but output is the matrix    
        LB = 0
        centers_gp = rand(d, k, ngroups) # zeros(d, k, ngroups) # initial var to save centers for each group
        LB_gp = zeros(1, ngroups) # initial group lb value
        # lambda dimensions: d*k*(ngroups+1) [0-ngroups]
        # lambda[:,:,1] --> lambda_0, lambda[:,:,ng+1] --> lambda_ng, both are 0 here.
        if centers_init === nothing
            centers_init = copy(centers_gp) # rand(d, k, ngroups) #
        end
        if ncores == 1
            for i = 1:ngroups
                # assign is not necessary
                print("=")  
                centers, objv = opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                            lambda[:,:,i:(i+1)], centers_init[:,:,i], w_sos, lower, upper, true, solver);
                LB_gp[i] = objv # update cuts info: opt lb value
                centers_gp[:,:,i] = centers; # i for groups and centers are corresponding to i+1 of lambda
                LB += objv;
            end
        else
            rlt_gp = pmap(i -> opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                        lambda[:,:,i:(i+1)], centers_init[:,:,i], w_sos, lower, upper, true, solver), 1:ngroups)
            # group centers and group lower bound
            for i in 1:length(rlt_gp)
                LB_gp[i] = rlt_gp[i][2] # update opt value
                centers_gp[:,:,i] = rlt_gp[i][1]
            end
            # lower bound
            LB = @distributed (+) for rlt in rlt_gp
                rlt[2]
            end
        end
        println("")
        # if we get a LB larger than current max LB, then 
        # 1. check if the gap between LB and maxLB is large, then it is an improve, so non-improve doesn't update
        # 2. check if the current LB is already close to UB enough, than no need to do further calculation and break
        if maxLB <= LB
            if (LB - maxLB) >= max(mingap, mingap*abs(qUB), 0.1*(abs(qUB)-maxLB))
                trial_no_improve = 0
            end 
            maxLB = LB
            group_centers = copy(centers_gp) 
            #group_cuts = cuts
            if (qUB-LB)<= mingap || (qUB-LB) <= mingap*abs(qUB)
                println(LB)
                break
            end
        end

        # update lambda before the new loop
        # here we only need to update lambda[:,:,2:ngroups] (actaully is 1:(ngroups-1))
        lambda[:,:,2:ngroups], sk_vec, hk = updateLambda_subgrad_2(vec(lambda[:,:,2:ngroups]), sk_old_vec, qUB, LB, centers_gp, alpha)
        if LB <= LB_old # red
            alpha = 0.66*alpha # here if #red = 1, means once we hit a decrease, we reduce the alpha
        else
            if hk >= 0 # green
                alpha = 1.1*alpha
            end # hk <0 is yellow and no change on alpha
        end
        sk_old_vec = sk_vec # update sk_old as pervious sk
        LB_old = LB

        println(LB)
        trial_no_improve += 1
        trial += 1
        #i += 1
    end
    return max(0, maxLB), lambda, group_centers
end


function GP(X, d, k, ngroups, groups, w_sos=nothing, lambda=nothing, centers_init = nothing, lower=nothing, upper=nothing, solver="CPLEX")
    ncores = nprocs() 	 
    if lambda === nothing # no LD, lambda always zero 
        lambda = zeros(d, k, ngroups+1) # d*k*(g+1) initialize lambda with 0
    end
    # println(norm(lambda, 2))
    centers_gp = rand(d, k, ngroups) # zeros(d, k, ngroups) # initial var to save centers for each group
    LB_gp = zeros(1, ngroups) # initial group lb value
    # lambda dimensions: d*k*(ngroups+1) [0-ngroups]
    # lambda[:,:,1] --> lambda_0, lambda[:,:,ng+1] --> lambda_ng, both are 0 here.
    if centers_init === nothing
        centers_init = copy(centers_gp) # rand(d, k, ngroups) # donot use copy to save the memory
    end
    if ncores == 1
        LB = 0
        for i = 1:ngroups
            # assign is not necessary
            print("=")  
            centers, objv = opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                        lambda[:,:,i:(i+1)], centers_init[:,:,i], w_sos, lower, upper, true, solver);
            LB_gp[i] = objv
            centers_gp[:,:,i] = centers; # i for groups and centers are corresponding to i+1 of lambda
            LB += objv;
            #println("group    ", i, "   objv:   ", objv, "     ",LB)
        end
    else
        rlt_gp = pmap(i -> opt_functions.global_OPT3_LD(X[:,groups[i]], k, 
                    lambda[:,:,i:(i+1)], centers_init[:,:,i], w_sos, lower, upper, true, solver), 1:ngroups)
        # group centers and group lower bound
        for i in 1:length(rlt_gp)
            centers_gp[:, :, i] = rlt_gp[i][1]
            LB_gp[i] = rlt_gp[i][2] # update cuts info: opt value
        end
        # lower bound
        LB = @distributed (+) for rlt in rlt_gp
            rlt[2]
        end
    end
    println("")
    return LB, nothing, centers_gp
end

############## auxilary functions for adaptive sub-grouping ##############

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
    return out, out_idx
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
    Random.seed!(123)
    ~, clst_idx = unique_inverse(assign) # clst_label is [1,2,3], clst_idx is [[1,2],[3,4]]
    groups = [[] for i=1:ngroups]
    
    # for i in clst_label: can not use clst_label since sometimes length(clst_idx) < max(clst_label) (usually equal)
    # for example: clst_idx: [[1,2] [3,4,5]] clst_label: [1,3]. 
    # This may happen if at particular node, some centers are too ridiculous that no sample are close to them.
    for i in 1:length(clst_idx) 
        if length(clst_idx[i]) == 1 # if this cluster only have one sample, put it directly to group 1
            append!(groups[1],clst_idx[i][1]) 
        else
            # using floor to guarantee one cluster can assign points to every group
            k_sub = floor(Int, length(clst_idx[i])/ngroups) # get number of sub-cluster for each cluster, the higher the k_sub, the more sparse of the subgroup
            if k_sub <=1
                k_sub = 2
            end
            # try 5 trials of kmeans to get the best assignment
            n_trial = 5
            mini_cost = Inf
            X_sub = X[:,clst_idx[i]]
            for tr = 1:n_trial
                clst_rlt = kmeans(X_sub, k_sub) # get sub-cluster from one cluster
                if clst_rlt.totalcost <= mini_cost
                    mini_cost = clst_rlt.totalcost
                    assign = clst_rlt.assignments
                end
            end
            #println(clst_rlt.assignments)
            clst_groups = strGrp_nofill(assign, ngroups) # get the grouping label from one cluster
            for j in 1:length(clst_groups) # we put index of points in sub-clusters from cluster i that belongs to group j
                append!(groups[j], clst_idx[i][clst_groups[j]]);
            end
        end
    end
    return groups
end



############## Lower bound calculation with adaptive sub-grouping and largrangean decomposition ##############
# centers here only used to generate the weight of the SOS1
#function getLowerBound_adptGp_LD(X, k, centers, parent_lambda = nothing, parent_groups=nothing, lower=nothing, upper=nothing,  glbLB=-Inf)
function getLowerBound_adptGp_LD(X, k, w_sos, assign, node = nothing, UB = Inf, mode = "fixed", solver = "CPLEX", LD = true)
    parent_lambda = node.lambda # initial guess of the lambda
    parent_centers = node.group_centers # initial guess of variable centers for each group
    parent_groups = node.groups 
    lower = node.lower
    upper = node.upper
    glbLB = node.LB
    # first generate new grouping based the assignment of current centers
    #~, assign = obj_assign(centers, X); # if the center is ridiculous, then obj function can not get a good ub
    obj_ub = UB
    d, n = size(X);
    ngroups = length(parent_groups)
    if mode == "fixed"
        groups = parent_groups
    else
        groups = kmeans_group(X, assign, ngroups)
    end

    if LD
        # calculate the lower bound with largrangean decomposition
        LB, lambda, group_centers = LD_2(X, d, k, ngroups, groups, obj_ub, w_sos, parent_lambda, parent_centers, lower, upper, solver)
    else
        LB, lambda, group_centers = GP(X, d, k, ngroups, groups, w_sos, parent_lambda, parent_centers, lower, upper, solver)
    end

    # check if LB with new grouping lower than the LB of parent node
    if (LB < glbLB) && (mode != "fixed") #|| (LB > obj_ub) # if LB is smaller, than adopt the parent grouping
        groups = parent_groups
        # calculate the lower bound with largrangean decomposition
        LB_n, lambda_n, group_centers_n, group_cuts_n = LD_2(X, d, k, ngroups, groups, obj_ub, w_sos, parent_lambda, parent_centers, lower, upper, solver)
        if LB_n > LB   
            lambda = lambda_n
            group_centers = group_centers_n
            group_cuts = group_cuts_n
            LB = LB_n
        end
    end
    GC.gc()
    return LB, groups, lambda, group_centers
end



# end of the module
end