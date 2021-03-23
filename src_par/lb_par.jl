module lb_par

using Random, Distributions
using LinearAlgebra
using MLDataUtils, Clustering #, MLBase

using Distributed, SharedArrays

using lb_functions

@everywhere using opt_functions


#export getLowerBound_adptGp


############## Original lower bound calculation ##############
function LB_calc(X, k, groups, ngroups, lower, upper)
    @distributed (+) for i = 1:ngroups
    	global_OPT3(X[:,groups[i]], k, lower, upper, true)[2]
    end
end

# first original function for calcualting lower bound
function getLowerBound_Test_par(X, k, centers, lower=nothing, upper=nothing)
    
    ~, assign = opt_functions.obj_assign(centers, X); # objv as a qUB
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

    #println(ngroups)

    return LB_calc(X, k, groups, ngroups, lower, upper)
end



function LD_2_par(X, d, k, ngroups, groups, qUB, lower=nothing, upper=nothing)
    lambda = zeros(d, k, ngroups+1) # d*k*(g+1) initialize lambda with 0
    # inital calculation for LB with zero lambda
    LB = 0
    # start LB caculation with lambda updating process
    maxLB = -Inf
    #i = 0
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
        
        # rlt_gp: (center, objv)
        rlt_gp = pmap(i -> opt_functions.global_OPT3_LD(X[:,groups[i]], k, lambda[:,:,i:(i+1)], lower, upper, true), 1:ngroups)
        # centers
        for i in 1:length(rlt_gp)
            centers_gp[:, :, i] = rlt_gp[i][1]
        end
        # lower bound
        LB = @distributed (+) for rlt in rlt_gp
            rlt[2]
        end
        # update lambda before the new loop
        # here we only need to update lambda[:,:,2:ngroups] (actaully is 1:(ngroups-1))
        lambda[:,:,2:ngroups] = lb_functions.updateLambda_subgrad_2(vec(lambda[:,:,2:ngroups]), qUB, LB, centers_gp, alpha)
        alpha = 0.8*alpha 
        println(LB)
        #i += 1
    end

    return maxLB
end

# first original function for calcualting lower bound
function getLowerBound_LD_par(X, k, centers, lower=nothing, upper=nothing)
    obj_ub, assign = obj_assign(centers, X); # objv as a qUB
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group, 5*k for large problems (d*k>10 or n>1500)
    
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
    LB = LD_2_par(X, d, k, ngroups, groups, obj_ub, lower, upper)
    
    return LB
end



# function with kmeans grouping but has a guarantee with accending lower bound
function getLowerBound_adptGp_par(X, k, centers, parent_groups=nothing, lower=nothing, upper=nothing,  glbLB=-Inf)
    # first generate new grouping based the assignment of current centers
    ~, assign = obj_assign(centers, X);
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group, 5*k for large problems (d*k>10 or n>1500)
    groups = lb_functions.kmeans_group(X, assign, ngroups)
    # calculate the lower bound
    LB = LB_calc(X, k, groups, ngroups, lower, upper)

    # check if LB with new grouping lower than the LB of parent node
    if LB < glbLB # if LB is smaller, than adopt the parent grouping
        groups = parent_groups
        LB = max(LB, LB_calc(X, k, groups, ngroups, lower, upper))
    end
    return LB, groups
end



function getLowerBound_adptGp_LD_par(X, k, centers, parent_groups=nothing, lower=nothing, upper=nothing,  glbLB=-Inf)
    # first generate new grouping based the assignment of current centers
    obj_ub, assign = obj_assign(centers, X);
    d, n = size(X);
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group, 5*k for large problems (d*k>10 or n>1500)
    groups = lb_functions.kmeans_group(X, assign, ngroups)
    #println(length.(groups))
    # calculate the lower bound with largrangean decomposition
    LB = LD_2_par(X, d, k, ngroups, groups, obj_ub, lower, upper)

    # check if LB with new grouping lower than the LB of parent node
    if (LB < glbLB) #|| (LB > obj_ub) # if LB is smaller, than adopt the parent grouping
        groups = parent_groups
        # calculate the lower bound with largrangean decomposition
        LB_n = LD_2_par(X, d, k, ngroups, groups, obj_ub, lower, upper)
        LB = max(LB, LB_n)
    end
    return LB, groups
end











# first original function for calcualting lower bound
function getLowerBound_Test_par2(X, k, centers, lower=nothing, upper=nothing)
    
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

    X_gp = pmap(i -> X[:, groups[i]], 1:ngroups)
    LB = @distributed (+) for i = 1:ngroups
    	global_OPT3(X_gp[i], k, lower, upper, true)[2]
    end

    return LB
end



# first original function for calcualting lower bound
function getLowerBound_Test_par3(X, k, centers, lower=nothing, upper=nothing)
    
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

    objv_gp = pmap(i -> global_OPT3(X[:, groups[i]], k, lower, upper, true)[2], 1:ngroups)
    LB = @distributed (+) for i = 1:ngroups
    	objv_gp[i]
    end

    return LB
end


# end of the module
end