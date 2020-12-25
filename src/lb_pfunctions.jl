module lb_pfunctions

using Random, Distributions
using LinearAlgebra
using MLDataUtils, Clustering #, MLBase

using Distributed, SharedArrays

@everywhere using opt_functions


#export getLowerBound_adptGp


############## Original lower bound calculation ##############
function LB_calc(X, k, groups, ngroups, lower, upper)
    @distributed (+) for i = 1:ngroups
    	global_OPT3(X[:,groups[i]], k, lower, upper, true)[3]
    end
end

# first original function for calcualting lower bound
function getLowerBound_Test_parallel(X, k, centers, lower=nothing, upper=nothing)
    
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

    #println(ngroups)

    return LB_calc(X, k, groups, ngroups, lower, upper)
end



# first original function for calcualting lower bound
function getLowerBound_Test_parallel2(X, k, centers, lower=nothing, upper=nothing)
    
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
    	global_OPT3(X_gp[i], k, lower, upper, true)[3]
    end

    return LB
end



# first original function for calcualting lower bound
function getLowerBound_Test_parallel3(X, k, centers, lower=nothing, upper=nothing)
    
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

    objv_gp = pmap(i -> global_OPT3(X[:, groups[i]], k, lower, upper, true)[3], 1:ngroups)
    LB = @distributed (+) for i = 1:ngroups
    	objv_gp[i]
    end

    return LB
end


# end of the module
end