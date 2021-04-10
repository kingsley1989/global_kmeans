module ub_functions

using Clustering
using opt_functions


function getUpperBound(X, k, lower=nothing, upper=nothing, tol = 0)
    n_trial = 10
    if lower === nothing
        UB = Inf
        result = nothing
        for tr = 1:n_trial
            clst_rlt = kmeans(X, k) # get sub-cluster from one cluster
            if clst_rlt.totalcost <= UB - tol
                UB = clst_rlt.totalcost
                result = clst_rlt
            end
        end
        centers = result.centers
        assign = result.assignments
        #ind = sortperm(centers[1,:])
        #centers = centers[:, ind]
        #=for i = 1:n_trial
            centers_trial, ~, UB_trial = local_OPT(X, k)
            if UB_trial <= UB - tol
	            UB = UB_trial
                centers = centers_trial
            end
        end=#
    else
	    centers, assign, UB = local_OPT(X, k, lower, upper)
        #= for i = 2:n_trial
            print("*")
            centers_trial, ~, UB_trial = local_OPT(X, k, lower, upper)
            if UB_trial <= UB - tol
                UB = UB_trial
                centers = centers_trial
            end
        end 
        println("") =#
    end
    return centers, assign, UB
end



# end of the module
end

