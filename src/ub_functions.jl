module ub_functions

using Clustering
using opt_functions


function getUpperBound(X, k, lower=nothing, upper=nothing, tol = 0)
    n_trial = 10
    if lower === nothing
        UB = Inf
        result = nothing
        for tr = 1:n_trial
            clst_rlt = kmeans(X, k)
            if clst_rlt.totalcost <= UB - tol
                UB = clst_rlt.totalcost
                result = clst_rlt
            end
        end
        centers = result.centers
        assign = result.assignments
    else
	    centers, assign, UB = local_OPT(X, k, lower, upper)
    end
    return centers, assign, UB
end


# end of the module
end

