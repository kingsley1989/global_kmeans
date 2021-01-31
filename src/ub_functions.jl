module ub_functions

using Clustering
using opt_functions


function getUpperBound(X, k, lower=nothing, upper=nothing, tol = 0)
    n_trial = 3
    if lower === nothing	
        result = kmeans(X, k)
    	UB = result.totalcost
    	centers = result.centers
        for i = 2:n_trial
            result = kmeans(X, k)
            if result.totalcost <= UB - tol # tol is the tolerance
                    UB = result.totalcost
                centers = result.centers
            end
        end
        #=for i = 1:n_trial
            centers_trial, ~, UB_trial = local_OPT(X, k)
            if UB_trial <= UB - tol
	            UB = UB_trial
                centers = centers_trial
            end
        end=#
    else
	    centers, ~, UB = local_OPT(X, k, lower, upper)
        for i = 2:n_trial
            print("*")
            centers_trial, ~, UB_trial = local_OPT(X, k, lower, upper)
            if UB_trial <= UB - tol
                UB = UB_trial
                centers = centers_trial
            end
        end
        println("")
    end
    return centers, UB
end



# end of the module
end

