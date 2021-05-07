module obbt

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX, Gurobi#, SCIP
using Random
#using opt_functions

using Distributed, SharedArrays
#@everywhere using opt_functions

#using grb_env

@everywhere using opt_functions

export OBBT_min, OBBT_max

# time_lapse = 60 # 1 mins


# nlines represents 2*nlines lines added as the outer approximation for the problem
function OBBT_min(X, k, UB, w_sos, lower=nothing, upper=nothing, mute=false, nlines = 1, solver="CPLEX")
    d, n = size(X)
    lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    dmat_max = opt_functions.max_dist(X, d, k, n, lower, upper)  
    
    lwr_center = zeros(d,k) # initialize the lower bound of center
    println("Start OBBT(minimum) process:")
    # for each dimension and cluster, we have a variable to solve
    if nprocs() == 1
        for dim in 1:d
            for clst in 1:k
                println("Solving the minimum bound of variable: centers[$dim, $clst].")
                
                opt_val = opt_functions.obbt_OPT(X, k, d, n, UB, w_sos, dmat_max, lower, upper, mute, nlines, solver, "min", dim, clst) 
                #println(result_count(m))
                if opt_val > lower[dim, clst]
                    lwr_center[dim, clst] = opt_val
                else
                    lwr_center[dim, clst] = lower[dim, clst]
                    println("No better bound found.")
                end
                #=
                ~, sol, ~ = opt_functions.global_OPT_oa(X, k, UB, w_sos, lower, upper, mute, 3, time_lapse, solver, "min", dim, clst)
                if sol > lower[dim, clst]
                    lwr_center[dim, clst] = sol
                else    
                    lwr_center[dim, clst] = lower[dim, clst]
                    println("No better bound found.")
                end
                =#

            end
        end
    else
        obbt_lwr = pmap(set -> opt_functions.obbt_OPT(X, k, d, n, UB, w_sos, dmat_max, lower, upper, 
                                    mute, nlines, solver, "min", set[1], set[2]), [(x,y) for x in 1:d, y in 1:k])
        lwr_center = max.(lower, obbt_lwr) # getindex.(rlt_obbt, 2)
    end

    return lwr_center
end


function OBBT_max(X, k, UB, w_sos, lower=nothing, upper=nothing, mute=false, nlines = 1, solver="CPLEX")
    d, n = size(X)
    lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    dmat_max = opt_functions.max_dist(X, d, k, n, lower, upper)

    upr_center = zeros(d,k) # initialize the lower bound of center
    println("Start OBBT(maximum) process:")
    # for each dimension and cluster, we have a variable to solve
    if nprocs() == 1
        for dim in 1:d
            for clst in 1:k
                println("Solving the maximum bound of variable: centers[$dim, $clst].")
                
                opt_val = opt_functions.obbt_OPT(X, k, d, n, UB, w_sos, dmat_max, lower, upper, mute, nlines, solver, "max", dim, clst) 
                # println(result_count(m))
                if opt_val < upper[dim, clst]
                    upr_center[dim, clst] = opt_val
                else    
                    upr_center[dim, clst] = upper[dim, clst]
                    println("No better bound found.")
                end
                #=
                ~, LB, ~ = opt_functions.global_OPT_oa(X, k, UB, w_sos, lower, upper, mute, 3, solver, "max", dim, clst)
                if LB < upper[dim, clst]
                    upr_center[dim, clst] = LB
                else    
                    upr_center[dim, clst] = upper[dim, clst]
                    println("No better bound found.")
                end
                =#

            end
        end
    else
        obbt_upr = pmap(set -> opt_functions.obbt_OPT(X, k, d, n, UB, w_sos, dmat_max, lower, upper, 
                                    mute, nlines, solver, "max", set[1], set[2]), [(x,y) for x in 1:d, y in 1:k])
        upr_center = min.(upper, obbt_upr) # getindex.(rlt_obbt, 2)
    end

    return upr_center
end


end