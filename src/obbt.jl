module obbt

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX#, SCIP
using Random
using opt_functions

export OBBT_min, OBBT_max

time_lapse = 600 # 10 mins

# nlines represents 2*nlines lines added as the outer approximation for the problem
function OBBT_min(X, k, UB, lower=nothing, upper=nothing, mute=false, nlines = 1)
    d, n = size(X)
    lower, upper = opt_functions.init_bound(X, d, k, lower, upper)

    dmat_max = zeros(k,n)
    for j = 1:n
    	for i = 1:k
            max_distance = 0
            for t = 1:d
                max_distance += max((X[t,j]-lower[t,i])^2, (X[t,j]-upper[t,i])^2)
            end	
            dmat_max[i,j] = max_distance
	    end
    end    

    lwr_center = zeros(d,k) # initialize the lower bound of center
    println("Start OBBT(minimum) process:")
    # for each dimension and cluster, we have a variable to solve
    for dim in 1:d
        for clst in 1:k
            println("Solving the minimum bound of variable: centers[$dim, $clst].")
            m = Model(CPLEX.Optimizer);
            if mute
                set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
            end
            set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse) # maximum runtime limit is 10 mins
            @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
            @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])

            @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
            @variable(m, lower[t,i]^2 <= w[t in 1:d, i in 1:k] <= upper[t,i]^2) # add the horizontal line of the lower bottom line bound 
            @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
            itval = (upper-lower)./2/nlines # total 2*nlines, separate the range into 2*nlines sections
            for line in 0:(nlines-1)
                lwr = lower+itval.*line
                upr = upper-itval.*line
                @constraint(m, [t in 1:d, i in 1:k], 2*lwr[t,i]*centers[t,i]-lwr[t,i]^2 <= w[t,i])
                @constraint(m, [t in 1:d, i in 1:k], 2*upr[t,i]*centers[t,i]-upr[t,i]^2 <= w[t,i])
            end

            @variable(m, lambda[1:k, 1:n], Bin)
            @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
            @variable(m, costs[1:n], start=rand());
            @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-lambda[i,j]))
            @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-lambda[i,j]))
            @constraint(m, sum(costs[j] for j in 1:n)<= UB) # add the constraint that the total cost should lower than current UB

            @objective(m, Min, centers[dim, clst]);
            optimize!(m);
            #println(result_count(m))
            if result_count(m) >= 1
                lwr_center[dim, clst] = getobjectivevalue(m)
            else    
                lwr_center[dim, clst] = lower[dim, clst]
                println("No feasible solution found.")
            end
        end
    end

    return lwr_center
end


function OBBT_max(X, k, UB, lower=nothing, upper=nothing, mute=false, nlines = 1)
    d, n = size(X)
    lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    
    dmat_max = zeros(k,n)
    for j = 1:n
    	for i = 1:k
            max_distance = 0
            for t = 1:d
                max_distance += max((X[t,j]-lower[t,i])^2, (X[t,j]-upper[t,i])^2)
            end	
            dmat_max[i,j] = max_distance
	    end
    end    

    upr_center = zeros(d,k) # initialize the lower bound of center
    println("Start OBBT(maximum) process:")
    # for each dimension and cluster, we have a variable to solve
    for dim in 1:d
        for clst in 1:k
            println("Solving the maximum bound of variable: centers[$dim, $clst].")
            m = Model(CPLEX.Optimizer);
            if mute
                set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
            end
            set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse) # maximum runtime limit is 10 mins
            @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
            @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])

            @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
            @variable(m, lower[t,i]^2 <= w[t in 1:d, i in 1:k] <= upper[t,i]^2) # add the horizontal line of the lower bottom line bound 
            @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
            itval = (upper-lower)./2/nlines # total 2*nlines, separate the range into 2*nlines sections
            for line in 0:(nlines-1)
                lwr = lower+itval.*line
                upr = upper-itval.*line
                @constraint(m, [t in 1:d, i in 1:k], 2*lwr[t,i]*centers[t,i]-lwr[t,i]^2 <= w[t,i])
                @constraint(m, [t in 1:d, i in 1:k], 2*upr[t,i]*centers[t,i]-upr[t,i]^2 <= w[t,i])
            end

            @variable(m, lambda[1:k, 1:n], Bin)
            @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
            @variable(m, costs[1:n], start=rand());
            @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-lambda[i,j]))
            @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-lambda[i,j]))
            @constraint(m, sum(costs[j] for j in 1:n)<= UB) # add the constraint that the total cost should lower than current UB

            @objective(m, Max, centers[dim, clst]);
            optimize!(m);
            # println(result_count(m))
            if result_count(m) >= 1
                upr_center[dim, clst] = getobjectivevalue(m)
            else    
                upr_center[dim, clst] = upper[dim, clst]
                println("No feasible solution found.")
            end
        end
    end

    return upr_center
end


end