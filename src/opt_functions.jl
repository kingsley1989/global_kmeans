module opt_functions

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX#, SCIP
using Random

export obj_assign, local_OPT, global_OPT3, global_OPT_base, global_OPT_linear


function obj_assign(centers, X)
    d, n = size(X)   	 
    k = size(centers, 2)	 
    dmat = zeros(k, n)
    for i=1:n
    	for j = 1:k
    	    dmat[j,i] = sum((X[:,i] .- centers[:,j]).^2) # dmat[j,i] is the distance from point i to center j
	    end    
    end

    assign = Vector{Int}(undef, n)
    costs = Vector{Float64}(undef, n)
    for j = 1:n
        c, a = findmin(dmat[:, j]) # find the closest cluster that point j belongs to
	    assign[j] = a # a is the cluster label of point j
        costs[j] = c # c is the distance of point j to center of cluster a
    end    
    return sum(costs), assign # sum costs is the total sse, assign is the current clustering assignment
end


function local_OPT(X, k, lower=nothing, upper=nothing)
    d, n = size(X)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d # get the feasible region of center
        lower_data[i] = minimum(X[i,:])
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k)
    upper_data = repeat(upper_data, 1, k)

    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end

    m = Model(Ipopt.Optimizer);
    set_optimizer_attribute(m, "print_level", 0);
    @variable(m, lower[i,j] <= centers[i in 1:d, j in 1:k] <= upper[i,j], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    @variable(m, dmat[1:k,1:n], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
    @variable(m, 0<=lambda[1:k,1:n]<=1,	start=rand());
    @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
    @variable(m, costs[1:n], start=rand());
    @constraint(m, [j in 1:n], costs[j] == sum(lambda[i,j]*dmat[i,j] for i in 1:k));
    @objective(m, Min, sum(costs[j] for j in 1:n));
    optimize!(m);
    centers = value.(centers)
    objv, assign = obj_assign(centers, X)
    #objv = getobjectivevalue(m)
    return centers, assign, objv
end

function global_OPT_base(X, k, lower=nothing, upper=nothing, mute=false)
    d, n = size(X)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d
        lower_data[i] = minimum(X[i,:])
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k)
    upper_data = repeat(upper_data, 1, k)
    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
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

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 43200) # maximum runtime limit is 4 hours
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
    @variable(m, lambda[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
    @variable(m, costs[1:n], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-lambda[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-lambda[i,j]))

    @objective(m, Min, sum(costs[j] for j in 1:n));
    optimize!(m);
    centers = value.(centers)
    gap = relative_gap(m) # get the relative gap for cplex solver
    objv, assign = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    return centers, objv, assign, gap
end

# nlines represents 2*nlines lines added as the outer approximation for the problem
function global_OPT_linear(X, k, lower=nothing, upper=nothing, mute=false, nlines = 3)
    d, n = size(X)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d
        lower_data[i] = minimum(X[i,:])
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k)
    upper_data = repeat(upper_data, 1, k)
    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
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

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 43200) # maximum runtime limit is 4 hours
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

    @objective(m, Min, sum(costs[j] for j in 1:n));
    optimize!(m);
    centers = value.(centers)
    gap = relative_gap(m) # get the relative gap for cplex solver
    objv, assign = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    return centers, objv, assign, gap
end

function global_OPT3(X, k, lower=nothing, upper=nothing, mute=false)
    d, n = size(X)
    d, n = size(X)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d
        lower_data[i] = minimum(X[i,:])
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k)
    upper_data = repeat(upper_data, 1, k)
    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
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

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 900) # maximum runtime limit is 1 hours
    set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.1) # set the relative gap to 0.1
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
    @variable(m, lambda[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
    @variable(m, costs[1:n], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-lambda[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-lambda[i,j]))

    @objective(m, Min, sum(costs[j] for j in 1:n));
    optimize!(m);
    centers = value.(centers)
    #objv, assign = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    # get the real objective of the lower bound problem and no need to get he value assign
    objv = getobjectivevalue(m)
    return centers, objv
end


# here the labmda is the largrange multiplier
function global_OPT3_LD(X, k, lambda, lower=nothing, upper=nothing, mute=false)
    d, n = size(X)
    d, n = size(X)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d
        lower_data[i] = minimum(X[i,:])
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k)
    upper_data = repeat(upper_data, 1, k)
    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
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

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", 900) # maximum runtime limit is 1 hours
    set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.1) # set the relative gap to 0.1
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));
    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1);
    @variable(m, costs[1:n], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-b[i,j]))

    @objective(m, Min, sum(costs[j] for j in 1:n)+
                sum((lambda[:,i,2]-lambda[:,i,1])'*centers[:,i] for i in 1:k));
    optimize!(m);
    centers = value.(centers)
    #objv, assign = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    # get the real objective of the lower bound problem and no need to get he value assign
    objv = getobjectivevalue(m)
    return centers, objv
end

end 

