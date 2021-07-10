module opt_functions

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX, Gurobi#, SCIP
using Random
using InteractiveUtils

export obj_assign, local_OPT, global_OPT_base, global_OPT3_LD

time_lapse = 900 # 15 mins
obbt_time = 180 # 1 mins

############# auxilary functions #############
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



function init_bound(X, d, k, lower=nothing, upper=nothing)
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d # get the feasible region of center
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)

    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
    return lower, upper
end



function max_dist(X, d, k, n, lower, upper)
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
    return dmat_max
end



############# local optimization function (Ipopt) #############
function local_OPT(X, k, lower=nothing, upper=nothing)
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)

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



############# global optimization solvers #############
# pure cplex solvers
function global_OPT_base(X, k, lower=nothing, upper=nothing, mute=false)
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse*16) # maximum runtime limit is time_lapse*16 or set to 4/12 hours
    set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 0) # 0 for qcp relax and 1 for lp oa relax.
    # set_optimizer_attribute(m, "MIQCPMethod", 1) # 0 for qcp relax and 1 for lp oa relax, -1 for auto This is for Gurobi
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
    node = node_count(m)
    gap = relative_gap(m) # get the relative gap for cplex solver
    objv, ~ = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    return centers, objv, node, gap
end



# reduced bb subproblem solvers with largranian decomposition
# here the labmda is the largrange multiplier
function global_OPT3_LD(X, k, lambda, ctr_init, w_sos=nothing, lower=nothing, upper=nothing, mute=false, solver="CPLEX")
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)
    
    #w_bin = rlt.centers # weight of the binary variables
    if solver=="CPLEX"
        m = Model(CPLEX.Optimizer);
        if mute
            set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
        end
        set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse) # maximum runtime limit is 1 hours
        # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
        # set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.05) 
    else # solver is Gurobi
        m = Model(Gurobi.Optimizer);
        if mute
            set_optimizer_attribute(m, "OutputFlag", 0)
        end
        set_optimizer_attribute(m, "Threads",1)
        set_optimizer_attribute(m, "TimeLimit", time_lapse) # maximum runtime limit is 1 hours
        # set_optimizer_attribute(m, "PreMIQCPForm", 0) # improve the speed of bb process
        # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
        # set_optimizer_attribute(m, "MIPGap", 0.05) 
    end
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=ctr_init[t,i]);
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j] - centers[t,i])^2 for t in 1:d ));

    #@constraint(m, [i in 1:k, j in 1:n], [dmat[i,j] X[:,j]-centers[:,i]] in SecondOrderCone())
    #@constraint(m, [i in 1:k, j in 1:n], [dmat[i,j]; X[:,j]-centers[:,i]] in SecondOrderCone())
    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1);
    @constraint(m, [j in 1:n], b[:,j] in MOI.SOS1(w_sos[:,j])) #SOS1 constraint

    @variable(m, costs[1:n]>=0, start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-b[i,j]))

    @objective(m, Min, sum(costs[j] for j in 1:n)+
                sum((lambda[:,i,2]-lambda[:,i,1])'*centers[:,i] for i in 1:k));
    optimize!(m);


    centers = value.(centers)
    objv = objective_bound(m)
    return centers, objv
end

end 

