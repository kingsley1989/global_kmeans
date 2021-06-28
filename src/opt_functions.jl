module opt_functions

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX, Gurobi#, SCIP
using Random
using grb_env
using InteractiveUtils

export obj_assign, local_OPT, global_OPT3, global_OPT_base, global_OPT_linear, global_OPT3_LD, global_OPT_oa, global_OPT_oa_base

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


# nlines represents 2*nlines lines added as the outer approximation for the problem
function linear_OPT(X, k, lower=nothing, upper=nothing, mute=false, nlines = 3, time = 900)
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
    end
    set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", time) # maximum runtime limit is 4 hours
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])

    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    #@variable(m, lower[t,i]^2 <= w[t in 1:d, i in 1:k] <= upper[t,i]^2) # add the horizontal line of the lower bottom line bound 
    @variable(m, 0 <= w[t in 1:d, i in 1:k], start=rand()) # add the horizontal line of the lower bottom line bound 
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
    itval = (upper-lower)./2/nlines # total 2*nlines, separate the range into 2*nlines sections
    for line in 0:(nlines-1)
        lwr = lower+itval.*line
        upr = upper-itval.*line
        @constraint(m, [t in 1:d, i in 1:k], 2*lwr[t,i]*centers[t,i]-lwr[t,i]^2 <= w[t,i])
        @constraint(m, [t in 1:d, i in 1:k], 2*upr[t,i]*centers[t,i]-upr[t,i]^2 <= w[t,i])
    end
    # add constraint for upper bound of w, may not necessary
    # @constraint(m, [t in 1:d, i in 1:k], w[t,i] <= (upper[t,i]+lower[t,i])*centers[t,i]-upper[t,i]*lower[t,i])

    @variable(m, lambda[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(lambda[i,j] for i in 1:k) == 1);
    @variable(m, costs[1:n], start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-lambda[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-lambda[i,j]))

    @objective(m, Min, sum(costs[j] for j in 1:n));
    optimize!(m);
    if result_count(m) >= 1
        centers = value.(centers)
        LB = objective_bound(m) #getobjectivevalue(m)
    else # in bb process, there may be infeasible in some node, and thus no LB and center exist.
        center = zeros(d,k) 
        LB = Inf
        println("No feasible solution found. This node is fathomed.")
    end
    return centers, LB, m
end

# this function is the pure linear relaxiation solver for mssc problem
function global_OPT_linear(X, k, lower=nothing, upper=nothing, mute=false, nlines = 3)
    centers, LB, m = linear_OPT(X, k, lower, upper, mute, nlines, 14400)
    if LB != Inf # only have result when solution exists
        gap = relative_gap(m) # get the relative gap for cplex solver
        objv, assign = obj_assign(centers, X) # here the objv should be a lower bound of CPLEX
    else # no solution, printout error message
        println("No solution for global optimization of linearized relaxation.")
        center = Inf
        objv = Inf
        assign = Inf
        gap = Inf
    end
    return centers, objv, assign, gap
end


# oa problem initialization functions
function oa_init_OPT(X, k, d, n, w_sos, dmat_max, lower, upper, mute=false, solver="CPLEX", time = 180)
    println("Initialize linear relaxed problem.")
    if solver == "CPLEX"
        m = Model(CPLEX.Optimizer);
        if mute
            set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
        end
        set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", time) # maximum runtime limit is 10 mins
        set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 1) # 0 for qcp relax and 1 for lp oa relax.
    else # Gurobi
        m = Model(() -> Gurobi.Optimizer(GUROBI_ENV[]))
        if mute
            set_optimizer_attribute(m, "OutputFlag", 0)
        end
        set_optimizer_attribute(m, "Threads",1)
        set_optimizer_attribute(m, "TimeLimit", time_lapse) # maximum runtime limit is 1 hours
        set_optimizer_attribute(m, "MIQCPMethod", 1) # 0 for qcp relax and 1 for lp oa relax, -1 for auto This is for Gurobi
        set_optimizer_attribute(m, "NoRelHeurWork", Inf) # set NoRel for fast searching of good solution
    end

    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])
    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    @variable(m, 0 <= w[t in 1:d, i in 1:k], start=rand()) # add the horizontal line of the lower bottom line bound 
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
    
    # initial linear relaxiation constraints
    @constraint(m, [t in 1:d, i in 1:k], 2*lower[t,i]*centers[t,i]-lower[t,i]^2 <= w[t,i])
    @constraint(m, [t in 1:d, i in 1:k], 2*upper[t,i]*centers[t,i]-upper[t,i]^2 <= w[t,i])
    # binary constraints
    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1);
    if w_sos !== nothing
        @constraint(m, [j in 1:n], b[:,j] in MOI.SOS1(w_sos[:,j]))
    end

    @variable(m, costs[1:n]>=0, start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-b[i,j]))
    
    return m #centers_fix, LB, assign, (UB-LB)/min(abs(LB), abs(UB))
end

# global oa functions that adpatively adding linear constraints
# w and centers from previous solver is quaried for checking violations
function global_OPT_oa(X, k, UB, w_sos=nothing, lower=nothing, upper=nothing, mute=false, max_iter = 5, time=180, solver="CPLEX", obbt = nothing, dim=0, clst=0)
    eps_v = 1e-5 # violations tolarance between w and quadratic term
    d, n = size(X)
    lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    dmat_max = opt_functions.max_dist(X, d, k, n, lower, upper)

    m = oa_init_OPT(X, k, d, n, w_sos, dmat_max, lower, upper, mute, solver, time)
    if obbt == "min"
        @objective(m, Min, m[:centers][dim, clst]);
        fea_sol = UB
        UB = upper[dim, clst]
        @constraint(m, sum(m[:costs][j] for j in 1:n)<= fea_sol) # add the constraint that the total cost should lower than current UB
    elseif obbt == "max"
        @objective(m, Max, m[:centers][dim, clst]);
        fea_sol = UB
        UB = lower[dim, clst]
        @constraint(m, sum(m[:costs][j] for j in 1:n)<= fea_sol)
    else # obbt == nothing, this is the oa for main problem
        @objective(m, Min, sum(m[:costs][j] for j in 1:n));
    end

    w_fix = nothing
    centers_fix = nothing
    if obbt == "max"
        LB = -Inf
    else
        LB = Inf
    end
    iter = 1
    # start adding constraints
    while (abs(UB-LB)/min(abs(LB), abs(UB))>= 0.01) & (iter <= max_iter)
        println("Solving linear relaxed problem $iter")
        # adding constraints, if w violates the quadratic term, add the linear constraint
        if w_fix !== nothing
            for t in 1:d
                for i in 1:k
                    if w_fix[t, i] < centers_fix[t,i]^2 - eps_v # if violates by at least eps_v
                        v = centers_fix[t, i]
                        @constraint(m, 2*v*(m[:centers][t,i])-v^2 <= (m[:w][t,i]))
                        println("Constraint is added at $v for cluster $i and dimension $t.")
                    end
                end
            end
        end
        optimize!(m)
        # check if there are solutions found.
        if has_values(m) #result_count(m) >= 1
            centers_fix = value.(m[:centers])
            w_fix = value.(m[:w])
            if obbt === nothing
                tLB = objective_bound(m) # always use the lower bound of the relaxed problem
            else
                tLB = objective_value(m)
            end
            if relative_gap(m) <= 0.01
                println("Lower bound after being converged: $tLB")
            else # else 
                println("Lower bound without being converged: $tLB")
            end
        else # in bb process, there may be infeasible in some node, and thus no LB and center exist.
            println("No feasible solution found at iter $iter")
            
        end
        if obbt == "max"
            if LB > tLB
                LB = tLB
            end
        else
            if LB < tLB # update LB so that LB is always the highest LB
                LB = tLB
            end
        end
        iter += 1
    end
    return centers_fix, LB, (UB-LB)/min(abs(LB), abs(UB))
end

function global_OPT_oa_base(X, k, UB, lower=nothing, upper=nothing, mute=false, max_iter = 5)
    centers, LB, gap = global_OPT_oa(X, k, UB, nothing, lower, upper, mute, max_iter, 14400)
    objv, assign = obj_assign(centers, X)
    return centers, objv, assign, gap
end


function obbt_OPT(X, k, d, n, UB, w_sos, dmat_max, lower=nothing, upper=nothing, mute=false, nlines=1, solver="CPLEX", obbt="min", dim=0, clst=0)
    if solver == "CPLEX"
        m = Model(CPLEX.Optimizer);
        if mute
            set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0)
        end
        set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
        set_optimizer_attribute(m, "CPX_PARAM_TILIM", obbt_time) # maximum runtime limit is 10 mins
        set_optimizer_attribute(m, "CPX_PARAM_MIQCPSTRAT", 1) # 0 for qcp relax and 1 for lp oa relax.
    else # Gurobi
        m = Model(() -> Gurobi.Optimizer(GUROBI_ENV[]))
        if mute
            set_optimizer_attribute(m, "OutputFlag", 0)
        end
        set_optimizer_attribute(m, "Threads",1)
        set_optimizer_attribute(m, "TimeLimit", obbt_time) # maximum runtime limit is 1 hours
        set_optimizer_attribute(m, "MIQCPMethod", -1) # 0 for qcp relax and 1 for lp oa relax, -1 for auto This is for Gurobi
        #set_optimizer_attribute(m, "NoRelHeurWork", Inf) # set NoRel for fast searching of good solution
    end
    @variable(m, lower[t,i] <= centers[t in 1:d, i in 1:k] <= upper[t,i], start=rand());
    @constraint(m, [j in 1:k-1], centers[1,j]<= centers[1,j+1])

    @variable(m, 0<=dmat[i in 1:k, j in 1:n]<=dmat_max[i,j], start=rand());
    #@variable(m, lower[t,i]^2 <= w[t in 1:d, i in 1:k] <= upper[t,i]^2) # add the horizontal line of the lower bottom line bound 
    @variable(m, 0 <= w[t in 1:d, i in 1:k], start=rand()) # add the horizontal line of the lower bottom line bound 
    @constraint(m, [i in 1:k, j in 1:n], dmat[i,j] >= sum((X[t,j]^2 - 2*X[t,j]*centers[t,i] + w[t,i]) for t in 1:d ));
    itval = (upper-lower)./2/nlines # total 2*nlines, separate the range into 2*nlines sections
    for line in 0:(nlines-1)
        lwr = lower+itval.*line
        upr = upper-itval.*line
        @constraint(m, [t in 1:d, i in 1:k], 2*lwr[t,i]*centers[t,i]-lwr[t,i]^2 <= w[t,i])
        @constraint(m, [t in 1:d, i in 1:k], 2*upr[t,i]*centers[t,i]-upr[t,i]^2 <= w[t,i])
    end
    # add constraint for upper bound of w, may not necessary
    # @constraint(m, [t in 1:d, i in 1:k], w[t,i] <= (upper[t,i]+lower[t,i])*centers[t,i]-upper[t,i]*lower[t,i])

    @variable(m, b[1:k, 1:n], Bin)
    @constraint(m, [j in 1:n], sum(b[i,j] for i in 1:k) == 1);
    @constraint(m, [j in 1:n], b[:,j] in MOI.SOS1(w_sos[:,j]))

    @variable(m, costs[1:n]>=0, start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, sum(costs[j] for j in 1:n)<= UB) # add the constraint that the total cost should lower than current UB

    if obbt == "min"
        @objective(m, Min, centers[dim, clst]);
        optimize!(m);
        if result_count(m) <= 0
            objv = lower[dim, clst]
        else
            objv = objective_bound(m) # objective_value(m) # using objective_bound so that the value is the smallest value
        end
    else # obbt == "max"
        @objective(m, Max, centers[dim, clst]);
        optimize!(m);
        if result_count(m) <= 0
            objv = upper[dim, clst]
        else
            objv = objective_bound(m) # objective_value(m) # using objective_bound so that the value is the smallest value
        end
    end
    return objv
end


# reduced bb subproblem solvers using cplex
function global_OPT3(X, k, lower=nothing, upper=nothing, mute=false)
    d, n = size(X)
    lower, upper = init_bound(X, d, k, lower, upper)
    dmat_max = max_dist(X, d, k, n, lower, upper)

    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPX_PARAM_SCRIND", 0) # this varible control the print of CPLEX solving process
    end
    set_optimizer_attribute(m, "CPX_PARAM_THREADS",1)
    set_optimizer_attribute(m, "CPX_PARAM_TILIM", time_lapse) # maximum runtime limit is 1 hours
    # here the gap should always < mingap of BB, e.g. if mingap = 0.1%, then gap here should be < 0.1%, the default is 0.01%
    set_optimizer_attribute(m, "CPX_PARAM_EPGAP", 0.05)
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
    objv = objective_bound(m)
    return centers, objv
end


# reduced bb subproblem solvers with largranian decomposition
# here the labmda is the largrange multiplier
function global_OPT3_LD(X, k, lambda, ctr_init, lg_cuts=nothing, w_sos=nothing, lower=nothing, upper=nothing, mute=false, solver="CPLEX")
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
        m = Model(() -> Gurobi.Optimizer(GUROBI_ENV[]))
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
    if w_sos !== nothing
        @constraint(m, [j in 1:n], b[:,j] in MOI.SOS1(w_sos[:,j]))
    end

    @variable(m, costs[1:n]>=0, start=rand());
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] >= -dmat_max[i,j]*(1-b[i,j]))
    @constraint(m, [i in 1:k, j in 1:n], costs[j] - dmat[i,j] <= dmat_max[i,j]*(1-b[i,j]))
    

    if lg_cuts !== nothing
        #print(lg_cuts)
        for l in 1:length(lg_cuts[1])
            @constraint(m, lg_cuts[1][l] <= sum(costs[j] for j in 1:n)+
                    sum((lg_cuts[3][l][:,i]-lg_cuts[2][l][:,i])'*centers[:,i] for i in 1:k))
        end
    end

    @objective(m, Min, sum(costs[j] for j in 1:n)+
                sum((lambda[:,i,2]-lambda[:,i,1])'*centers[:,i] for i in 1:k));
    optimize!(m);


    centers = value.(centers)
    objv = objective_bound(m)  #getobjectivevalue(m)
    #InteractiveUtils.varinfo()
    return centers, objv
end

end 

