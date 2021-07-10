module bb_functions

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX, Gurobi#, SCIP
using Random
using Statistics

using Distributed, SharedArrays
@everywhere using ParallelDataTransfer

using ub_functions, lb_functions, opt_functions, branch, Nodes

export branch_bound

maxiter = 5000000
tol = 1e-6
mingap = 1e-3
time_lapse = 4*3600 # 4 hours


# function to record the finish time point
time_finish(seconds) = round(Int, 10^9 * seconds + time_ns())


# during iteration: 
# LB = node.LB is the best LB among all node and is in current iteration
# UB is the best UB, and node_UB the updated UB(it is possible: node_UB > UB) of current iteration (after run getUpperBound)
# node_LB is the updated LB of current itattion (after run probing or getLowerBound_adptGp_LD)
function branch_bound(X, k, method = "CF", mode = "fixed", solver = "CPLEX")
    
    x_max = maximum(X) # max value after transfer to non-zero value
    tnsf_max = false
    if x_max >= 20
        tnsf_max = true
        X = X/(x_max*0.05)
    end
    
    d, n = size(X);
    lower, upper = opt_functions.init_bound(X, d, k)

    UB = 1e15;
    max_LB = 1e15; # used to save the best lower bound at the end (smallest but within the mingap)
    centers = nothing;
    assign = nothing;
    w_sos = nothing;
    
    # groups is not initalized, will generate at the first iteration after the calculation of upper bound
    root = Node(lower, upper, -1, -1e15, nothing, nothing, nothing); 
    nodeList =[]
    push!(nodeList, root)
    iter = 0
    println(" iter ", " left ", " lev  ", "       LB       ", "       UB      ", "      gap   ")
    
    # get program end time point
    end_time = time_finish(time_lapse) # the branch and bound process ends after 6 hours

    #####inside main loop##################################
    calcInfo = [] # initial space to save calcuation information
    while nodeList!=[]

        # printNodeList(nodeList) # the printed LB of each node is the LB for its parent node
        # we start at the branch(node) with lowest Lower bound
	    LB, nodeid = getGlobalLowerBound(nodeList) # Here the LB is the best LB and also node.LB of current iteration
        node = nodeList[nodeid]
        #level = node.level
        groups = node.groups # the groups of this node is the grouping scheme of its parent node to get the LB
        deleteat!(nodeList, nodeid)
        # so currently, the global lower bound corresponding to node, LB = node.LB, groups = node.groups
        @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        # println("lower\n", node.lower)
        # println("upper\n",node.upper)
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])
        
        # time stamp should be checked after the retrival of the results
        if (iter == maxiter) || (time_ns() >= end_time)
            break
        end
        iter += 1

        ############# iteratively bound tightening #######################
        # node_LB = node.LB # seems useless
        delete_nodes = []

        if iter == 1 # first iteration, upper bound calculation is in root node
            node_centers, node_assign, node_UB = ub_functions.getUpperBound(X, k, nothing, nothing, tol)
            centers = node_centers
            assign = node_assign
            # generate sos1 weight
            max_d = opt_functions.max_dist(X, d, k, n, lower, upper) # k*n
            w_sos = reduce(hcat, [max_d[:,j] .- [sum((X[t,j] - centers[t,i])^2 for t in 1:d) for i in 1:k] for j in 1:n]) # k*n

            # determine the number of groups, 162/d-k points in each group, guarantee that 162 variables in a subproblem
            # however, the number of points in a group can not be too large,
            # otherwise consume too much memory (out of memory) and calculation is too slow
            if 162/d-k > 31 # set 31 so that d > 5 will not be affected
                if n > 200
                    ngroups = round(Int, n/20) # for pr2392 and toy-42000 use 20(k=3) and 15(k=4) as the divisor otherwise 25
                else
                    ngroups = round(Int, n/31)
                end
            else
                ngroups = round(Int, n/(162/d-k)) # round(Int, n/15); #
            end
            println(ngroups)
            # in root node, we can first genarate a grouping scheme based on best kmeans result
            # generated the initial groups for subgrouping optimization
            # in version of no-adaGP-bb function, groups information is not needed
            groups = lb_functions.kmeans_group(X, assign, ngroups)
        
            # update node with groups
            node = Node(lower, upper, node.level, node.LB, groups, node.lambda, nothing);
        else
            if (mode == "fixed") # if fixed, UB is the result of UB1
                t_ctr = mean(node.group_centers, dims=3)[:,:,1]
                t_UB, ~ = obj_assign(t_ctr, X)
                if (t_UB < UB)
                    node_UB = t_UB	      
                else
                    node_UB = UB	  
                end    
            else # else adaptive grouping, use the center, assign of the local solver (mode=="ada")
                node_centers, node_assign, node_UB = ub_functions.getUpperBound(X, k, node.lower, node.upper, tol)
                if ((node_UB-UB)/UB < 0.5) # using the solution from local solver if the objective is not too large (2*UB)
                    centers = node_centers # use the new center to generate group
                    assign = node_assign
                    # can not update groups in node, since we need prarent groups for adaptve calculation
                end
            end
        end    
        if (node_UB < UB)
            UB = node_UB
            # the following code delete branch with lb close to the global upper bound
            delete_nodes = []
            for (idx,n) in enumerate(nodeList)
                if (((UB-n.LB)<= mingap) || ((UB-n.LB) <=mingap*min(abs(UB), abs(n.LB))))
                    push!(delete_nodes, idx)
                end
            end
            deleteat!(nodeList, sort(delete_nodes))
		    #println("UB:  ", UB)
        end

        # println("LB:  ", LB)
        node_LB = LB

        if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            println("analytic LB  ",node_LB, "   >=UB    ", UB)
        else
            # The node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping
            if (method == "SG")
                node_LB, groups, ~, group_centers = lb_functions.getLowerBound_adptGp_LD(X, k, w_sos, assign, node, UB, mode, solver, false)
                # for mode != fixed, update grouping scheme under current node. 
                # This node is gonna split and its grouping should be saved as the parent groups for its children
                # Therefore, every iteration, node will first have the grouping scheme of its parent node
                # after lower bound calculation, groups will be updated to its current grouping scheme that get the current LB
                node = Node(node.lower, node.upper, node.level, node.LB, groups, nothing, group_centers)
            elseif (method == "LD+SG")
                node_LB, groups, lambda, group_centers = lb_functions.getLowerBound_adptGp_LD(X, k, w_sos, assign, node, UB, mode, solver)
                node = Node(node.lower, node.upper, node.level, node.LB, groups, lambda, group_centers)
            else # closedForm
                node_LB, group_centers = lb_functions.getLowerBound_analytic(X, k, node.lower, node.upper) # getLowerBound with closed-form expression
                # node_LB = lb_functions.getLowerBound_linear(X, k, node.lower, node.upper, 5) # getLowerBound with linearized constraints 
                node = Node(node.lower, node.upper, node.level, node.LB, node.groups, node.lambda, group_centers) 
            end
        end

        # here this condition include the condition UB < node_LB and the condition that current node's LB is close to UB within the mingap
        # Such node no need to branch
        if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            # save the best LB if it close to UB enough (within the mingap)
            if node_LB < max_LB
                max_LB = node_LB
            end
            # continue   
	    else
            bVarIdx, bVarIdy = branch.SelectVardMaxLBCenterRange(group_centers)
            println("branching on ", bVarIdx,"    ", bVarIdy )
            # the split value is chosen by the midpoint
            bValue = (node.upper[bVarIdx,bVarIdy] + node.lower[bVarIdx,bVarIdy])/2;
            branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k);
        end
    end
    if nodeList==[]
        println("all node solved")
        # save final calcuation information
        push!(calcInfo, [iter, length(nodeList), max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
    else
        max_LB = calcInfo[end][4]
    end
    println("solved nodes:  ",iter)
    @printf "%-52d  %-14.4e %-14.4e %-7.4f %s \n" iter  max_LB UB (UB-max_LB)/min(abs(max_LB),abs(UB))*100 "%"
    println("centers   ",centers)
    
    # transfer back to original value of optimal value
    if tnsf_max
        UB = UB .* (x_max*0.05)^2
    end
    return centers, UB, calcInfo
end


# end of the module
end 

