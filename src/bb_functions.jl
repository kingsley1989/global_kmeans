
module bb_functions

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX, Gurobi#, SCIP
using Random
using Statistics

using ub_functions, lb_functions, opt_functions, obbt, branch, probing, Nodes

export branch_bound

maxiter = 5000000
tol = 1e-6
mingap = 1e-3
time_lapse = 12*3600 # 4 hours


# function to record the finish time point
time_finish(seconds) = round(Int, 10^9 * seconds + time_ns())


# during iteration: 
# LB = node.LB is the best LB among all node and is in current iteration
# UB is the best UB, and node_UB the updated UB(it is possible: node_UB > UB) of current iteration (after run getUpperBound)
# node_LB is the updated LB of current itattion (after run probing or getLowerBound_adptGp_LD)
function branch_bound(X, k, method = "SCEN", mode = "fixed", cons = "SOS1", cuts_info = "lar-5", solver = "CPLEX")
    
    #println(X[:,1:20])
    #=	 
    med_x = median(X)
    if log(10, med_x) > 4
        X = X/10^(floor(log(10, med_x))-3) 
    end=#
    if minimum(X) < 0
        X = X.+ minimum(X)
    end
    if maximum(X) >= 20
        X = X/(maximum(X)*0.05)
    end
    
    d, n = size(X);
    #println(d, n)
    lower, upper = opt_functions.init_bound(X, d, k)

    UB = 1e15;
    max_LB = 1e15; # used to save the best lower bound at the end (smallest but within the mingap)
    centers = nothing;
    assign = nothing;
    w_sos = nothing;
    if length(cuts_info) == 5 && cuts_info[1:4] == "lar-"
        cuts, n_cuts = split(cuts_info, "-")
        n_cuts = parse(Int, n_cuts)
    else
        cuts = "nolar"
        n_cuts = 1
    end
    # groups is not initalized, will generate at the first iteration after the calculation of upper bound
    root = Node(lower, upper, -1, -1e15, nothing, nothing, nothing, nothing); 
    nodeList =[]
    push!(nodeList, root)
    iter = 0
    println(" iter ", " left ", " lev  ", "       LB       ", "       UB      ", "      gap   ")
    
    # get program end time point
    end_time = time_finish(time_lapse) # the branch and bound process ends after 6 hours

    #####inside main loop##################################
    calcInfo = [] # initial space to save calcuation information
    while nodeList!=[]
        if (iter == maxiter) || (time_ns() >= end_time)
           break
        end
        iter += 1

        # printNodeList(nodeList) # the printed LB of each node is the LB for its parent node
        # we start at the branch(node) with lowest Lower bound
	    LB, nodeid = getGlobalLowerBound(nodeList) # Here the LB is the best LB and also node.LB of current iteration
        node = nodeList[nodeid]
        level = node.level
        groups = node.groups # the groups of this node is the grouping scheme of its parent node to get the LB
        deleteat!(nodeList, nodeid)
        # so currently, the global lower bound corresponding to node, LB = node.LB, groups = node.groups
        @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        # println("lower\n", node.lower)
        # println("upper\n",node.upper)
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])

        ############# iteratively bound tightening #######################
        # node_LB = node.LB # seems useless
        delete_nodes = []

        if iter == 1 # first iteration, upper bound calculation is in root node
            node_centers, node_assign, node_UB = ub_functions.getUpperBound(X, k, nothing, nothing, tol)
            centers = node_centers
            assign = node_assign
            if cons == "SOS1"
                max_d = opt_functions.max_dist(X, d, k, n, lower, upper) # k*n
                w_sos = reduce(hcat, [max_d[:,j] .- [sum((X[t,j] - centers[t,i])^2 for t in 1:d) for i in 1:k] for j in 1:n]) # k*n
            else
                w_sos = nothing
            end
            # in root node, we can first genarate a grouping scheme based on best kmeans result
            # generated the initial groups for subgrouping optimization
            # in version of no-adaGP-bb function, groups information is not needed
            # determine the number of groups, 162/d-k points in each group, guarantee that 162 variables in a subproblem
            # however, the number of points in a group can not be too large,
            # otherwise consume too much memory (out of memory) and calculation is too slow
            if 162/d-k > 31 # set 31 so that d > 5 will not be affected
                if n > 200
                    ngroups = round(Int, n/25)
                else
                    ngroups = round(Int, n/31)
                end
            else
                ngroups = round(Int, n/(162/d-k)) # round(Int, n/15); #
            end
            println(ngroups)
            groups = lb_functions.kmeans_group(X, assign, ngroups)
            # assign maybe can tansitive so that donot need to call obj_assign every time.
            #println(groups)

            # insert OBBT function here to tightening the range of each variable
            lwr = OBBT_min(X, k, node_UB, w_sos, nothing, nothing, true, 2, solver) # lower # 
            upr = OBBT_max(X, k, node_UB, w_sos, nothing, nothing, true, 2, solver) # upper # 
            # update node with lower, upper, groups
            node = Node(lwr, upr, node.level, node.LB, groups, node.lambda, nothing, nothing);
            
            if cuts == "lar"
                # run first lower bound to get the initial group LB
                node_LB, groups, lambda, group_centers, group_cuts= lb_functions.getLowerBound_adptGp_LD(X, k, w_sos, assign, node, UB, mode, solver, n_cuts)
                # here array of -Inf may produce error for opt solver, 0 is ok in the case of clustering
                #group_LB = repeat([-1e15], 1, ngroups)
                println(group_cuts)
                # update node with info from getLB function. groups is not changed here for fixed mode
                node = Node(lwr, upr, node.level, node_LB, groups, node.lambda, group_centers, group_cuts);
            end
        else
            if (mode == "fixed") # if fixed, UB is the result of kmeans
	            node_UB = UB	      
            else # else adaptive grouping, use the center, assign of the local solver (mode=="adaLocal")
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


    	lwr, upr, node_LB = probing_base(X, k, centers, node.lower, node.upper, UB, mingap);
        node = Node(lwr, upr, node.level, node.LB, node.groups, node.lambda, node.group_centers, node.group_cuts);
        println("node.lower after prob:  ", node.lower)
        println("node.upper after prob:  ", node.upper)


        if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            println("analytic LB  ",node_LB, "   >=UB    ", UB)
        else
            # The node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping
            if (method == "Test")
                node_LB = lb_functions.getLowerBound_Test(X, k, centers, node.lower, node.upper) # getLowerBound_clust
            elseif (method == "LD")
                node_LB = lb_functions.getLowerBound_LD(X, k, centers, node.lower, node.upper) 
            elseif (method == "adaGp")
                # iThe node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping??
                # put global LB as the input of the function to check whether subgroupping get the better lb
                node_LB, groups = lb_functions.getLowerBound_adptGp(X, k, centers, groups, node.lower, node.upper, LB)
                # update grouping scheme under current node. 
                # This node is gonna split and its grouping should be saved as the parent groups for its children
                # Therefore, every iteration, node will first have the grouping scheme of its parent node
                # after lower bound calculation, groups will be updated to its current grouping scheme that get the current LB
                node = Node(node.lower, node.upper, node.level, node.LB, groups, nothing, nothing)
            elseif (method == "LD+adaGp")
                node_LB, groups, lambda, group_centers, group_cuts= lb_functions.getLowerBound_adptGp_LD(X, k, w_sos, assign, node, UB, mode, solver, n_cuts)
                if cuts == "lar"
                    #println(group_cuts)
                    #println(node.group_cuts)
                    #group_cuts[1] = max.(group_cuts[1], node.group_cuts[1]) # get the best lb cut for each subproblem
                    node = Node(node.lower, node.upper, node.level, node.LB, groups, lambda, group_centers, group_cuts)
                else
                    node = Node(node.lower, node.upper, node.level, node.LB, groups, lambda, group_centers, nothing)
                end
            elseif (method == "SCEN")
                node_LB = lb_functions.getLowerBound_analytic(X, k, node.lower, node.upper) # getLowerBound with closed-form expression
                # node_LB = lb_functions.getLowerBound_linear(X, k, node.lower, node.upper, 5) # getLowerBound with linearized constraints 
            else    
                node_LB = lb_functions.getLowerBound_oa(X, k, UB, node.lower, node.upper, 2) # getLowerBound with outer approximation 
            end
        end

        #if node_LB<LB # LB is node.LB this if statement just put all nodes have the lb greater than their parent node
        #    node_LB = LB
        #end 
        # println("nodeLB nodeUB   ",node_LB, "     ", node_UB) 
        # println("centers  ", centers)
        # here this condition include the condition UB < node_LB and the condition that current node's LB is close to UB within the mingap
        # Such node no need to branch
        if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            # save the best LB if it close to UB enough (within the mingap)
            if node_LB < max_LB
                max_LB = node_LB
            end
            # continue   
	    else
            #bVarIdx, bVarIdy = branch.SelectVarMaxRange(node)
            bVarIdx, bVarIdy = branch.SelectVardMaxLBCenterRange(group_centers)
            println("branching on ", bVarIdx,"    ", bVarIdy )
            # the split value is chosen by the midpoint
            bValue = (node.upper[bVarIdx,bVarIdy] + node.lower[bVarIdx,bVarIdy])/2;
            branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k);
        end
        #println("After split:")
        #@printf "%-6d %-6d %-10d %-10.4f %-10.4e %-10.4f %s \n" iter length(nodeList) node.level node_LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
    end
    if nodeList==[]
        println("all node solved")
        #=if UB > 0
            LB = min(UB-mingap, UB/(1+mingap)) 
        else
            LB = min(UB-mingap, UB*(1+mingap))
        end=#
        # save final calcuation information
        push!(calcInfo, [iter, length(nodeList), max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
    else
        max_LB = calcInfo[end][4]
    end
    println("solved nodes:  ",iter)
    @printf "%-52d  %-14.4e %-14.4e %-7.4f %s \n" iter  max_LB UB (UB-max_LB)/min(abs(max_LB),abs(UB))*100 "%"
    println("centers   ",centers)
    
    #println("obbt lower: ", lwr)
    #println("obbt upper: ", upr)
    return centers, UB, calcInfo
end


# end of the module
end 

