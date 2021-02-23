module bb_functions

using Clustering
using Printf
using JuMP
using Ipopt, CPLEX#, SCIP
using Random
using ub_functions, lb_functions

export Node, branch_bound


struct Node
    lower
    upper
    level::Int
    LB::Float64
    groups
end
Node() = Node(nothing, nothing, -1, -1e10, nothing)


maxiter = 5000
tol = 1e-6
mingap = 1e-3
time_lapse = 43200 # 4 hours


# function to print the node in a neat form
function printNodeList(nodeList)
    for i in 1:length(nodeList)
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:lower))) # reserve 3 decimal precision
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:upper)))
        println(getfield(nodeList[i],:level)) # integer
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:LB)))
    end
end


# function to record the finish time point
time_finish(seconds) = round(Int, 10^9 * seconds + time_ns())


function getGlobalLowerBound(nodeList) # if LB same, choose the first smallest one
    LB = 1e10
    nodeid = 1
    for (idx,n) in enumerate(nodeList)
    	#println("remaining ", idx,  "   ", n.LB)
        if n.LB < LB
            LB = n.LB
            nodeid = idx
        end
    end
    return LB, nodeid
end


function branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k)
    d, n = size(X);	 
    lower = copy(node.lower)
    upper = copy(node.upper)
    upper[bVarIdx, bVarIdy] = bValue # split from this variable at bValue
    for j = 2:k  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
    	if lower[1, j] <= lower[1, j-1]  
	        lower[1, j] = lower[1, j-1]
	    end
    end
    if sum(lower.<=upper)==d*k
    	left_node = Node(lower, upper, node.level+1, node_LB, node.groups)
	    push!(nodeList, left_node)
	    # println("left_node:   ", lower, "   ",upper)
    end

    lower = copy(node.lower)
    upper = copy(node.upper)
    lower[bVarIdx, bVarIdy] = bValue
    for j = 2:k
    	if lower[1, j] <= lower[1, j-1]
	        lower[1, j]	= lower[1, j-1]
	    end
    end
    if sum(lower.<=upper)==d*k
    	right_node = Node(lower, upper, node.level+1, node_LB, node.groups)
    	push!(nodeList, right_node)
	    # println("right_node:   ", lower,"   ",upper)
    end
end


function branch_bound(X, k)
    d, n = size(X);	 
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)  
    for i = 1:d
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
	    upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)
    # insert OBBT function here to tightening the range of each variable

    # generated the initial groups for subgrouping optimization
    # in this version of bb function, groups information is not needed
    result = kmeans(X, k)
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    groups = lb_functions.kmeans_group(X, result.assignments, ngroups)
    #println(groups)

    UB = 1e10;
    max_LB = 1e10; # used to save the best lower bound at the end (smallest but within the mingap)
    centers = nothing;
    root = Node(lower_data, upper_data, -1, -1e10, groups);
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
	    LB, nodeid = getGlobalLowerBound(nodeList)
        node = nodeList[nodeid]
        deleteat!(nodeList, nodeid)
        level = node.level
        @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        # println("lower\n", node.lower)
        # println("upper\n",node.upper)
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])

        ############# iteratively bound tightening #######################
        node_LB = node.LB
        delete_nodes = []

        if iter == 1
            node_centers, node_UB = ub_functions.getUpperBound(X, k, nothing, nothing, tol)
        else
            node_centers, node_UB = ub_functions.getUpperBound(X, k, node.lower, node.upper, tol)
        end    
        if (node_UB < UB)
            UB = node_UB
            centers = node_centers
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
        
        # The node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping
        # node_LB = lb_functions.getLowerBound_Test(X, k, centers, node.lower, node.upper) # getLowerBound_clust
        # node_LB = lb_functions.getLowerBound_analytic(X, k, node.lower, node.upper) # getLowerBound with closed-form expression
        node_LB = lb.functions.getLowerBound_linear(X, k, node.lower, node.upper, false, 5) # getLowerBound with linearized constraints 
        #if node_LB<LB # this if statement just put all nodes have the lb greater than their parent node
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
            #SelectVarMaxRange	centers = rand(d, k)
            bVarIdx = 1
            bVarIdy = 1
            maxrange = 1e-8
            # the following code choose the variable by the largest value range
            for i in 1:d
                for j in 1:k
                    range = (node.upper[i,j] -node.lower[i,j])
                    if range > maxrange
                        bVarIdx = i
                        bVarIdy = j
                        maxrange = range
                    end	
                end
            end
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
    
    return centers, UB, calcInfo
end


function branch_bound_LD(X, k)
    d, n = size(X);	 
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)  
    for i = 1:d
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
	    upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)

    # generated the initial groups for subgrouping optimization
    # in this version of bb function, groups information is not needed
    result = kmeans(X, k)
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    groups = lb_functions.kmeans_group(X, result.assignments, ngroups)
    #println(groups)

    UB = 1e10;
    max_LB = 1e10; # used to save the best lower bound at the end (smallest but within the mingap)
    centers = nothing;
    root = Node(lower_data, upper_data, -1, -1e10, groups);
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
	    LB, nodeid = getGlobalLowerBound(nodeList)
        node = nodeList[nodeid]
        deleteat!(nodeList, nodeid)
        level = node.level
        @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        # println("lower\n", node.lower)
        # println("upper\n",node.upper)
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])

        ############# iteratively bound tightening #######################
        node_LB = node.LB
        delete_nodes = []

        if iter == 1
            node_centers, node_UB = ub_functions.getUpperBound(X, k, nothing, nothing, tol)
        else
            node_centers, node_UB = ub_functions.getUpperBound(X, k, node.lower, node.upper, tol)
        end    
        if (node_UB < UB)
            UB = node_UB
            centers = node_centers
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
        
        # The node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping
        node_LB = lb_functions.getLowerBound_LD(X, k, centers, node.lower, node.upper) 
        #if node_LB<LB # this if statement just put all nodes have the lb greater than their parent node
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
            #SelectVarMaxRange	centers = rand(d, k)
            bVarIdx = 1
            bVarIdy = 1
            maxrange = 1e-8
            # the following code choose the variable by the largest value range
            for i in 1:d
                for j in 1:k
                    range = (node.upper[i,j] -node.lower[i,j])
                    if range > maxrange
                        bVarIdx = i
                        bVarIdy = j
                        maxrange = range
                    end	
                end
            end
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
    
    return centers, UB, calcInfo
end


function branch_bound_adptGp(X, k)
    d, n = size(X);	 
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)  
    for i = 1:d
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
	    upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)

    # generated the initial groups for subgrouping optimization
    result = kmeans(X, k)
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    groups = lb_functions.kmeans_group(X, result.assignments, ngroups)
    #println(groups)

    UB = 1e10;
    max_LB = 1e10; # used to save the best lower bound at the end (smallest but within the mingap)
    centers = nothing;
    root = Node(lower_data, upper_data, -1, -1e10, groups);
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
	    LB, nodeid = getGlobalLowerBound(nodeList)
        node = nodeList[nodeid]
        level = node.level
        groups = node.groups # the groups of this node is the grouping scheme of its parent node to get the LB
        deleteat!(nodeList, nodeid)

        @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        # println("lower\n", node.lower)
        # println("upper\n",node.upper)
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])

        ############# iteratively bound tightening #######################
        node_LB = node.LB # here node.LB is LB
        delete_nodes = []

        if iter == 1
            node_centers, node_UB = ub_functions.getUpperBound(X, k, nothing, nothing, tol)
        else
            node_centers, node_UB = ub_functions.getUpperBound(X, k, node.lower, node.upper, tol)
        end    
        if (node_UB < UB)
            UB = node_UB
            centers = node_centers
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
        
        # iThe node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping??
        # put global LB as the input of the function to check whether subgroupping get the better lb
        node_LB, groups = lb_functions.getLowerBound_adptGp(X, k, centers, groups, node.lower, node.upper, LB)
        #if node_LB<LB # this if statement just put all nodes have the lb greater than their parent node
        #    node_LB = LB
        #end 
        
        # update grouping scheme under current node. 
        # This node is gonna split and its grouping should be saved as the parent groups for its children
        # Therefore, every iteration, node will first have the grouping scheme of its parent node
        # after lower bound calculation, groups will be updated to its current grouping scheme that get the current LB
        node = Node(node.lower, node.upper, node.level, node.LB, groups)
        
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
            #SelectVarMaxRange	centers = rand(d, k)
            bVarIdx = 1
            bVarIdy = 1
            maxrange = 1e-8
            # the following code choose the variable by the largest value range
            for i in 1:d
                for j in 1:k
                    range = (node.upper[i,j] -node.lower[i,j])
                    if range > maxrange
                        bVarIdx = i
                        bVarIdy = j
                        maxrange = range
                    end	
                end
            end
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
    
    return centers, UB, calcInfo
end


function branch_bound_adptGp_LD(X, k)
    d, n = size(X);	 
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)  
    for i = 1:d
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
	    upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)

    # generated the initial groups for subgrouping optimization
    result = kmeans(X, k)
    ngroups = round(Int, n/k/10); # determine the number of groups, 10*k points in each group
    groups = lb_functions.kmeans_group(X, result.assignments, ngroups)
    #println(groups)

    UB = 1e10;
    max_LB = 1e10; # used to save the best lower bound at the end (smallest but within the mingap)
    centers = nothing;
    root = Node(lower_data, upper_data, -1, -1e10, groups);
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
	    LB, nodeid = getGlobalLowerBound(nodeList)
        node = nodeList[nodeid]
        level = node.level
        groups = node.groups # the groups of this node is the grouping scheme of its parent node to get the LB
        deleteat!(nodeList, nodeid)

        @printf "%-6d %-6d %-10d %-10.4f %-10.4f %-10.4f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
        # println("lower\n", node.lower)
        # println("upper\n",node.upper)
        # save calcuation information for result demostration
        push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])

        ############# iteratively bound tightening #######################
        node_LB = node.LB # here node.LB is LB
        delete_nodes = []

        if iter == 1
            node_centers, node_UB = ub_functions.getUpperBound(X, k, nothing, nothing, tol)
        else
            node_centers, node_UB = ub_functions.getUpperBound(X, k, node.lower, node.upper, tol)
        end    
        if (node_UB < UB)
            UB = node_UB
            centers = node_centers
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
        
        # The node may has lb value smaller than the global lb, it is not good but is possible if we have the subgroupping??
        # put global LB as the input of the function to check whether subgroupping get the better lb
        node_LB, groups = lb_functions.getLowerBound_adptGp_LD(X, k, centers, groups, node.lower, node.upper, LB)
        #if node_LB<LB # this if statement just put all nodes have the lb greater than their parent node
        #    node_LB = LB
        #end 
        
        # update grouping scheme under current node. 
        # This node is gonna split and its grouping should be saved as the parent groups for its children
        # Therefore, every iteration, node will first have the grouping scheme of its parent node
        # after lower bound calculation, groups will be updated to its current grouping scheme that get the current LB
        node = Node(node.lower, node.upper, node.level, node.LB, groups)
        
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
            #SelectVarMaxRange	centers = rand(d, k)
            bVarIdx = 1
            bVarIdy = 1
            maxrange = 1e-8
            # the following code choose the variable by the largest value range
            for i in 1:d
                for j in 1:k
                    range = (node.upper[i,j] -node.lower[i,j])
                    if range > maxrange
                        bVarIdx = i
                        bVarIdy = j
                        maxrange = range
                    end	
                end
            end
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
    
    return centers, UB, calcInfo
end


# end of the module
end 

