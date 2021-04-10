module branch

using Nodes

export branch!



function SelectVarMaxRange(node)
    dif = node.upper -node.lower
    ind = findmax(dif)[2]
    return ind[1], ind[2]
end

function SelectVardMaxLBCenterRange(group_centers)
    d, k, ngroups = size(group_centers)
    dif = zeros(d, k)
    for dim in 1:d
        for clst in 1:k
            dif[dim, clst] = maximum(group_centers[dim, clst,:]) - minimum(group_centers[dim, clst,:])
        end
    end
    #println("LBCenterRange:   ",dif)
    ind = findmax(dif)[2]
    return ind[1], ind[2]
end


function branch!(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k)
    d, n = size(X);
    lower = copy(node.lower)
    upper = copy(node.upper)
    upper[bVarIdx, bVarIdy] = bValue # split from this variable at bValue
    for j = 1:(k-1)  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
    	if upper[1, k-j] >= upper[1, k-j+1]  
	        upper[1, k-j] = upper[1, k-j+1]
	    end
    end
    if sum(lower.<=upper)==d*k
    	left_node = Node(lower, upper, node.level+1, node_LB, node.groups, node.lambda)
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
    	right_node = Node(lower, upper, node.level+1, node_LB, node.groups, node.lambda)
    	push!(nodeList, right_node)
	    # println("right_node:   ", lower,"   ",upper)
    end
end

end