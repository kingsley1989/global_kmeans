module Nodes

using Printf

export Node

struct Node
    lower
    upper
    level::Int
    LB::Float64
    groups
    lambda
end

Node() = Node(nothing, nothing, -1, -1e15, nothing,nothing)


# function to print the node in a neat form
function printNodeList(nodeList)
    for i in 1:length(nodeList)
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:lower))) # reserve 3 decimal precision
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:upper)))
        println(getfield(nodeList[i],:level)) # integer
        println(map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:LB)))
    end
end


end