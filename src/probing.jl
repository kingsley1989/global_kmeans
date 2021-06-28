module probing

using lb_functions

export probing_base


function probing_base(X, k, centers, lower_o, upper_o, UB, mingap)
    lower = copy(lower_o)
    upper = copy(upper_o)	 
    d, n = size(X)

    fathom = false
    node_LB, ~ =lb_functions.getLowerBound_analytic(X, k, lower, upper)
    if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            println("analytic LB  ",node_LB, "   >=UB    ", UB)
	    fathom = true 	    
    end
    for trial = 1:2
    	if fathom == true
	        break
    	end
        for dim in 1:d
            if fathom == true
                break
            end
            for clst in 1:k
                step = (upper[dim, clst] - lower[dim, clst])/6
                for insi = 1:5
                    if (upper[dim, clst] - lower[dim, clst]) <= (step+1e-6)
                        break
                    end
                    lower_trial = copy(lower)
                    upper_trial	= copy(upper)
                    upper_trial[dim, clst] = lower[dim, clst] + step
                    if dim == 1
                        for j = 1:(k-1)  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                            if upper_trial[1, k-j] >= upper_trial[1, k-j+1]
                                upper_trial[1, k-j] = upper_trial[1, k-j+1]
                            end
                        end
                    end
                    
                    LB_trial, ~ =lb_functions.getLowerBound_analytic(X, k, lower_trial, upper_trial)
                    if (UB-LB_trial)<= mingap || (UB-LB_trial) <= mingap*abs(UB)
                        println(trial, "  lower[ ",dim, ",",  clst,"]  from", lower[dim, clst], "  to ", upper_trial[dim, clst])
                        lower[dim, clst] = upper_trial[dim, clst] 
                        if dim == 1
                            for j = 2:k  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                                if lower[1, j] <= lower[1, j-1]
                                    lower[1, j] = lower[1, j-1]
                                end
                            end
                        end 
                    end

                    if (upper[dim, clst] - lower[dim, clst]) <=	(step+1e-6)
                        break
                    end
                    lower_trial = copy(lower)
                    upper_trial = copy(upper)
                    lower_trial[dim, clst] = upper[dim, clst] - step
                    if dim == 1
                        for j = 2:k  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                            if lower_trial[1, j] <= lower_trial[1, j-1]
                                lower_trial[1, j] = lower_trial[1, j-1]
                            end
                        end
                    end
                    LB_trial, ~ = lb_functions.getLowerBound_analytic(X, k, lower_trial, upper_trial)

                    if (UB-LB_trial)<= mingap || (UB-LB_trial) <= mingap*abs(UB)
                        println(trial, "  upper[ ",dim, ",",  clst,"]  from", upper[dim, clst], "  to ",	lower_trial[dim, clst])
                        upper[dim, clst] = lower_trial[dim, clst]
                        if dim == 1
                            for j = 1:(k-1)  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                                if upper[1, k-j] >= upper[1, k-j+1]
                                    upper[1, k-j] = upper[1, k-j+1]
                                end
                            end
                        end
                    end
                end

                if (upper[dim, clst] - lower[dim, clst]) <= (step+1e-6)
                    node_LB, ~ =lb_functions.getLowerBound_analytic(X, k, lower, upper)
                    if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*abs(UB)
                        println("analytic LB  ",node_LB, "   >=UB    ", UB)
                        fathom = true
                        break
                    end
                end    
            end
        end
    end
    return lower, upper, node_LB
end


end