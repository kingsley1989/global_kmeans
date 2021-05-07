# This is the file for precompiling self-created packages in niagara system

if !("src/" in LOAD_PATH)
    push!(LOAD_PATH, "src/")
end

using Nodes, data_process, grb_env, probing, branch, obbt
using bb_functions, lb_functions, ub_functions, opt_functions