module grb_env

using Gurobi

export GUROBI_ENV

const GUROBI_ENV = Ref{Gurobi.Env}()

function __init__()
    global GUROBI_ENV[] = Gurobi.Env()
end

end