#! /bin/bash
#SBATCH --nodes=1
#SBATCH -t 0-16:30
#SBATCH --array=20,50,100 
#SBATCH --output=info-%x-%a.out

cd ${SLURM_SUBMIT_DIR}

module load NiaEnv/2019b
#module load ddt # load this module to prevent error of julia module loading error
module load julia/1.5.3
module load mycplex/20.1.0
module use /scinet/niagara/software/commercial/modules
module load gurobi/9.0.2

#,10,50,100

julia -e "using Colors; using Rmath"
julia test/testing_toy.jl ${SLURM_NTASKS} ${CLUST_NUM} $((${SLURM_ARRAY_TASK_ID}*100)) ${LAR_CUT} ${SOLVER} > toy-s-${CLUST_NUM}-$((${SLURM_ARRAY_TASK_ID}*100)).out

:<<EOF
    # input argument of julia
    # args[1] HPC or ntasks
    # args[2] number of Clusters
    # args[3] number of points in a cluster
    # args[4] largrangian cuts: lar, nolar
    # args[5] solver: CPLEX, Gurobi
EOF

