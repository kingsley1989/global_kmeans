#! /bin/bash
#SBATCH --nodes=10
#SBATCH -t 0-16:30
#SBATCH --array=0-3
#SBATCH --output=info-%x-%a.out

cd ${SLURM_SUBMIT_DIR}

module load NiaEnv/2019b
#module load ddt # load this module to prevent error of julia module loading error
module load julia/1.5.3
module load mycplex/20.1.0
module use /scinet/niagara/software/commercial/modules
module load gurobi/9.0.2

datasets=("spnet3D" "rng_agr" "rds_cnt" "rds")

julia -e "using Colors; using Rmath"
julia test/testing_real.jl ${SLURM_NTASKS} ${CLUST_NUM} ${datasets[${SLURM_ARRAY_TASK_ID}]} ${LAR_CUT} ${SOLVER} > real-l-${datasets[${SLURM_ARRAY_TASK_ID}]}.out

:<<EOF
    # input argument of julia
    # args[1] HPC or ntasks
    # args[2] number of Clusters
    # args[3] largrangian cuts: lar, nolar
    # args[4] solver: CPLEX, Gurobi
EOF

