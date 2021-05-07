#! /bin/bash
#SBATCH --nodes=1
#SBATCH -t 0-06:30
#SBATCH --array=0-10
#SBATCH --output=info-%x-%a.out

cd ${SLURM_SUBMIT_DIR}

module load NiaEnv/2019b
#module load ddt # load this module to prevent error of julia module loading error
module load julia/1.5.3
module load mycplex/20.1.0
module use /scinet/niagara/software/commercial/modules
module load gurobi/9.0.2

datasets=("body" "cncret" "glass" "gr202" "gr666" "hemi" "iris" "pr2392" "seeds" "u1060" "vowel")

julia -e "using Colors; using Rmath"
julia test/testing_real.jl ${SLURM_NTASKS} ${CLUST_NUM} ${datasets[${SLURM_ARRAY_TASK_ID}]} ${LAR_CUT} ${SOLVER} > real-s-${datasets[${SLURM_ARRAY_TASK_ID}]}.out

:<<EOF
    # input argument of julia
    # args[1] HPC or ntasks
    # args[2] number of Clusters
    # args[3] largrangian cuts: lar, nolar
    # args[4] solver: CPLEX, Gurobi
EOF

