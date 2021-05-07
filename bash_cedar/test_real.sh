#! /bin/bash
#SBATCH --nodes=1
#SBATCH -t 0-06:30
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-4
#SBATCH --output=%x-%a.out

cd ~/scratch/

module load StdEnv/2020 hdf5/1.10.6 julia/1.6.0
module load mycplex/12.10.0
module load gurobi/9.1.0

datasets=("glass" "iris" "seed" "pr2392" "hemi")

julia -e "using Colors; using Rmath"
julia test/testing_${datasets[${SLURM_ARRAY_TASK_ID}]}.jl ${SLURM_NTASKS} ${CLUST_NUM} ${LAR_CUT} ${SOLVER}

:<<EOF
    # input argument of julia
    # args[1] HPC or ntasks
    # args[2] number of Clusters
    # args[3] largrangian cuts: lar, nolar
    # args[4] solver: CPLEX, Gurobi
EOF

