#! /bin/bash
#SBATCH --nodes=1
#SBATCH -t 0-06:30
#SBATCH --mem-per-cpu=4G
#SBATCH --array=5,7
#SBATCH --output=%x-%a.out

cd ~/scratch/

module load StdEnv/2020 hdf5/1.10.6 julia/1.6.0
module load mycplex/12.10.0
module load gurobi/9.1.0

#,10,50,100

julia -e "using Colors; using Rmath"
julia test/testing_toy.jl ${SLURM_NTASKS} ${CLUST_NUM} $((${SLURM_ARRAY_TASK_ID}*100)) ${LAR_CUT} ${SOLVER}

:<<EOF
    # input argument of julia
    # args[1] HPC or ntasks
    # args[2] number of Clusters
    # args[3] number of points in a cluster
    # args[4] largrangian cuts: lar, nolar
    # args[5] solver: CPLEX, Gurobi
EOF

