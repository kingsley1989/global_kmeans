#! /bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=40
#SBATCH -t 0-06:30
#SBATCH --mem-per-cpu=3G
#SBATCH --array=0-1
#SBATCH --output=%x-%a.out

module load StdEnv/2020 hdf5/1.10.6 julia/1.5.2
module load mycplex/12.10.0

datasets=("glass" "iris") 
# "seed" "pr2392" "hemi")

julia -e "using Colors; using Rmath"
julia test/testing_${datasets[${SLURM_ARRAY_TASK_ID}]}.jl ${PAR_MODE} ${CLUST_NUM}

:<<EOF
    # input argument of julia
    # args[1] HPC or not 
    # args[2] number of Clusters
EOF

