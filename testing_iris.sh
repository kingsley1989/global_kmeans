#! /bin/bash
#SBATCH -t 0-20:00
#SBATCH --mem=10G
module load StdEnv/2020 hdf5/1.10.6 julia/1.5.2
julia -e "using Colors; using Rmath"
julia testing_iris.jl
