#! /bin/bash
#SBATCH -t 0-5:00
#SBATCH --mem=8G

module load StdEnv/2020 hdf5/1.10.6 julia/1.5.2
module load mycplex/12.10.0

julia -e "using Colors; using Rmath"
julia test/testing_iris.jl
