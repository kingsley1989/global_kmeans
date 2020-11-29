#! /bin/bash
#SBATCH -t 0-15:00
#SBATCH --mem=4G
module load StdEnv/2020 hdf5/1.10.6 julia/1.5.2
julia testing_hemi.jl
