#!/bin/bash
# This is the nested bash file for testing both the real and toy datasets
# $1: parallel mode if HPC then run on hpc server, else a number of procs on local, if 1, then it is run on serial
# $2: cluster number
# $3: largrangian cuts
# $4: solver: Gurobi or CPLEX
# $5: specify the running problem size: 
#   s for small problem d<7 and n <1,000
#   m for median problem 7<=d<=15 n<100,000 
#   l for large problem n >= 100,000
# $6: some useful distinguishable phrase
# run chmod +x ./test_niagara.sh first

# must load these module and precompile julia self-created module before sbatch
module load julia/1.5.3
module load mycplex/20.1.0
module use /scinet/niagara/software/commercial/modules
module load gurobi/9.0.2

julia src/pre_load.jl # preload and compile self-created modules

export CLUST_NUM=$2
#export PAR_MODE=$2
export LAR_CUT=$3
export SOLVER=$4
sbatch -n $1 --export=CLUST_NUM,LAR_CUT,SOLVER real-$5.sh
sbatch -n $1 --export=CLUST_NUM,LAR_CUT,SOLVER toy-$5.sh

sleep 3h

squeue -u khua1989 > joblist.out

while (( `wc -l < joblist.out `>1 ))
do
    echo `wc -l < joblist.out`
    sleep 2h
    squeue -u khua1989 > joblist.out
done

echo "Test problem with $2 clusters." > rlt-$1-$2-$3-$4-$5.out
echo -e "Datasets\tTime\tGap\tIter\n" >> rlt-$1-$2-$3-$4-$5.out

for file in `find . -type f -iname "real-$5*.out" `
do
    if [ -f "$file" ]
    then 
        tail -1 $file | head -n 1 >> rlt-$1-$2-$3-$4-$5.out
        #tail -1 $file >> result-$1-$2-$5.out
    fi
done

for file in `find . -type f -iname "toy-$5*.out" `
do
    if [ -f "$file" ]
    then 
        tail -1 $file | head -n 1 >> rlt-$1-$2-$3-$4-$5.out
        #tail -1 $file >> result-$1-$2-$5.out
    fi
done
