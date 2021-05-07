#!/bin/bash
# This is the nested bash file for testing both the real and toy datasets
# $1: cluster number
# $2: parallel mode if HPC then run on hpc server, else a number of procs on local, if 1, then it is run on serial
# $3: largrangian cuts
# $4: solver: Gurobi or CPLEX
# $5: some useful distinguishable phrase
# run chmod +x ./test_cedar.sh first


export CLUST_NUM=$1
#export PAR_MODE=$2
export LAR_CUT=$3
export SOLVER=$4
sbatch -n $2 test_real.sh --export=CLUST_NUM,LAR_CUT,SOLVER 
sbatch -n $2 test_toy.sh --export=CLUST_NUM,LAR_CUT,SOLVER


sleep 3h

squeue -u khua1989 > joblist.out

while (( `wc -l < joblist.out `>1 ))
do
    echo `wc -l < joblist.out`
    sleep 2h
    squeue -u khua1989 > joblist.out
done

echo "Test problem with $1 clusters\n" > result-$1-$2-$5.out
echo -e "Datasets\tTime\tGap\tIter\n" >> result-$1-$2-$5.out

for file in `find . -type f -iname "test_*.out" `
do
    if [ -f "$file" ]
    then 
        tail -1 $file >> result-$1-$2-$5.out
    fi
done

