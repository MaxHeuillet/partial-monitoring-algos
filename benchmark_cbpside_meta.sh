#!/bin/bash

horizon=20000
nfolds=96
var=1
alg = 'CBPside005'

for context_type in 'quintic' 'linear'

    do

    for game in 'LE' 'AT' 

        do 

            for task in 'imbalanced' 'balanced'

                do
    
		        echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg 'VAR' $var
    
                sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg,VAR=$var ./partial_monitoring/benchmark_cbpside.sh 
                ((var++))
                
                done
        
        done

    done
