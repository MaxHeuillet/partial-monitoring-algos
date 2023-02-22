#!/bin/bash

horizon=20000
nfolds=96

for game in 'LE' 'AT' 

    do 

        for task in 'imbalanced' 'balanced'

            do

                for alg in 'TSPM_0' 'TSPM_1' 

                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,GAME=$game,TASK=$task,ALG=$alg, ./partial_monitoring/benchmark_tspm.sh 
                    done
                
                done
        
        done

    done