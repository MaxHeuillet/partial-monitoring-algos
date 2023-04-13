#!/bin/bash

horizon=20000
nfolds=96

for game in 'LE' 'AT' 

    do 

        for task in 'imbalanced' 'balanced'

            do

                for alg in 'RandCBP_1' 'RandCBP_18' 'RandCBP_116' 'RandCBP_132' 

                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds  'GAME' $game 'TASK' $task 'ALG' $alg
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,GAME=$game,TASK=$task,ALG=$alg, ./partial_monitoring/benchmark_randcbp_cont.sh 
                    done
                
            done
        
    done