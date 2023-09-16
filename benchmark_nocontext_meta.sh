#!/bin/bash

horizon=20000
nfolds=96

for game in  'AT' 'LE' 

    do 

        for task in 'balanced' 'imbalanced'

            do
            for alg in 'RandCBP_10_5' 'RandCBP_1_5' 'RandCBP_18_5' 'RandCBP_116_5' 'RandCBP_132_5'  'RandCBP_10_5' 'RandCBP_1_10' 'RandCBP_18_10' 'RandCBP_116_10' 'RandCBP_132_10'  'RandCBP_10_5' 'RandCBP_1_20' 'RandCBP_18_20' 'RandCBP_116_20' 'RandCBP_132_20'  'RandCBP_10_5' 'RandCBP_1_100' 'RandCBP_18_100' 'RandCBP_116_100' 'RandCBP_132_100'
                    
                do
		        echo 'horizon' $horizon 'nfolds' $nfolds 'GAME' $game 'TASK' $task 'ALG' $alg 
    
                sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,GAME=$game,TASK=$task,ALG=$alg ./partial_monitoring/benchmark_nocontext.sh 

                done
                
            done
        
    done


