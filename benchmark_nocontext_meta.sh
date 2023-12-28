#!/bin/bash

horizon=20000
nfolds=96

for game in  'LE' 'AT' 

    do 

        for task in 'all' 'balanced' 'imbalanced' 

            do
            for alg in 'RandCBP_12_5' 'RandCBP_1_5' 'RandCBP_2_5' 'RandCBP_12_10' 'RandCBP_10_5' 'RandCBP_1_10' 'RandCBP_2_10' 'RandCBP_10_10'  'RandCBP_12_20' 'RandCBP_1_20' 'RandCBP_2_20' 'RandCBP_10_20'  
                    
                do 
		        echo 'horizon' $horizon 'nfolds' $nfolds 'GAME' $game 'TASK' $task 'ALG' $alg 
    
                sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,GAME=$game,TASK=$task,ALG=$alg ./partial_monitoring/benchmark_nocontext.sh 

                done
                
            done
        
    done


#'RandCBP_1_5' 'RandCBP_18_5' 'RandCBP_116_5' 'RandCBP_132_5'  'RandCBP_1_10' 'RandCBP_18_10' 'RandCBP_116_10' 'RandCBP_132_10'  'RandCBP_1_20' 'RandCBP_18_20' 'RandCBP_116_20' 'RandCBP_132_20' 