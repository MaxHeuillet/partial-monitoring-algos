#!/bin/bash

horizon=20000
nfolds=96
var=1

for context_type in 'linear' 'quintic' 

    do

    for game in 'LE' 'AT' 

        do 

            for task in 'imbalanced' 'balanced'

                do

                for alg in 'RandCBPside2005_1_5_07' 'RandCBPside2005_18_5_07' 'RandCBPside2005_116_5_07' 'RandCBPside2005_132_5_07' 'RandCBPside2005_1_10_07' 'RandCBPside2005_18_10_07' 'RandCBPside2005_116_10_07' 'RandCBPside2005_132_10_07' 'RandCBPside2005_1_20_07' 'RandCBPside2005_18_20_07' 'RandCBPside2005_116_20_07' 'RandCBPside2005_132_20_07' 'RandCBPside2005_1_100_07' 'RandCBPside2005_18_100_07' 'RandCBPside2005_116_100_07' 'RandCBPside2005_132_100_07'   

                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg 'VAR' $var
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg,VAR=$var ./partial_monitoring/benchmark_randcbpside2.sh 
                    ((var++))
                    done
                
                done
        
        done

    done