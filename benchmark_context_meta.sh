#!/bin/bash

horizon=20000
nfolds=96
var=1

for context_type in 'linear' #'quintic'

    do

    for game in  'AT' 'LE'  

        do 

            for task in 'balanced' #'imbalanced'

                do
                # for alg in 'PGTS' 'PGIDSratio'
                for alg in 'CBPside' 'RandCBPside2_10_5_07' 'RandCBPside2_1_5_07' 'RandCBPside2_18_5_07' 'RandCBPside2_116_5_07' 'RandCBPside2_132_5_07' 'RandCBPside2_10_10_07' 'RandCBPside2_1_10_07' 'RandCBPside2_18_10_07' 'RandCBPside2_116_10_07' 'RandCBPside2_132_10_07' 'RandCBPside2_10_20_07' 'RandCBPside2_1_20_07' 'RandCBPside2_18_20_07' 'RandCBPside2_116_20_07' 'RandCBPside2_132_20_07' 'RandCBPside2_10_100_07' 'RandCBPside2_1_100_07' 'RandCBPside2_18_100_07' 'RandCBPside2_116_100_07' 'RandCBPside2_132_100_07'        
                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg 'VAR' $var
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg,VAR=$var ./partial_monitoring/benchmark_context.sh 
                    # python3 ./partial_monitoring/benchmark_context.py --horizon $horizon --n_folds $nfolds --game $game --alg $alg --task $task --context_type $context_type
                    ((var++))
                    done
                
                done
        
        done

    done

