#!/bin/bash


horizon=100000
nfolds=96

for context_type in 'linear' 'quintic'

    do

    for game in  'AT' 

        do 

            for task in  'easy' 'difficult'

                do

                for alg in 'PGIDSratio'

                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg 
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg ./partial_monitoring/experiment.sh 
                    done
                
                done
        
        done

    done
