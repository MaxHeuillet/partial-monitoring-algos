#!/bin/bash


horizon=100000
nfolds=96
var=1

for context_type in 'quintic'

    do

    for game in  'AT' 

        do 

            for task in 'easy' 'difficult'

                do

                for alg in 'PGIDSratio'

                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg 'VAR' $var
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg,VAR=$var ./partial_monitoring/experiment.sh 
                    ((var++))
                    done
                
                done
        
        done

    done
