#!/bin/bash


horizon=100000
nfolds=2

for context_type in 'linear' #'quintic'

    do

    for game in 'LE' 'AT' 

        do 

            for task in 'easy' 'difficult'

                do

                for alg in 'random' 'CBPside005' 'CBPside0001' 'RandCBPside005' 'RandCBPside0001'

                    do

		    echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg

                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg ./partial_monitoring/experiment.sh 
                    ((var++))
                    done
                
                done
        
        done

    done
