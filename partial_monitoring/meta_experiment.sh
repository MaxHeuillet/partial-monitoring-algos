#!/bin/bash


horizon  = 50
nfolds = 1

for context_type in 'linear' #'quintic'

    do

    for game in 'LE' 'AT' 

        do 

            for task in 'easy' 'difficult'

                do

                for alg in 'random' 'CBPside005' 'CBPside0001' 'RandCBPside005' 'RandCBPside0001'

                    do
    
                    sbatch ./partial_monitoring/experiment.sh --export=HORIZON=horizon,NFOLDS=nfolds,CONTEXT_TYPE=context_type,GAME=game,TASK=task,ALG=alg

                    done
                
                done
        
        done

    done