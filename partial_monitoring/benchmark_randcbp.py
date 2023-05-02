
import numpy as np

import multiprocessing as mp

from functools import partial
import pickle as pkl
import gzip
import os

import games

import randcbp

import gzip
import pickle as pkl

import subprocess

######################
######################

def evaluate_parallel(n_folds, horizon, alg, game, task, label):

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    pool = mp.Pool( processes = ncpus ) 
    ev = Evaluation(horizon, task)

    np.random.seed(1)
    distributions = []
    labels = []
    nfolds = []

    for _ in range(n_folds):
        if ev.task == 'imbalanced':
            p = np.random.uniform(0, 0.2)
        else:
            p = np.random.uniform(0.4, 0.5)
        distributions.append( [p, 1-p] )
        labels.append( label )
        nfolds.append(n_folds)
    return None
    # return np.asarray(  pool.map( partial( ev.eval_policy_once, alg, game ), zip(distributions, range(n_folds), labels, nfolds ) ) ) 

class Evaluation:

    def __init__(self, horizon,task):
        self.task = task
        self.horizon = horizon

    def get_outcomes(self, game):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        alg.reset()
        distribution, jobid, label, nfolds = job
        np.random.seed(jobid)

        outcome_distribution =  {'spam':distribution[0], 'ham':distribution[1]}

        game.set_outcome_distribution( outcome_distribution , jobid )

        action_counter = np.zeros( (game.n_actions, self.horizon) )

        # generate outcomes obliviously
        outcomes = self.get_outcomes(game )

        for t in range(self.horizon):
            #print(t)

            # Environment chooses one outcome and one context associated to this outcome
            outcome = outcomes[t]

            # policy chooses one action
            # print('t', t,  'outcome', outcome, 'context', context)
            action = alg.get_action(t)

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, None )

            for i in range(game.n_actions):
                if i == action:
                    action_counter[i][t] = action_counter[i][t-1] +1
                else:
                    action_counter[i][t] = action_counter[i][t-1]
            
            result = np.array( [ game.delta(i) for i in range(game.n_actions) ]).T @ action_counter
            
            with gzip.open( './partial_monitoring/results/benchmark_randcbp/{}/{}_{}_{}_{}_{}.pkl.gz'.format(game.name, self.task,  self.horizon, nfolds, label, jobid) ,'wb') as f:
                pkl.dump(result,f)

        return  True 


###################################
# Synthetic Experiments
###################################

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
parser.add_argument("--game", required=True, help="game")
parser.add_argument("--task", required=True, help="task") #balanced, imbalanced
parser.add_argument("--algo_name", required=True, help="algorithme")
args = parser.parse_args()

horizon = int(args.horizon)
n_folds = int(args.n_folds)

game = games.label_efficient(  ) if args.game == 'LE' else games.apple_tasting(False) 

algo_name = args.algo_name.split('_')

if algo_name[1] == '1':
    sigma = 1
elif algo_name[1] == '18':
    sigma = 1/8
elif algo_name[1] == '116':
    sigma = 1/16
elif algo_name[1] == '132':
    sigma = 1/32

if algo_name[2] == '5':
    K = 5
elif algo_name[2] == '10':
    K = 10
elif algo_name[2] == '20':
    K = 20
elif algo_name[2] == '100':
    K = 100

epsilon = 10e-7
alpha = 1.01

alg =  randcbp.RandCBP(  game, alpha, sigma, K, epsilon)  

result = evaluate_parallel(n_folds, horizon, alg, game, args.task , args.algo_name )

with gzip.open( './partial_monitoring/results/benchmark_randcbp/{}/{}_{}_{}_{}.pkl.gz'.format(game.name, '{}'.format(args.task) , horizon, n_folds,  args.algo_name) ,'wb') as g:

    for jobid in range(n_folds):

        with gzip.open(  './partial_monitoring/results/benchmark_randcbp/{}/{}_{}_{}_{}_{}.pkl.gz'.format(game.name, '{}'.format(args.task), horizon, n_folds,  args.algo_name, jobid) ,'rb') as f:
            r = pkl.load(f)

        pkl.dump( r, g)
                
        bashCommand = 'rm ./partial_monitoring/results/benchmark_randcbp/{}/{}_{}_{}_{}_{}.pkl.gz'.format(game.name, '{}'.format(args.task), horizon, n_folds,  args.algo_name, jobid)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()