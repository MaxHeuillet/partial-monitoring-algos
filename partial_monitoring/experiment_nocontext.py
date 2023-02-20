
import numpy as np
import os

import multiprocessing as mp

from functools import partial
import pickle as pkl
import gzip
# import plotly.graph_objects as go
import os


import games

import randcbp
import cpb_side_gaussian

import gzip
import pickle as pkl

######################
######################

def evaluate_parallel(n_folds, horizon, alg, game, task):

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    pool = mp.Pool( processes = ncpus ) 
    task = Evaluation(horizon, type)

    np.random.seed(1)
    distributions = []

    for _ in range(n_folds):
        p = np.random.uniform(0, 0.2) if task == 'easy' else np.random.uniform(0.4,0.5)
        distributions.append( [p, 1-p] )
        
    return np.asarray(  pool.map( partial( task.eval_policy_once, alg, game ), zip(distributions, range(n_folds) ) ) ) 

class Evaluation:

    def __init__(self, horizon,type):
        self.type = type
        self.horizon = horizon

    def get_outcomes(self, game):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        alg.reset()
        distribution, jobid = job
        np.random.seed(jobid)

        outcome_distribution =  {'spam':distribution[0], 'ham':distribution[1]}

        game.set_outcome_distribution( outcome_distribution )

        action_counter = np.zeros( (game.n_actions, self.horizon) )

        # generate outcomes obliviously
        outcomes = self.get_outcomes(game, jobid)

        for t in range(self.horizon):
            #print(t)

            # Environment chooses one outcome and one context associated to this outcome
            outcome = outcomes[t]

            # policy chooses one action
            # print('t', t,  'outcome', outcome, 'context', context)
            action = alg.get_action(t, None)

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, None )
            
            regret = np.array( [ game.delta(i) for i in range(game.n_actions) ]).T @ action_counter

        return  np.cumsum( regret ) 


###################################
# Synthetic Experiments
###################################

import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
parser.add_argument("--game", required=True, help="game")
parser.add_argument("--task", required=True, help="task") #easy of difficult
parser.add_argument("--algo_name", required=True, help="algorithme")
args = parser.parse_args()

horizon = int(args.horizon)
n_folds = int(args.n_folds)

game = games.label_efficient(  ) if args.game == 'LE' else games.apple_tasting(False) 

algo_name = args.algo_name.plit('_')

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

alg =  randcbp.RandCBP(  game, horizon, alpha, sigma, K, epsilon)  

result = evaluate_parallel(n_folds, horizon, alg, game, args.task  )

with gzip.open( './partial_monitoring/results/benchmark_randcbp/{}/_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.horizon, args.n_folds, args.algo_name) ,'wb') as f:
    pkl.dump(result,f)