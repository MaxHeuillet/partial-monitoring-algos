
import numpy as np
import os

import multiprocessing as mp

from functools import partial
import pickle as pkl
import gzip
# import plotly.graph_objects as go
import os


import games

import random_algo
import cpb_side
import partial_monitoring.randcbpside as randcbpside

import synthetic_data

import gzip
import pickle as pkl

from sklearn.preprocessing import PolynomialFeatures

######################
######################

def evaluate_parallel(nbCores, n_folds, horizon, alg, game, type, context_type):
    print("nbCores:", nbCores, "nbFolds:", n_folds, "Horizon:", horizon)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    print(ncpus)
    pool = mp.Pool( processes = ncpus ) 
    task = Evaluation(horizon, type)

    np.random.seed(1)
    distributions = []
    context_generators = []
    context_types = []

    for jobid in range(n_folds):
        print('prepare job {}'.format(jobid) )
        
        p = np.random.uniform(0, 0.2) if type == 'easy' else np.random.uniform(0.4,0.5)
        distributions.append( [p, 1-p] )

        if context_type == 'linear':
            d = 2
            margin = 0.01
            contexts = synthetic_data.LinearContexts( np.array([0,1]), 0, d, margin) 
            context_generators.append( contexts )
            context_types.append('linear')

        elif context_type == 'quintic':
            contexts = synthetic_data.QuinticContexts( 2, 0.01)
            context_generators.append( contexts )
            context_types.append('quintic')

        else: 
            contexts = synthetic_data.ToyContexts( )
            context_generators.append( contexts )
            context_types.append('toy')
        
    return np.asarray(  pool.map( partial( task.eval_policy_once, alg, game ), zip(distributions , context_generators ,context_types, range(n_folds) ) ) ) 

class Evaluation:

    def __init__(self, horizon,type):
        self.type = type
        self.horizon = horizon

    def get_outcomes(self, game, job_id):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        #print('reset alg')

        alg.reset()

        distribution, context_generator, context_type, jobid = job

        #print('job info received')

        np.random.seed(jobid)

        outcome_distribution =  {'spam':distribution[0],'ham':distribution[1]}

        game.set_outcome_distribution( outcome_distribution, jobid )

        # action_counter = np.zeros( (game.n_actions, self.horizon) )

        # generate outcomes obliviously
        outcomes = self.get_outcomes(game, jobid)
        contexts = [ context_generator.get_context(outcome) for outcome in outcomes ]

        #print('contexts generated')

        if context_type == 'quintic':
            contexts = np.array(contexts).squeeze()
            contexts = PolynomialFeatures(6).fit_transform( contexts)
            contexts = contexts.tolist()
            dim = len(contexts[0])
            contexts = [ np.array(elmt).reshape( (dim,1) ) for elmt in contexts]
            # contexts.reshape( shape[0], shape[1],1 )
             
        # context_generator.generate_unique_context()
        # contexts = [ context_generator.get_same_context(outcome) for outcome in outcomes ]
        # print('theta', context_generator.w )

        cumRegret =  np.zeros(self.horizon, dtype =float)
        
        for t in range(self.horizon):
            #print(t)
            
            # Environment chooses one outcome and one context associated to this outcome
            outcome = outcomes[t]
            context = contexts[t]

            # policy chooses one action
            # print('t', t,  'outcome', outcome, 'context', context)
            action = alg.get_action(t, context)

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )
            
            # print('nu', alg.nu / alg.n )
            regret = game.LossMatrix[action, outcome] - np.min( game.LossMatrix[...,outcome] )
            # print( 'regret:' , regret )
            cumRegret[t] =  regret
            # print()
        # regret = np.array( [ game.delta(i) for i in range(game.n_actions) ]).T @ action_counter
        #context_regret = np.cumsum( cumRegret )

        return  np.cumsum( cumRegret ) #regret

def run_experiment(name, task, n_cores, n_folds, horizon, game, algos, colors, labels, context_type):

    for alg, color, label in zip( algos, colors, labels):

        result = evaluate_parallel(n_cores, n_folds, horizon, alg, game, '{}'.format(task), context_type )

        with gzip.open( './partial_monitoring/contextual_results/{}/{}_{}_{}_{}_{}.pkl.gz'.format(name, task, context_type, horizon, n_folds, label) ,'wb') as f:
            pkl.dump(result,f)

    return True


###################################
# Synthetic Contextual experiments
###################################


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
args = parser.parse_args()

horizon = args.horizon
n_folds = args.n_folds

game = games.label_efficient(  ) # games.apple_tasting(False) 

   # algos = [ random_algo.Random(  game, horizon, ),     
   #         cpb_side.CPB_side(  game, horizon, 1.01, 0.05), 
   #         cpb_side.CPB_side(  game, horizon, 1.01, 0.001), 
   #         cpb_side_gaussian.RandCPB_side(game, horizon, 1.01, 0.05, 1/8, 10, False, 10e-7),
   #         cpb_side_gaussian.RandCPB_side(game, horizon, 1.01, 0.001, 1/8, 10, False, 10e-7)   ]

   # colors = [  [0,0,0],  [0,250,0] , [0,0,250],  [200,0,200], [150,0,150]  ] 
   # labels = [  'random',  'CBPside005',  'CPBside0001', 'RandCBPside005', 'RandCBPside0001' ] 

context_type = 'linear' #'quintic'

#algos = [ random_algo.Random(  game, horizon, )  ]
#labels = ['random']
algos = [ cpb_side.CPB_side(  game, 2, horizon, 1.01, 0.05)  ]
labels = ['CBPside005']
#algos = [ cpb_side.CPB_side(  game, 2, horizon, 1.01, 0.001)  ]
#labels = ['CBPside0001']
#algos = [ cpb_side_gaussian.RandCPB_side(game, horizon, 1.01, 0.05, 1/8, 10, False, 10e-7)  ]
#labels = ['RandCBPside005']
#algos = [ cpb_side_gaussian.RandCPB_side(game, horizon, 1.01, 0.001, 1/8, 10, False, 10e-7)  ]
#labels = ['CBPside0001']

colors = [  [0,0,0] ]

#run_experiment('LE', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels, context_type)
run_experiment('LE', 'difficult', n_cores, n_folds, horizon, game, algos, colors, labels, context_type)

#run_experiment('AT', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels, context_type)
#run_experiment('AT', 'difficult', n_cores, n_folds, horizon, game, algos, colors, labels, context_type)
