
import numpy as np
import os

from multiprocessing import Pool
from functools import partial
import pickle as pkl
import gzip
# import plotly.graph_objects as go

import games

import random_algo
import cpb_side
import cpb_side_gaussian

import synthetic_data

import gzip
import pickle as pkl

from sklearn.preprocessing import PolynomialFeatures

######################
######################

def evaluate_parallel(nbCores, n_folds, horizon, alg, game, type, context_type):
    print("nbCores:", nbCores, "nbFolds:", n_folds, "Horizon:", horizon)
    pool = Pool(processes = nbCores) 
    task = Evaluation(horizon, type)

    np.random.seed(1)
    distributions = []
    context_generators = []
    context_types = []

    for jobid in range(n_folds):
        
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

        alg.reset()

        distribution, context_generator, context_type, jobid = job

        np.random.seed(jobid)

        outcome_distribution =  {'spam':distribution[0],'ham':distribution[1]}

        game.set_outcome_distribution( outcome_distribution, jobid )

        # action_counter = np.zeros( (game.n_actions, self.horizon) )

        # generate outcomes obliviously
        outcomes = self.get_outcomes(game, jobid)
        contexts = [ context_generator.get_context(outcome) for outcome in outcomes ]

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

            if t % 1000 == 0 :
                print(t)

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

        print(label)

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
n_cores = None

context_type = 'linear'

game = games.label_efficient(  ) #, games.apple_tasting(False) ]:

algos = [ cpb_side_gaussian.RandCPB_side(game, horizon, 1.01, 0.05, 1/8, 10, False, 10e-7) ]    

colors = [  [0,0,0]  ] 
labels = [  'RandCBPside005' ] 

run_experiment('LE', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels, context_type)


# ###################################
# # Non-contextual
# ###################################

# ### Label Efficient game:

# horizon = 10
# n_cores = 1
# n_folds = 1

# game = games.label_efficient(  )

# algos = [ random_algo.Random(  game, horizon, ), 
#           cpb.CPB(  game, horizon, 1.01),  
#           PM_DMED.PM_DMED(  game, horizon, 100), 
#           PM_DMED.PM_DMED(  game, horizon, 10), 
#           PM_DMED.PM_DMED(  game, horizon, 5), 
#           PM_DMED.PM_DMED(  game, horizon, 1),
#           PM_DMED.PM_DMED(  game, horizon, 0.1),  
#           TSPM.TSPM_alg(  game, horizon, 1),
#           TSPM.TSPM_alg(  game, horizon, 0),
#           bpm.BPM(game, horizon) ]

# colors = [  [0,0,0], [0,0,0], [250,0,0], [0,250,0], [0,150,0], [0,0,250], [0,0,200], [0,0,150], [0,0,100],  [225,0,225], [150 , 0, 150], [0 , 250, 250] ] 
# labels = [   'random', 'CBP', 'RandCBP', 'PM_DMEDc100', 'PM_DMEDc10', 'PM_DMEDc5', 'PM_DMEDc1', 'PM_DMEDc01', 'TSPM_R0', 'TSPM_R1', 'BPM_LEAST'  ]  

# run_experiment('LE', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels)

# run_experiment('LE', 'difficult', n_cores, n_folds, horizon, game, algos, colors, labels)

# ### Apple Tasting game:

# horizon = 10
# n_cores = 1
# n_folds = 1

# game = games.apple_tasting(False)

# algos = [ random_algo.Random(  game, horizon, ), 
#           cpb.CPB(  game, horizon, 1.01),  
#           cpb_gaussian.CPB_gaussian(  game, horizon, 1.01, True, 1/16, 10, False),
#           PM_DMED.PM_DMED(  game, horizon, 100), 
#           PM_DMED.PM_DMED(  game, horizon, 10), 
#           PM_DMED.PM_DMED(  game, horizon, 1),
#           PM_DMED.PM_DMED(  game, horizon, 0.1),  
#           TSPM.TSPM_alg(  game, horizon, 1),
#           TSPM.TSPM_alg(  game, horizon, 0),
#           bpm.BPM(game, horizon) ]

# colors = [  [0,0,0], [250,0,0], [0,250,0], [0,150,0], [0,0,250], [0,0,200], [0,0,150], [0,0,100],  [225,0,225], [150 , 0, 150], [0 , 250, 250] ] 
# labels = [   'random', 'CBP', 'RandCBP', 'PM_DMEDc100', 'PM_DMEDc10', 'PM_DMEDc1', 'PM_DMEDc01', 'TSPM_R0', 'TSPM_R1', 'BPM_LEAST'  ]  

# run_experiment('AT', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels)

# run_experiment('AT', 'difficult', n_cores, n_folds, horizon, game, algos, colors, labels)
