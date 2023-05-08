
import numpy as np


from multiprocess import Pool
#import multiprocessing as mp
import os

from functools import partial
import pickle as pkl
import gzip
# import plotly.graph_objects as go

import games

import cbpside
import randcbpside2
import randcbpside

import synthetic_data

import gzip
import pickle as pkl

import subprocess
from sklearn.preprocessing import PolynomialFeatures

######################
######################

def evaluate_parallel( evaluator, alg, game):

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    print('ncpus',ncpus)
    
    pool = Pool(processes=ncpus)

    np.random.seed(1)
    context_generators = []
    w = np.random.uniform(0,0.1)
    w = w / w.sum()

    for seed in range(evaluator.n_folds):
        
        if evaluator.context_type == 'linear':
            d = 10
            margin = 0.01
            
            contexts = synthetic_data.LinearContexts( w ) 
            context_generators.append( contexts )

        elif evaluator.context_type == 'quintic':
            d = 2
            contexts = synthetic_data.PolynomialContexts( d, 0.01)
            context_generators.append( contexts )

        else: 
            contexts = synthetic_data.ToyContexts( )
            context_generators.append( contexts )
        
    return  pool.map( partial( evaluator.eval_policy_once, alg, game ), zip(context_generators, range(n_folds) ) ) 

class Evaluation:

    def __init__(self, game_name, task, n_folds, horizon, game, label, context_type):
        self.game_name = game_name
        self.task = task
        self.n_folds = n_folds
        self.horizon = horizon
        self.game = game
        self.label =  label
        self.context_type = context_type
        

    def get_outcomes(self, game, ):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        alg.reset()

        context_generator, jobid = job

        np.random.seed(jobid)

        contexts = [ context_generator.get_context() for _ in range(self.horizon) ]

        if self.context_type == 'quintic':
            contexts = np.array(contexts).squeeze()
            contexts = PolynomialFeatures(6).fit_transform( contexts)
            contexts = contexts.tolist()
            dim = len(contexts[0])
            contexts = [ np.array(elmt).reshape( (dim,1) ) for elmt in contexts]
             
        cumRegret =  np.zeros(self.horizon, dtype =float)

        for t in range(self.horizon):

            if t % 1000 == 0 :
                print(t)

            context = contexts[t]
            distribution = context_generator.get_distribution(context)
            outcome = np.random.choice( 2 , p = distribution )

            action = alg.get_action(t, context)

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( distribution ) for i in range(alg.N) ]  )
            loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
            val = loss_diff @ np.array( distribution )
            cumRegret[t] =  val

        print('dump {}'.format(jobid))
        result = np.cumsum( cumRegret)
        with gzip.open( './partial_monitoring/contextual_results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(self.game_name, self.task, self.context_type, self.horizon, self.n_folds, self.label, jobid) ,'wb') as f:
            pkl.dump(result,f)

        return True

def run_experiment(game_name, task, n_folds, horizon, game, algos, labels, context_type):

    for alg, label in zip( algos, labels):

        print(label)
        evaluator = Evaluation(game_name, '{}'.format(task), n_folds, horizon, game, label, context_type)

        result = evaluate_parallel(evaluator, alg, game)
        
        with gzip.open( './partial_monitoring/contextual_results/{}/benchmark_{}_{}_{}_{}_{}.pkl.gz'.format(game_name, task, context_type, horizon, n_folds, label) ,'wb') as g:

            for jobid in range(n_folds):

                with gzip.open(  './partial_monitoring/contextual_results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(game_name, task, context_type, horizon, n_folds, label, jobid) ,'rb') as f:
                    r = pkl.load(f)

                pkl.dump( r, g)
                
                bashCommand = 'rm ./partial_monitoring/contextual_results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(game_name, task, context_type, horizon, n_folds, label, jobid)
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
    
    return True


###################################
# Synthetic Contextual experiments
###################################

import argparse

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

parser = argparse.ArgumentParser()

parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
parser.add_argument("--game", required=True, help="game")
parser.add_argument("--task", required=True, help="task")
parser.add_argument("--context_type", required=True, help="context type")
parser.add_argument("--algo", required=True, help="algorithme")

args = parser.parse_args()

horizon = int(args.horizon)
n_folds = int(args.n_folds)


games = {'LE': games.label_efficient(  ),'AT':games.apple_tasting(False)}
game = games[args.game]

dim = 28 if args.context_type == 'quintic' else 2


algos_dico = {
          
          'CBPside005':cbpside.CBPside(game, dim, 1.01, 0.05),

          'RandCBPside2005_1_5_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 5, 10e-7),
          'RandCBPside2005_18_5_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 5,  10e-7),
          'RandCBPside2005_116_5_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 5,   10e-7),
          'RandCBPside2005_132_5_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 5,   10e-7), 

          'RandCBPside2005_1_10_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 10,   10e-7),
          'RandCBPside2005_18_10_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 10,  10e-7),
          'RandCBPside2005_116_10_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 10,   10e-7),
          'RandCBPside2005_132_10_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 10,   10e-7), 

          'RandCBPside2005_1_20_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 20,   10e-7),
          'RandCBPside2005_18_20_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 20,   10e-7),
          'RandCBPside2005_116_20_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 20,   10e-7),
          'RandCBPside2005_132_20_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 20,  10e-7), 

          'RandCBPside2005_1_100_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 100,  10e-7),
          'RandCBPside2005_18_100_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 100,   10e-7),
          'RandCBPside2005_116_100_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 100,   10e-7),
          'RandCBPside2005_132_100_07':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 100,  10e-7), 

          'RandCBPside2005_1_5_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 5,  0.1),
          'RandCBPside2005_18_5_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 5,   0.1),
          'RandCBPside2005_116_5_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 5,   0.1),
          'RandCBPside2005_132_5_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 5,   0.1), 

          'RandCBPside2005_1_10_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 10,   0.1),
          'RandCBPside2005_18_10_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 10,   0.1),
          'RandCBPside2005_116_10_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 10,  0.1),
          'RandCBPside2005_132_10_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 10,   0.1), 

          'RandCBPside2005_1_20_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 20,   0.1),
          'RandCBPside2005_18_20_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 20,   0.1),
          'RandCBPside2005_116_20_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 20,  0.1),
          'RandCBPside2005_132_20_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 20,   0.1), 

          'RandCBPside2005_1_100_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1, 100,  0.1),
          'RandCBPside2005_18_100_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/8, 100,  0.1),
          'RandCBPside2005_116_100_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/16, 100, 0.1),
          'RandCBPside2005_132_100_01':randcbpside2.RandCPBside(game, dim, 1.01, 0.05, 1/32, 100,  0.1), 
           
          
          'RandCBPside005_1_5_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1, 5,   10e-7),
          'RandCBPside005_18_5_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/8, 5,  10e-7),
          'RandCBPside005_116_5_07':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/16, 5,   10e-7),
          'RandCBPside005_132_5_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/32, 5,   10e-7), 

          'RandCBPside005_1_10_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1, 10,  10e-7),
          'RandCBPside005_18_10_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/8, 10,   10e-7),
          'RandCBPside005_116_10_07':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/16, 10,   10e-7),
          'RandCBPside005_132_10_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/32, 10,   10e-7), 

          'RandCBPside005_1_20_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1, 20,   10e-7),
          'RandCBPside005_18_20_07':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/8, 20,   10e-7),
          'RandCBPside005_116_20_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/16, 20,   10e-7),
          'RandCBPside005_132_20_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/32, 20,   10e-7), 

          'RandCBPside005_1_100_07':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1, 100,   10e-7),
          'RandCBPside005_18_100_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/8, 100,  10e-7),
          'RandCBPside005_116_100_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/16, 100,   10e-7),
          'RandCBPside005_132_100_07':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/32, 100,   10e-7), 

          'RandCBPside005_1_5_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1, 5,   0.1),
          'RandCBPside005_18_5_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/8, 5,   0.1),
          'RandCBPside005_116_5_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/16, 5,   0.1),
          'RandCBPside005_132_5_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/32, 5,   0.1), 

          'RandCBPside005_1_10_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1, 10,   0.1),
          'RandCBPside005_18_10_01':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/8, 10,   0.1),
          'RandCBPside005_116_10_01':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/16, 10,   0.1),
          'RandCBPside005_132_10_01':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/32, 10,  0.1), 

          'RandCBPside005_1_20_01':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1, 20,   0.1),
          'RandCBPside005_18_20_01':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/8, 20,   0.1),
          'RandCBPside005_116_20_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/16, 20,   0.1),
          'RandCBPside005_132_20_01':randcbpside.RandCPBside(game, dim,  1.01, 0.05, 1/32, 20,   0.1), 

          'RandCBPside005_1_100_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1, 100,   0.1),
          'RandCBPside005_18_100_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/8, 100,   0.1),
          'RandCBPside005_116_100_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/16, 100,   0.1),
          'RandCBPside005_132_100_01':randcbpside.RandCPBside(game, dim,   1.01, 0.05, 1/32, 100,   0.1) }

algos = [ algos_dico[ args.algo ] ]
labels = [  args.algo ] 

run_experiment(args.game, args.task, n_folds, horizon, game, algos, labels, args.context_type)
