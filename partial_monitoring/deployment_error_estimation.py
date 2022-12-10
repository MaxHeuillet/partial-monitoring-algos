import numpy as np
from multiprocess import Pool
from functools import partial
import plotly.graph_objects as go
import gzip
import pickle as pkl

import games
import synthetic_data
import random_algo


import cpb_side
import cpb_side_gaussian
import linucb


import PGIDSratio
import synthetic_data
import numpy as np
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class FakeIds():

    def __init__(self, M, imbalance):

        self.imbalance = imbalance
        self.M = M
        self.dim = len(imbalance)
        self.labels = [i for i in range(self.dim)]
        
        
    def get_contexts(self, horizon, game):

        contexts = np.empty( ( horizon, 10) )
        outcomes = np.zeros( horizon, dtype = int)

        for i in range(horizon):

            label = np.random.choice( self.labels , p = self.imbalance)

            contexts[i] = np.zeros(10)
            pred_label = np.random.choice(  self.labels , p = self.M[label] )
            contexts[i][pred_label] = 1
            
            if pred_label != label:
                outcomes[i] = 1
            else:
                outcomes[i] = 0
                
        return outcomes, contexts

    def get_context(self, ):
        label = np.random.choice( self.labels , p = self.imbalance)

        context = np.zeros(self.dim)
        pred_label = np.random.choice(  self.labels , p = self.M[label] )
        context[pred_label] = 1
            
        if pred_label != label:
            outcome = 1
        else:
            outcome = 0
                
        return outcome, context

class Evaluation:

    def __init__(self, test):
        self.test = test
        pass

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):
        print('try')
        
        context_generator, ground_truth, n_labels, jobid = job

        alg.set_nlabels(n_labels)
        alg.reset()
        
        np.random.seed(jobid)

        epsilon = 0.025
        t = 0
        queries_counter = 0
        latest_estimate = np.ones(len( ground_truth )) * 1000
        # print('ground truth', self.ground_truth )
        status = True

        start = time.time()

        while status == True:
            
            outcome, context = context_generator.get_context() 
            context = context.reshape((-1,1))

            if t % 10000 == 0 and t>0 :
                print(t, 'latest estimate', latest_estimate)

            if t>2 and alg.name == 'randcbpside':
                estimates = []
                for i in range( n_labels ):
                    sim = np.zeros( n_labels )
                    sim[i] = 1
                    estimate = alg.contexts[1]['weights'] @ sim
                    estimates.append( estimate[0] )
                latest_estimate = estimates

            elif t>2 and alg.name == 'random':
                latest_estimate = alg.weights[:,0]

        
            if ( abs( ground_truth - latest_estimate  ) <= epsilon ).all() :
                status = False            

            # policy chooses one action
            #print('t', t,  'outcome', outcome, 'context', context)
            action = alg.get_action(t, context)

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )

            t = t+1
            if action == 1:
                queries_counter += 1

            end = time.time()
            if end - start >= 60: #7200
                status = False 
                t, queries_counter = -1, -1  
            
        result = [t, queries_counter]
        print('hey')

        return result


class Random():

    def __init__(self, game, ):

        self.name = 'random'
        self.game = game
        # self.weights = np.ones( (self.n_labels,2) )
        # self.feedbacks = np.zeros( (self.n_labels,2) )
        # self.N = np.zeros( self.n_labels )

    def get_action(self, t, context = None ):
        
        pbt = np.ones( self.game.n_actions ) / self.game.n_actions
        action = np.random.choice(self.game.n_actions, 1,  p = pbt )[0]
        return action

    def update(self, action, feedback, outcome, t, context):
        if action == 1:
            idx = np.argmax(context)
            self.feedbacks[idx][feedback] += 1
            self.N[idx] += 1
            estimates = [ self.feedbacks[i] / self.N[i] if self.N[i] !=0 else np.zeros( (1,2) ) for i in range(len(self.N) ) ] 
            self.weights = np.vstack(estimates)

    def reset(self,):
        self.weights = np.ones( (self.n_labels,2) )
        self.feedbacks = np.zeros( (self.n_labels,2) )
        self.N = np.zeros( self.n_labels )

    def set_nlabels(self, nlabels):
        self.n_labels = nlabels


def confusion_matrix(M, errors, n_labels):

    for i in range(n_labels):

        M[i][i] -= errors[i]
        n_splits = np.random.randint(1, n_labels-1) 

        coefs = np.random.uniform(0, 1, n_splits)
        coefs = coefs / sum(coefs)

        for idx in range(n_splits):

            status = True
            while status == True:
                location = np.random.randint(0, n_labels)
                if location != i:
                    status = False

            M[i][location] += coefs[idx] *  errors[i]

    return M

def truncate(value):
    if value<-1 or value>1:
        status = False
        while status == False:
            value = abs( np.random.normal(0, 0.1 ) )
            if value <1:
                status = True
    return value


def get_ground_truth(M, imbalance):
    # correct probabilities
    probas_correct = np.diag(M) * imbalance

    # correct probabilities
    probas_incorrect = M.copy()
    np.fill_diagonal( probas_incorrect, 0 ) 
    probas_incorrect = np.sum( probas_incorrect.T * imbalance, 1)

    # final probabilites
    final_probas = probas_incorrect / (  probas_correct + probas_incorrect )
    return final_probas


def evaluate_parallel( alg, game):

    ncpus = 5#int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    print('ncpus',ncpus)
    
    pool = Pool(processes=ncpus)

    np.random.seed(1)

    list_groundtruth = []
    list_generators = []
    list_nlabels = []

    evaluator = Evaluation( None ) 
    n_trials = 5
    print('hye')
    for jobid in range(n_trials):

        n_labels = np.random.randint(3, 10)
        list_nlabels.append(n_labels)

        imbalance = np.array( [ np.random.uniform(50, 75) if np.random.uniform(0,1)<0.1 else np.random.uniform(0,25) for _ in range(n_labels) ] )
        imbalance = imbalance / sum(imbalance)

        M = np.identity(n_labels)
        errors = np.array( [ truncate( abs( np.random.normal(0.5, 0.5 ) ) ) if np.random.uniform(0,1)<0.1 else truncate( abs( np.random.normal(0, 0.2 ) ) ) for _ in range(n_labels) ] )
        M = confusion_matrix(M, errors, n_labels)
        
        ground_truth = get_ground_truth(M, imbalance)

        list_groundtruth.append( ground_truth )

        list_generators.append( FakeIds(M, imbalance ) )

    print('launch')
    return  pool.map( partial( evaluator.eval_policy_once, alg, game ), zip( list_generators, list_groundtruth, list_nlabels, range(n_trials) ) ) 

import time 
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


game = games.apple_tasting(False)

n_trials = 5

results_steps = np.zeros( (n_trials, 2) )
results_queries = np.zeros( (n_trials, 2) )

alg = Random(  game,  )
# algo =  cpb_side_gaussian.RandCPB_side(game , None, 1.01, 0.001, 1/8, 10, False, 10e-7) 

result = evaluate_parallel( alg,  game)

print(result)
