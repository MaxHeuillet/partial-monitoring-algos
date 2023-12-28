import numpy as np
import geometry_v3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader
import copy

from scipy.optimize import minimize
import copy
import pickle
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import multiprocessing
import os

import statsmodels.api as sm


class CBPside():

    def __init__(self, game, alpha, m, num_cls, device):

        self.name = 'CBPside_logistic'
        self.device = device
        
        self.num_workers = 1 #int ( os.environ.get('SLURM_CPUS_PER_TASK', default=1) )
        print('num workers', self.num_workers  )

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = len( np.unique( game.FeedbackMatrix ) )

        self.SignalMatrices = game.SignalMatrices

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [], self.num_workers)
        self.mathcal_N = game.mathcal_N

        self.N_plus =  game.N_plus

        self.V = game.V

        self.v = game.v 

        self.W = self.getConfidenceWidth( )

        self.alpha = alpha
            
        self.eta =  self.W**(2/3) 
        self.m = m

        self.num_cls = num_cls

    def convert_pred_format(self,pred):
        final = []
        for k in range(self.game.N):
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                if np.unique(self.game.FeedbackMatrix[k]).shape[0] > 1:
                    per_action.append( pred[s] )
                else:
                    per_action.append( pred[s] )
            final.append( np.array(per_action) )
        return final

    def convert_conf_format(self,conf, dc_list):
        final = []
        for k in range(self.game.N):
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                if np.unique(self.game.FeedbackMatrix[k]).shape[0] > 1:
                    per_action.append( conf[s] )
                else:
                    per_action.append( conf[s] )
            final.append( np.array([max(per_action)]) )
        return final, None

    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            # print(pair)
            for k in self.V[ pair[0] ][ pair[1] ]:
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec , np.inf) ] )
        return W

    def reset(self, d):
        self.d = d
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.X1_train, self.y1, = [], []

        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':[], 'labels':[], 'weights': None, 'V_it_inv': self.lbd * np.identity(self.d) } )

    def encode_context(self, X):

        ci = np.zeros(1, self.d)
        x_list = []
        for a in range(self.A):
            inputs = []
            for l in range(a):
                inputs.append(ci)
            inputs.append(X)
            for l in range(a+1, self.A):
                inputs.append(ci)
            inputs = torch.cat(inputs, dim=1).to(torch.float32)
            x_list[a] = inputs
        return x_list


    def get_action(self, t, X):

        self.X = X
        halfspace = []
        self.x_list = self.encode_context(X)

        q = []
        w = []

        for a in range(self.N):
            pred = model( self.x_list, self.contexts[a]['labels'])
            q.append(pred)


        
        # print( 'estimate', q )
        # print('conf   ', w )
        # print('index', self.index)

        #print('########################### eliminate actions')
        for pair in self.mathcal_N:
            tdelta, c = 0, 0
            for k in  self.V[ pair[0] ][ pair[1] ]:
                tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
            #print('pair', pair, 'tdelta', tdelta, 'confidence', c)
            # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
            # print('sign', np.sign(tdelta) )
            #tdelta = tdelta[0]
            #c = np.inf
            if( abs(tdelta) >= c):
                halfspace.append( ( pair, np.sign(tdelta) ) ) 
            
        
        # print('########################### compute halfspace')
        # print('halfspace', halfspace)
        #print('##### step1')
        code = self.halfspace_code(  sorted(halfspace) )
        #print('##### step2')
        P_t = self.pareto_halfspace_memory(code, halfspace)
        #print('##### step3')
        if len(P_t)>1:
            N_t = self.neighborhood_halfspace_memory(code, halfspace)
            # N_t = [ [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [6, 7], [6, 8], [6, 9], [6, 10], [7, 8], [7, 9], [7, 10], [8, 9], [8, 10], [9, 10] ]
            #self.neighborhood_halfspace_memory(code,halfspace)
            #print(N_t)
        else:
            N_t = []
        print('P_t', len(P_t), P_t, 'N_t', N_t)
        
        # print('########################### rarely sampled actions')
        Nplus_t = []
        for pair in N_t:
            Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )
        Nplus_t = np.unique(Nplus_t)

        V_t = []
        for pair in N_t:
            V_t.extend( self.V[ pair[0] ][ pair[1] ] )
        V_t = np.unique(V_t)
        R_t = []

        for k, eta in zip(V_t, self.eta):
            if eta>0:
                indx = self.index[k]
                feature = self.dc_list[k][indx]
                V_it_inv = self.contexts[k]['V_it_inv']
                V_it_inv = V_it_inv.to(self.device)
                V_feature = torch.matmul(V_it_inv, feature.t() )
                feature_V = torch.matmul(feature, V_it_inv)
                val =  torch.matmul(feature_V, V_feature).item()
                t_prime = t+2
                rate = self.eta[k] * t_prime**(2/3)  * ( self.alpha * np.log(t_prime) )**(1/3)  
                if val > 1/rate : 
                    R_t.append(k)

                V_it_inv = V_it_inv.cpu()
                del V_it_inv
                torch.cuda.empty_cache()

        # print('########################### play action')
        union1= np.union1d(  P_t, Nplus_t )
        union1 = np.array(union1, dtype=int)
        
        explored = 1 if len(union1)>=2 else 0

        
        print('union1', union1, 'R', R_t)
        S =  np.union1d(  union1  , R_t )
        S = np.array( S, dtype = int)
        # print('S', S)
        S = np.unique(S)
        # print()
        values = { i:self.W[i]*w[i] for i in S}
        # print('value', values)
        action = max(values, key=values.get)
        history = {'monitor_action':action, 'explore':explored, }
            
        return action, history


    def update(self, action, feedback, outcome, t, X):

        
        feedbacks = np.zeros( (self.A, self.A) )
        feedbacks[feedback] = 1

        self.contexts[action]['labels'].append( Y_t )
        self.contexts[action]['features'].append( self.x_list )

        V_it_inv = self.contexts[action]['V_it_inv']
        self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 
        weights = Y_it @ X_it.T @ self.contexts[action]['V_it_inv']
        self.contexts[action]['weights'] = weights

        if (t>self.N):

            if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): 

                pass

        return global_loss, global_losses




    def halfspace_code(self, halfspace):
        string = ''
        for element in halfspace:
            pair, sign = element
            string += '{}{}{}'.format(pair[0],pair[1], sign)
        return string 


    def pareto_halfspace_memory(self, code, halfspace):

        result = self.memory_pareto.get(code)
        
        if result is None:

            result = geometry_pulp.getParetoOptimalActions(
                self.game.LossMatrix, 
                self.N, 
                self.M, 
                halfspace, 
                self.num_workers  ) 
            self.memory_pareto[code] = result

        return result

    def neighborhood_halfspace_memory(self, code, halfspace):

        result = self.memory_neighbors.get(code)
        

        if result is None:
            print('step 3 b')
            result = geometry_pulp.getNeighborhoodActions(
                self.game.LossMatrix, 
                self.N, 
                self.M, 
                halfspace, 
                self.mathcal_N, 
                self.num_workers
            )
            self.memory_neighbors[code] = result

        return result

