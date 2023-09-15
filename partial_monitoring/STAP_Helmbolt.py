import numpy as np
import geometry_v3


import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from scipy.optimize import minimize
import copy
import pickle
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from scipy.special import logit, expit
    
class CustomDataset(Dataset):
    def __init__(self, ):
        self.obs = None
        self.labels = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.labels[index]
    
    def append(self, X , y,):
        self.obs = X if self.obs is None else np.concatenate( (self.obs, X), axis=0) 
        self.labels = y if self.labels is None else np.concatenate( (self.labels, y), axis=0)

class STAP_Helmbolt():

    def __init__(self, game, budget, m, H, device):

        self.name = 'helmbolt'
        self.device = device

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.budget = budget
        self.counter = 0
        self.over_budget = False

        self.m = m
        self.H = H

        self.nb_mistakes = 0


    def reset(self, d):
        self.d = d
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.hist = CustomDataset()
        self.feedbacks = []

        self.over_budget = False
        self.counter = 0
        self.nb_mistakes = 0

        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':[], 'labels':[], 'weights': None, 'V_it_inv': np.identity(self.d) } )

    def get_action(self, t, X, y_pred):

        prediction = self.contexts[0]['weights'] @ X
        probability = expit(prediction)
        self.pred_action = 0 if probability < 0.5 else 1

        print('prediction', prediction, probability, self.pred_action)

        r = np.random.uniform(0,1)
        flip_proba = np.sqrt( (1+self.nb_mistakes)/(t+1) )

        if self.pred_action == 1 and r <= flip_proba and self.over_budget==False:
            action = 0
            explored = 1
        else:
            action = self.pred_action
            explored = 0

        history = {'monitor_action':action, 'explore':explored, 'model_pred':y_pred, 'counter':self.counter, 'over_budget':self.over_budget}
            
        return action, history

    def update(self, action, feedback, outcome, t, X):

        ### update exploration component:
        e_y = np.zeros( (self.M,1) )
        e_y[outcome] = 1
        Y_t = self.game.SignalMatrices[action] @ e_y 

        if self.counter > self.budget:
            self.over_budget = True

        if action == 0:
            self.counter += 1

            if outcome != self.pred_action:
                self.nb_mistakes += 1

        self.contexts[action]['labels'].append( Y_t )
        self.contexts[action]['features'].append( X )

        Y_it = np.array( self.contexts[action]['labels'] )
        X_it =  np.array( self.contexts[action]['features'] )

        Y_it =  np.squeeze(Y_it, 2).T # Y_it.reshape( (sigma, n) )
        X_it =  np.squeeze(X_it, 2).T #X_it.reshape( (d, n) )
        
        V_it_inv = self.contexts[action]['V_it_inv']
        self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 
        weights = Y_it @ X_it.T @ self.contexts[action]['V_it_inv']
        self.contexts[action]['weights'] = weights
        