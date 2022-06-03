from math import log, exp, pow
import numpy as np

class PMGame(object):
    __slots__ = ['N', 'M', 'OutcomeDist', 'LossMatrix', 'FeedbackMatrix', 'FeedbackMatrix_symb', 'Actions_dict', 'Outcomes_dict', 'title', 'game_type']
    def __init__(self, N, M, title =""):
        self.N = N # Number of learner actions
        self.M = M # Number of environment outcomes
        self.OutcomeDist = np.ones(M )/M # outcome distribution (for stochastic PM)
        self.LossMatrix = np.ones(shape=(N,M) ) # Loss MAtrix
        self.FeedbackMatrix = np.empty(shape=(N,M) ) # Feedback (numeric form)
        self.FeedbackMatrix_symb = np.empty(shape=(N,M), dtype=object) # Feedback (symbolic)        
        self.Actions_dict = { a : "{0}".format(a) for a in range(self.N)} # Actions semantic
        self.Outcomes_dict = { a : "{0}".format(a) for a in range(self.M)} # Outcomes semantic
        self.title = title
        self.game_type = "generic"

def is_set(x, n, K):
    return x & 2**(K-n-1) != 0 

def arm_reward(x, n, K):
    if is_set(x,n,K):
        return 1.
    else:
        return 0.


#Generate a PM instance for Bernoulli Bandit 
def BernoulliBandit(Arms):
    K = len(Arms) # number of arms
    Arms = np.array(Arms )
    pm = PMGame(K,2**K, str(K)+"-armed bandit") # PM Game with K actions and 2**K outcomes
    pm.game_type = "bandit"
    
    
    pm.Outcomes_dict = { a : "{0:b}".format(a).zfill(int( log(pm.M,2) ) ) for a in range(pm.M)}
    pm.Actions_dict= { a : "arm {0}".format(a) for a in range(K) }
  
    ## 1 - Each outcome is a binary reward vector of dimension K encoded as an integer
    for x in range(pm.M):
        px = 1.
        for a in range(K):
            if is_set(x,a,K):
                px *= Arms[a]
            else:
                px *= 1. - Arms[a]
        pm.OutcomeDist[x] = px

    ## 2 - Loss and Feedback matrices
    for a in range(K):
        for x in range(pm.M):
            pm.LossMatrix[a,x] = 1.0 - arm_reward(x,a,K)
            pm.FeedbackMatrix[a,x] = arm_reward(x,a,K)
            if arm_reward(x,a,K):
                pm.FeedbackMatrix_symb[a,x] = 'win '
            else:
                pm.FeedbackMatrix_symb[a,x] = 'loss'
    return pm



def AppleTasting(Dist):
    pm = PMGame(2, 2,"Apple tasting game") # It's a PM Game with 2 actions and 2 outcomes
    pm.game_type = "apple tasting"
    assert len(Dist) == 2
    pm.OutcomeDist = np.array(Dist )
    
    pm.LossMatrix = np.array(
     [[1, 0],
      [0, 1]] )

    pm.FeedbackMatrix = np.array(
            [[0, 0],
             [1, -1]] )

    pm.FeedbackMatrix_symb = np.array(
            [['blind', 'blind'],
             ['rotten', 'good']], dtype=object)

    pm.Actions_dict = { 0:'sell apple', 1:'taste apple'}
    pm.Outcomes_dict = { 0:'rotten', 1:'good'}
    
    return pm