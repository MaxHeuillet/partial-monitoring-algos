import geometry_v3
import numpy as np
import scipy
from scipy.stats import multivariate_normal

class TSPM_alg:

    def __init__(self, game, horizon, R ):
      self.game = game
      self.horizon = horizon
      self.R = R

      self.N = game.n_actions
      self.M = game.n_outcomes
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.SignalMatrices = game.SignalMatricesAdim
    #   print(self.SignalMatrices)
      self.sts = [  s.T @ s  for s in  self.SignalMatrices ] 

      self.lbd = 0.001
      self.B0 = self.lbd * np.identity(self.M)
      self.b0 = np.zeros( (self.M, 1) ) 
      self.B = self.B0
      self.b = self.b0

      self.n , self.q = [] , []

      for i in range(self.N):
          self.n.append( np.zeros( self.A ) )
          self.q.append(  np.zeros( self.A ) )

    def in_simplex(self,p):
        res = False
        if p.sum()<=1 and (p<=1).all() and (p>=0).all():
            res = True
        return res

    def reset(self,):
        self.B0 = self.lbd * np.identity(self.M)
        self.b0 = np.zeros( (self.M, 1) ) 
        self.B = self.B0
        self.b = self.b0
        self.n , self.q = [] , []

        for i in range(self.N):
            self.n.append( np.zeros( self.A ) )
            self.q.append(  np.zeros( self.A ) )



    def sample_from_g(self,):
        condition = False

        one_vec = np.atleast_2d( np.ones(self.M - 1) ).T 
        C = self.B[:self.M-1, :self.M-1]
        d =  np.atleast_2d(self.B[:self.M-1, -1]).T 

        D = (np.dot(d, one_vec.T) + np.dot(one_vec, d.T)) / 2
        f = self.B[-1, -1]

        b_alpha = np.atleast_2d(self.b[:self.M-1,0 ]).T 
        b_M = self.b[-1, 0] 

        B_tilde = C - 2 * D + f * np.dot(one_vec, one_vec.T)
        b_tilde = f * one_vec - d + b_alpha - b_M * one_vec

        B_tilde_inv = np.linalg.inv(B_tilde)
        Bb = B_tilde_inv @ b_tilde

        while condition == False: 
            p =  np.random.multivariate_normal(  Bb[:,0],  B_tilde_inv  )
            if self.in_simplex(p):
                condition = True
        return np.concatenate( [ p , [1 - p.sum()] ] )

    def accept_reject(self,t):
        limit = 100
        threshold = 0
        while threshold < limit:
            p_tilde = self.sample_from_g()
            u_tilde = np.random.uniform(0, 1)
            threshold += 1
            if self.R * u_tilde <  self.F( p_tilde ) / self.G( p_tilde )   :
                #print('rejects', threshold)
                return p_tilde
            if threshold == limit-1:
                print('limit threshold ', t)
                #return [0.5] * self.M

    def F(self, p):
        result = 1
        mean_vec = np.linalg.inv( self.B ) @ self.b
        inv = np.linalg.inv( self.B )
        for i in range(self.N):
            result *= np.exp( - self.n[i].sum() * scipy.special.kl_div( self.q[i] , self.SignalMatrices[i] @ p  ).sum() )
        posterior = multivariate_normal(mean=  mean_vec[:,0], cov= inv ).pdf( p )
        #print('F posterior', posterior)
        result *= posterior 

        return result

    def G(self,p):
        result = 1
        mean_vec = np.linalg.inv( self.B ) @ self.b
        inv = np.linalg.inv( self.B )
        for i in range(self.N):
            result *= np.exp( -1/2 * self.n[i].sum() * np.linalg.norm( self.q[i]  - self.SignalMatrices[i] @ p ) ** 2  )
        posterior =  multivariate_normal(mean=  mean_vec[:,0], cov= inv  ).pdf( p )
        #print('G posterior', posterior)
        result *=  posterior 
        return result

    def get_action(self, t):


        #counter = [i for i in range(self.N) if self.n[i].sum() < 10 * self.A ] 
        #if len(counter) > 0:
        #    action = counter[0]

        #if t < self.N:
        #    action = t

        #else:

        p_tilde = self.accept_reject(t)

        action = np.argmin(  self.game.LossMatrix @ p_tilde  ) 

        return action

    def update(self,  action, feedback, outcome, X, t):

      self.B = self.B + self.sts[action]
      
      idx = self.feedback_idx( feedback ) 
      e_y = np.zeros( (self.A, 1) )
      e_y[idx] = 1

      self.b += self.SignalMatrices[action].T @ e_y 

      self.n[action][idx] += 1
      self.q[action] = self.n[action] / self.n[action].sum() 
    #   print('n', self.n, 'q',self.q)

    def feedback_idx(self, feedback):
        idx = None
        if self.N ==2:
            if feedback == 0:
                idx = 0
            elif feedback == 1:
                idx = 1
        elif self.N == 3:
            if feedback == 1:
                idx = 0
            elif feedback == 0.5:
                idx = 1
            elif feedback == 0.25:
                idx = 2
        else:
            if feedback == 1:
                idx = 0
            elif feedback == 2:
                idx = 1
        return idx
