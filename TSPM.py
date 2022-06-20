import geometry_v3
import numpy as np
import scipy


class TSPM_alg:

    def __init__(self, game, horizon,  ):
      self.game = game
      self.horizon = horizon

      self.N = game.n_actions
      self.M = game.n_outcomes
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.SignalMatrices = geometry_v3.calculate_signal_matrices(game.FeedbackMatrix, self.N, self.M, self.A)
      self.sts = [  s.T @ s  for s in  self.SignalMatrices ] 

      self.lbd = 0.001
      self.B0 = self.lbd * np.identity(self.M)
      self.b0 = np.zeros(2)
      self.R = 1
      self.B = self.B0
      self.b = self.b0
      self.n = [ np.zeros( self.M ) for i in range(self.N) ]
      self.q = [ np.zeros( self.M ) for i in range(self.N) ]


    def in_simplex(self,p):
        res = False
        if p.sum == 1 and (p>=0).all():
            res = True
        return res

    def sample_from_g(self,):
        condition = False
        B_inv = self.B #self.C - 2 * self.D + norm.cdf(x, mean, std) 
        B_inv = np.linalg.inv( B_inv )
        product = B_inv @ self.b 
        print('product', product, 'B_inv', B_inv)
        while condition == False:
            p = np.random.normal( product, B_inv  )
            print(p)
            if self.in_simplex(p):
                condition = True
        return [ p , 1- p.sum() ]

    def accept_reject(self,):
        while True:
            p_tilde = self.sample_from_g()
            u_tilde = np.random.uniform(0,1)
            if self.R * u_tilde < self.F(p_tilde) * self.G(p_tilde):
                return p_tilde

    def F(self, p):
        result = 1
        for i in range(self.N):
            result *= np.exp( -self.n[i].sum() * scipy.special.kl_div( self.q[i], self.SignalMatrices[i] @ p )  )
        result *= self.prior
        return result

    def G(self,p):
        result = 1
        for i in range(self.N):
            result *= np.exp( -1/2 * self.n[i].sum() * np.linalg.norm( self.q[i] - self.SignalMatrices[i] @ p )**2  )
        result *= self.prior
        return True

    def get_action(self, t):

      p_tilde = self.accept_reject()
      action = np.argmax( [ self.game.L[i,...] @ p_tilde for i in range(self.N)] )

      return action

    def update(self, action, feedback, outcome):

      self.B = self.B + self.sts[action]
      self.b = self.b + self.SignalMatrices[action] @ np.eye(self.M)[outcome]
      
      self.n[action][outcome] += 1
      self.q = [ self.n[action]/sum(self.n[action]) for i in range(self.N) ]