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

      self.SignalMatrices = game.SignalMatricesAdim
    #   print(self.SignalMatrices)
      self.sts = [  s.T @ s  for s in  self.SignalMatrices ] 

      self.lbd = 0.001
      self.B0 = self.lbd * np.identity(self.M)
      self.b0 = np.zeros( (self.M, 1) ) 
      self.B = self.B0
      self.b = self.b0

      self.R = 1

      self.n , self.q = [] , []
      for i in range(self.N):
          self.n.append( np.zeros( self.A ) )
          self.q.append(  np.zeros( self.A ) )

    def in_simplex(self,p):
        res = False
        if p.sum() <= 1 and (p>=0).all():
            res = True
        return res

    def sample_from_g(self,):
        condition = False

        # 1. calc tilde{B} and tilde{b}
        one_vec = np.atleast_2d( np.ones(self.M - 1) ).T # \mathbb{1}_{M-1}
        # sub matrix of B
        C = self.B[:self.M-1, :self.M-1]
        d =  self.B[:self.M-1, -1].T
        D = (np.dot(d, one_vec.T) + np.dot(one_vec, d.T)) / 2
        f = self.B[-1, -1]
        # print('C', C, 'd', d, 'D', D, 'f', f)
        # sub vector of b
        b_alpha = self.b[:self.M-1,0 ].T #0
        b_M = self.b[-1, 0] 
        B_tilde = C - 2 * D + f * np.dot(one_vec, one_vec.T)
        b_tilde = f * one_vec - d + b_alpha - b_M * one_vec

        B_tilde_inv = np.linalg.inv(B_tilde)
        Bb = B_tilde_inv @ b_tilde
        # print( 'sampler dans la distribution:',  Bb , B_tilde_inv )

        while condition == False:
            p = np.random.normal( Bb, B_tilde_inv  )[0][0]
            # print('echantillon',p)
            if p<=1 and p>=0: #self.in_simplex(p):
                condition = True
        # print(p)
        return np.array( [ p , 1 - p ] )

    def accept_reject(self,):
        limit = 100
        threshold = 0
        while threshold < limit:
            p_tilde = self.sample_from_g()
            u_tilde = np.random.uniform(0, 1)
            # print(p_tilde)
            # print( 'mean', np.linalg.inv( self.B ) @ self.b , 'var', np.linalg.inv( self.B ) )
            # print( 'Ru',  self.R * u_tilde,  'F', self.F(p_tilde), 'G', self.G(p_tilde)  )
            
            threshold+=1
            if ( self.R * u_tilde <  self.F( p_tilde ) / self.G( p_tilde ) ).all()  :
                return p_tilde
            if threshold == 99:
                print('limit trial exceeded')

    def F(self, p):
        result = 1
        for i in range(self.N):
            # print('homemade KL', self.kl_div(  'package KL', scipy.special.kl_div( self.q[i], self.SignalMatrices[i] @ p ).sum() )
            # print('n[i]:', self.n[i].sum() , ' q[i]:', self.q[i] , ' Sp:', self.SignalMatrices[i] @ p , ' KL:', scipy.special.kl_div( self.q[i] , self.SignalMatrices[i] @ p  ).sum() )
            result *= np.exp( - self.n[i].sum() * scipy.special.kl_div( self.q[i] , self.SignalMatrices[i] @ p  ).sum() )
        # print('result',result)
        result *= np.diagonal( scipy.stats.norm.pdf( p , np.linalg.inv( self.B ) @ self.b ,   np.linalg.inv( self.B ) ) )
        return result

    def G(self,p):
        result = 1
        for i in range(self.N):
            result *= np.exp( -1/2 * self.n[i].sum() * np.linalg.norm( self.q[i]  - self.SignalMatrices[i] @ p )**2  )
        result *=  np.diagonal( scipy.stats.norm.pdf( p , np.linalg.inv( self.B ) @ self.b ,  np.linalg.inv( self.B ) ) ) 
        return result

    def get_action(self, t):

        if t < self.N:
            action = t

        else:
            p_tilde = self.accept_reject()
            # print('p_tilde:', p_tilde)
            action = np.argmin(  self.game.LossMatrix @ p_tilde  )
            # print('mean:', np.linalg.inv(self.B) @ self.b, '  var:', np.linalg.inv(self.B) )
            # print(p_tilde, self.game.LossMatrix @ p_tilde, action )

        return action

    def update(self, action, feedback, outcome):

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
            print('error')
        return idx