import geometry_v3
import numpy as np


class BPM:

    def __init__(self, game, horizon, p0, sigma0 ):
      self.game = game
      self.horizon = horizon

      self.N = game.n_actions
      self.M = game.n_outcomes
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.SignalMatrices = self.game.SignalMatrices  #geometry_v3.calculate_signal_matrices(game.FeedbackMatrix, self.N, self.M, self.A)
      self.sstInvs = [  np.linalg.inv( s @ s.T ) for s in  self.SignalMatrices ] 
      self.Ps = [ self.SignalMatrices[i].T @  self.sstInvs[i] @ self.SignalMatrices[i] for i in range(self.N) ] 

      self.p0 = p0 
      self.sigma0 = sigma0
      self.p = p0 
      self.sigma = sigma0
      self.sigmaInv = np.linalg.inv( sigma0 )

      self.sample_size = 100
      self.numC = 10
      self.samples = np.zeros( ( self.sample_size * self.numC * 2, self.M) )
      self.cInequality = self.getCells()
      self.ActiveActions = None
      self.n = np.zeros(self.N)

    def update(self, action, feedback, outcome):
      # self.feedback_counter[action][outcome] +=1
      self.n[action] += 1
      curr_sigmaInv = self.sigmaInv

      self.sigmaInv = curr_sigmaInv + self.Ps[action]
      self.sigma = np.linalg.inv(  self.sigmaInv )

      current_p = self.p
      Y_t = self.SignalMatrices[action] @ np.eye(self.M)[outcome]
      new_p = self.sigma @ ( curr_sigmaInv @ current_p + self.SignalMatrices[action].T @ self.sstInvs[action] @ Y_t  )
      new_p = abs(new_p)
      self.p = new_p/sum(new_p)

    def populateSamples(self, t):

      for i in range(self.sample_size):

        x = np.random.uniform(0, 1, self.M)
        stocX = x / x.sum()

        xnormMat = ( ( stocX - self.p ).T @ self.sigmaInv @ ( stocX - self.p) ) / np.log(t+2)
        samp1 = np.zeros(self.M)
        samp2 = np.zeros(self.M)
        # print('xnormmat',xnormMat, 'stocX', stocX)
        xnorm = np.sqrt( xnormMat )

        for j in range(self.numC):
          samp1 =  ( self.p + ( ( stocX  -self.p ) / xnorm ) * ( j / self.numC ) )
          samp2 = ( self.p - ( ( stocX - self.p ) / xnorm ) * ( j / self.numC ) ) 
          self.samples[ 2 * i * self.numC + 2 * j ]= samp1.T
          self.samples[ 2 * i * self.numC + 2 * j + 1 ] = samp2.T

    def getCells(self,):
      cInequality = np.zeros( (self.N, self.M, self.N) )
      for i in range(self.N):
        res = self.game.LossMatrix[i,...] - self.game.LossMatrix
        cInequality[i] = res.T
      return cInequality

    def get_action(self, t):

      self.populateSamples(t)
      currentActiveActions = []
      score = np.zeros(self.N)
      for i in range(self.N):
        temp = self.cInequality[i].T @ self.samples.T
        boolMat =  temp <= np.zeros( (self.N, 2 * self.numC * self.sample_size) ) 
        sumBool = np.sum( boolMat, 0)
        if np.max(sumBool) == self.N:
          currentActiveActions.append(i)

      self.activeActions = currentActiveActions
      
      numActiveActions = len( currentActiveActions )
      for i in range( numActiveActions ):
        score[  self.activeActions[i] ] = self.horizon - self.n[ self.activeActions[i] ]
      # print('active actions', currentActiveActions, 'score', score, 'sumbool', sumBool )
      chosen = np.argmax(score)
      return chosen


      # self.feedback_counter = np.zeros( (self.N,self.M) )
      # self.sample_num = 100
      
    # def getOptimalAction(self, p):
    #   expectedLoss = self.game.LossMatrix @ p
    #   return np.argmin( expectedLoss )

    # def sampleP(self,):
    #   z= np.random.normal(0,1, self.M)
    #   A = self.sigma @ self.sigma.T
    #   r = A @ z
    #   return r
    
    # def nonNegative(self, p):
    #   for i in range(self.M):
    #     if p[i] < 0:
    #       return False
    #   return True

    # def get_action(self, t):
    #   optimalActions = []
    #   c = 0
    #   while c<self.sample_num:
    #     p = self.sampleP()
    #     if self.nonNegative(p):
    #       optimal = self.getOptimalAction(p)
    #       optimalActions.append( optimal )
    #       c += 1
    #   print('optimal actions', optimalActions)
    #   minFeedback = 100
    #   minFeedbackOptimal = -1
    #   feedbackRowwise = np.sum( self.feedback_counter,1 )
    #   for i in optimalActions:
    #     if feedbackRowwise[i] < minFeedback:
    #       minFeedback = feedbackRowwise[i]
    #       minFeedbackOptimal = i    
    #   return minFeedbackOptimal  