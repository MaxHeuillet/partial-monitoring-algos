
# from signal import valid_signals
import geometry_v3
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy
import cyipopt
import tnlp


class PM_DMED:

    def __init__(self, game, horizon, c):

      self.game = game
      self.horizon = horizon
      self.N = game.n_actions
      self.M = game.n_outcomes
    
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.LC = set( [ i for i in range(self.N) ] )
      self.ZR = set(  [ i for i in range(self.N) ] )
      self.LN = set([])

      self.constraintNum = 300
      self.VERY_LARGE_DOUBLE = 10000000
      self.VERY_SMALL_DOUBLE  = 0.000001

      self.c = c

      self.feedback = [] 
      for i in range(self.N):
          self.feedback.append(  np.ones( self.A ) ) # ones and not zeros for avoiding infeasible models

    def get_action(self, t):
      action = self.LC.pop()

      return action 

    def reset(self, ):

      self.LC = set( [ i for i in range(self.N) ] )
      self.ZR = set( [ i for i in range(self.N) ] )
      self.LN = set( [] )

      self.feedback = [] 
      for i in range(self.N):
          self.feedback.append(  np.ones( self.A ) ) # ones and not zeros for avoiding infeasible models

    def update(self, action, feedback, outcome, context, t):
      #print('Zc', self.LC, 'Zr', self.ZR, 'Zn', self.LN )
      self.feedback[action][ self.feedback_idx(feedback) ] += 1

      if action in self.ZR:
        self.ZR.remove(action)

      
      for i in range(self.N):
        # print('not in', i not in self.ZR )
        # print('counter', sum(self.feedback[i]) < self.c * np.sqrt( np.log( t+1 ) ) )
        # print('ZR', self.ZR , 'counter', sum(self.feedback[i]), 'threshold', self.c * np.sqrt( np.log( t+1 ) ) )
        if i not in self.ZR and sum(self.feedback[i]) < self.c * np.sqrt( np.log( t+1 ) ):
          # print( '/forced exploration of {}'.format(i) )
          self.LN.add(i)

      ins_actions = self.insufficientActions(t+1)
      # print('insufficient actions', ins_actions)
      for j in range(self.N):
        if j in ins_actions and j not in self.ZR:
          self.LN.add(j)
          
      if len(self.LC) == 0:
        self.LC = self.LN.copy()
        self.ZR = self.LN.copy()
        self.LN = set([])  
        # print(self.LC)

    def use_cache(self, t):
      return (t>100) and ( (t%10)!=0 )

    def MLE(self,accuracy):

      lb = [ 0.0001 for _ in range(self.M)]
      ub = [ 0.999 for _ in range(self.M)]

      cl = [1.0]
      cu = [1.0]

      nlp = cyipopt.Problem( n=self.M, m=1,
                             problem_obj=tnlp.MLE_NLP(self.game, self.A, self.feedback ),
                             lb=lb, ub=ub, cl=cl, cu=cu )

      accuracy = 0.1

      cyipopt.set_logging_level(0)

      nlp.add_option('sb', 'yes')
      nlp.add_option('print_level', 0)
      nlp.add_option('mu_strategy', 'adaptive')
      nlp.add_option("tol", 0.1 * accuracy)
      nlp.add_option("acceptable_tol", 1 * accuracy)
      nlp.add_option("bound_relax_factor", 0.0)
      nlp.add_option("jac_c_constant", "yes")
      nlp.add_option("hessian_approximation", "limited-memory")

      x0 = [ 1/self.M for j in range(self.M) ]

      try:
        x, info = nlp.solve(x0)
        if info['status'] == 0 :
          solution = x
        else:
          # print('erreur optimisation 1')
          solution = np.random.uniform(0, 1, self.M)
          solution = solution / solution.sum()
  
      except TypeError:
        # print('erreur optimisation 2')
        solution = np.random.uniform(0, 1, self.M)
        solution = solution / solution.sum()
      #print('x',x)
      #print('info', info)

      return solution

    def insufficientActions(self, t):

      if self.use_cache(t): #use previous p
        suffExplore = self.suffExplore_cache.copy()
      else:
        p = self.MLE( 0.1/pow(t+1, 0.5) )
        #print('problematic p', p )
        #print('t', t)
        #print('value', np.max( [np.log(t), 1.0] ))

        suffExplore = self.sufficientExploration( p , np.max( [np.log(t), 1.0] )  )
        self.suffExplore_cache = suffExplore

      actions = [ i for i in range(self.N) if suffExplore[i] >= sum(self.feedback[i]) ]

      return actions

    # def getRandomPoint(self,):
    #   random_point = np.random.uniform(0,1,self.M)
    #   random_pbt = random_point / random_point.sum()
    #   return random_pbt


##########################################################################################
##########################################################################################
##########################################################################################

    def sufficientExploration(self, p, logt ):
      #print('major test')

      ps = [ self.getConstrainedRandomPoint( p ) for c in range(self.constraintNum) ]
      #print('ps')

      m = gp.Model( )
      m.Params.LogToConsole = 0

      vars = { i:0 for i in range(self.N) }

      Deltas = np.zeros(self.N)
      numZeros = 0
      for i in range(self.N):
        for j in range(self.M):
          ldiff = self.game.LossMatrix[i,j] - self.game.LossMatrix[self.game.i_star, j]
          #print('ldiff', ldiff)
          #print('p', p)
          Deltas[i] +=  ldiff * p[j]
        if Deltas[i] < self.VERY_SMALL_DOUBLE:
          numZeros += 1

      #print('hey')
      
      if numZeros>=2: #degenerate optimal solution
        vals = np.zeros(self.N)
        for i in range(self.N):
          if Deltas[i] < self.VERY_SMALL_DOUBLE:
            vals[i] = self.VERY_LARGE_DOUBLE
        return vals 

      for i in range(self.N):
        if i == self.game.i_star: 
          pass
        else:
          varName = "N_{}".format(i) 
          vars[i] =  m.addVar(0, GRB.INFINITY, Deltas[i], GRB.CONTINUOUS, varName ) 
          m.update()

      m.update()

      #print('hola')
      #print('vars', vars)
      for c in range(len(ps)):
        if self.getOptimalAction(ps[c]) == self.game.i_star:
          pass
        else:
          ps[c] = self.findBoundaryByBisection( p, ps[c])
          ConstExpr = 0
          for i in range(self.N):
            if i == self.game.i_star:
              pass
            else:
              di = self.kl_i(p, ps[c], i)
              ConstExpr += di * vars[i]

            if i == self.N:
              m.addConstr( ConstExpr >= logt,  'constraint{}'.format(c) )
      
      #print('heyyy')
      
      m.optimize()
      vals = np.zeros(self.N)
      for i in range(self.N):
        if i == self.game.i_star:
          vals[i] = self.VERY_LARGE_DOUBLE
        else:
          vals[i] =  vars[i].X 

      #print('vals')
      return vals

    def getOptimalAction(self, p):
      deltas = []
      for i in range( len(self.game.LossMatrix) ):
        deltas.append( self.game.LossMatrix[i,...].T @ p )
      return np.argmin(deltas)

    def getConstrainedRandomPoint(self, p):
      pSample = np.zeros( self.M )
      for a in range(self.A):
        psum = 0
        sum = 0
        for j in range(self.M):
          if self.game.SignalMatricesAdim[self.game.i_star][ a ][j]:
            pSample[j] = np.random.uniform(0,1)
            psum += p[j]
            sum += pSample[j]
        for j in range(self.M):
          if self.game.SignalMatricesAdim[self.game.i_star][ a ][j]:
            pSample[j] *= psum/sum
      return pSample

    def findBoundaryByBisection(self, p, p2):
      lb = 0
      ub = 0
      while ( (ub-lb) > 0.01 ):
        c = (ub+lb) / 2
        pm = (1-c) * p + c * p2
        if self.getOptimalAction(self.game.LossMatrix, pm )==self.game.i_star:
          lb = c
        else:
          ub = c
      return (1-lb) * p + lb * p2

    def kl_i(self, p1, p2, i):
      p1_sym = np.zeros(self.A)
      p2_sym = np.zeros(self.A)
      for j in range(self.M):
        p1_sym[ self.feedback_idx( self.game.FeedbackMatrix[i,j]) ] += p1[j]
        p2_sym[ self.feedback_idx( self.game.FeedbackMatrix[i,j]) ] += p2[j]
      return scipy.special.kl_div( p1_sym , p2_sym ).sum()

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


    


