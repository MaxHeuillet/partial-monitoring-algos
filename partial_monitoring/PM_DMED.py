
from signal import valid_signals
import geometry_v3
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy


class PM_DMED:

    def __init__(self, game, horizon):
      self.game = game
      self.horizon = horizon
      self.N = game.n_actions
      self.M = game.n_outcomes
    
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.Zc = [ i for i in range(self.N) ]
      self.Zr = [ i for i in range(self.N) ]
      self.Zn = None

      self.n , self.symbol_dist = [] , []

      for i in range(self.N):
          self.n.append( np.zeros( self.A ) )
          self.symbol_dist.append(  np.zeros( self.A ) )

    def get_action(self, t):

      if t < self.N:
        action = t

      else:
        action = np.random.choice(self.Zc)


    def reset(self,p ):
      pass

    
    def optimization_problem(self, p):
      m = gp.Model( )
      m.Params.LogToConsole = 0

      vars = []
      for i in range(self.M):
        varName =  'p_{}'.format(i) 
        vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName) )
        m.update()

      obj = 0
      for i in range(self.N):
        calc = self.game.SignalMatricesAdim[i] @ vars
        kl_dv = 0
        for k in range(self.M):
          if p[k] == 0 or calc[k] == 0:
            kl_dv += 0
          kl_dv += p[k] * np.log( p[k] / calc[k] )
          obj += kl_dv

      m.setObjective(obj, GRB.MINIMIZE)
      m.optimize()
      result = m.objVal
      
      return result

    def get_random_pbt(self,):
      random_point = np.random.uniform(0,1,self.M)
      random_pbt = random_point / random_point.sum()
      return random_pbt

    def update(self, action, feedback, outcome, t):

      self.Zc.remove(action)
      self.Zr.remove(action)

      idx = self.feedback_idx( feedback ) 
      e_y = np.zeros( (self.A, 1) )
      e_y[idx] = 1

      self.n[action][idx] += 1
      self.symbol_dist[action] = self.n[action] / self.n[action].sum() 

      ######### algorithm PM-DMED:

      hat_p = self.optimization_problem( self.get_random_pbt() )
      i_t =  np.argmin(  self.game.LossMatrix @ hat_p  ) 

      if i_t not in self.Zr:
        self.Zn.append(i_t)
      
      # sub sampled actions
      actions_subset = [ i for i in self.N if i not in self.Zr ]
      actions_subset = [ i for i in actions_subset if self.n[i] < self.c * np.sqrt( np.log(t) ) ]
      self.Zn.extend( actions_subset )

    def use_cache(self, t):
      return (t>100) and ( (t%10)!=0 )

    def getConstrainedRandomPoint(self, p):
      pSample = []
      for a in range(self.A):
        psum = 0
        sum = 0
        for j in range(self.M):
          if self.game.LossMatrix[a][j]:
            pSample[j] = np.random.uniform(0,1)
            psum += p[j]
            sum += pSample[j]
        for j in range(self.M):
          if self.game.LossMatrix[a][j]:
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
        p1_sym[ self.game.FeedbackMatrix[i,j] ] += p1[j]
        p2_sym[ self.game.FeedbackMatrix[i,j] ] += p2[j]
      return scipy.special.kl_div( p1_sym , p2_sym ).sum()


    def sufficientExploration(self, p, constraintNum):

      ps = []
      for c in range(constraintNum):
        ps.append( self.getConstrainedRandomPoint( p ) )
      
      m = gp.Model( )
      m.Params.LogToConsole = 0

      vars = []
      Deltas = np.zeros(self.N)
      numZeros = 0
      for i in range(self.N):
        for j in range(self.M):
          Deltas[i] +=  [ self.game.LossMatrix[i,j] - self.game.LossMatrix[self.game.i_star, j] ] * p[j]
        if Deltas[i] < VERY_SMALL_DOUBLE:
          numZeros += 1
      
      if numZeros>=2:
        for i in range(self.N):
          if Deltas[i] < VERY_SMALL_DOUBLE:
            vals[i] = VERY_LARGE_DOUBLE
        return vals 

      obj = 0
      for i in  range(self.N):
        if i == self.game.i_star: 
          pass
        varName = "N_{}".format(i) 
        vars[i].append( m.addVar(0, GRB.INFINITY, Deltas[i], GRB.CONTINUOUS, varName ) ) 
        m.update()
        
      m.setObjective(15*x1 + 18*x2 + 30*x3, GRB.MAXIMIZE)
      m.update()

      expression = None
      for c in range(len(ps)):
        if self.getOptimalAction(self.game.LossMatrix, ps[c])==self.game.i_star:
          pass
        ps[c] = self.findBoundaryByBisection(self.game.LossMatrix, p, ps[c], self.game.i_star)
      for i in range(self.N):
        if i == self.game.i_star:
          pass
        double_di = self.kl_i(p, ps[c], i)
        expression += di * vars[i]

        if i == N:
          constName = "N_{}".format(i) 
        m.addConstr( expression[l] == ldiff[l],  'constraint{}'.format(l) )
        vars[i].append( m.addVar(0, GRB.INFINITY, Deltas[i], GRB.CONTINUOUS, varName ) ) 

        m.optimize()
        vals = []
        for i in range(N):
          if i == self.game.i_star:
            vals.append(VERY_LARGE_DOUBLE)
          else:
            vals.append( vars[i].X )
        return vals


    def MLE(self,accuracy, t):

      if self.use_cache(t):
        samples = cache.copy()
      else:
        samples = [ self.get_random_pbt() for _ in range(1000) ]
        cache = samples.copy()


#      VectorXd MLE(double accuracy){
#     // Create a new instance of your nlp
#     //  (use a SmartPtr, not raw)
#     SmartPtr<TNLP> nlp = new MLE_NLP(lossMatrix, feedbackMatrix, N, M, A, feedback);
#     //m_app->Options()->SetNumericValue("tol", 1.);
#     //nlp->num_option("acceptable_tol", accuracy);

#     // Create a new instance of IpoptApplication
#     //  (use a SmartPtr, not raw)
# //    SmartPtr<IpoptApplication> app = new IpoptApplication(true); //console output
#     SmartPtr<IpoptApplication> app = new IpoptApplication(false); //no console output
#     app->Options()->SetNumericValue("tol", 0.1*accuracy);
#     app->Options()->SetNumericValue("acceptance_tol", 1*accuracy);
# //    app->Options()->SetStringValue("nlp_scaling_method", "none");
#     app->Options()->SetNumericValue("bound_relax_factor", 0.0);
#     app->Options()->SetStringValue("jac_c_constant", "yes");
# //    app->Options()->SetStringValue("mehrotra_algorithm", "yes");
# //    app->Options()->SetStringValue("warm_start_init_point", "yes");
# //    app->Options()->SetStringValue("warm_start_bound_push", "yes");
# //    app->Options()->SetStringValue("linear_solver", "ma77");
# //    app->Options()->SetStringValue("linear_solver", "ma86");
# //    app->Options()->SetStringValue("linear_solver", "ma97");
# //    app->Options()->SetStringValue("linear_solver", "pardiso");
# //    app->Options()->SetStringValue("linear_solver", "mumps");
#     app->Options()->SetStringValue("hessian_approximation", "limited-memory"); //l-bfgs is very fast

#     //here change some options
 
#     // Intialize the IpoptApplication and process the options
#     ApplicationReturnStatus status;
#     status = app->Initialize();
#     bool success = (status == Solve_Succeeded) || (status == Solved_To_Acceptable_Level);
#     if (!success) {
#       printf("\n\n*** Error during initialization!\n");
#       //return (int) status;
#     }

#     // Ask Ipopt to solve the problem
#     status = app->OptimizeTNLP(nlp);

#     if (success) {
#       //printf("\n\n*** The problem solved!\n");
#       //std::cout << "feedback=" << feedback << std::endl;
#       VectorXd solution = ipopt_mle_solution;
#       return solution;
#     }
#     else {
#       printf("\n\n*** The problem FAILED!\n");
#       std::cout << "feedback=" << feedback << std::endl;
#       VectorXd randSolution = VectorXd::Zero(M);
#       double sum = 0.;
#       for(uint j=0;j<M;++j){
#         randSolution(j) = unitNormal(randomEngine);
#         sum += randSolution(j);
#       }
#       return randSolution / sum;
      



      


 