import gurobipy as gp
from gurobipy import GRB

n_cores = 16
horizon = 100
n_folds = 25

LossMatrix = np.array( [ [1, 0], [0, 1] ] )
FeedbackMatrix =  np.array([ [1, 1],[1, 0] ])
outcome_distribution = {'spam':0.05,'ham':0.95 }
    # LossMatrix = np.array( [ [1, 1, 0],[0, 1, 1], [1, 0, 1] ] )
    # FeedbackMatrix = np.array( [ [1, 0, 0],[0, 1, 0], [0, 0, 1] ] )
    # outcome_distribution = {'spam':0.2, 'ham':0.6, 'other':0.2}

task = SyntheticCase(LossMatrix, FeedbackMatrix, horizon, outcome_distribution) 
print('action optimale', task.i_star)
result = task.cpb_vanilla_v2( 10 )  

    # print( result )
    # print()
    
# regret = np.cumsum( np.array( [ task.delta(i) for i in range(2) ] ).T @ result )
# plt.plot(regret)

# game = games.apple_tasting(False, outcome_distribution)
# regret= np.array([ game.delta(i) for i in range(game.n_actions) ]).T @ result
# plt.plot(   regret , label = 'Bianchi', color = 'green' )

# result =  eval_feedexp_parallel(task, n_cores, n_folds, horizon, True, 'Piccolboni' ) 
# plt.plot(  np.mean( result , 0 ) , label = 'Piccolboni', color = 'green' )

# result =  eval_feedexp_parallel(task, n_cores, n_folds, horizon, True, 'Bianchi' ) 
# plt.plot(  np.mean( result , 0 ) , label = 'Bianchi', color = 'orange' )

# plt.legend()


        # p = [ ppl.Variable(j) for j in range(M) ] # declare M ppl Variables
        # cs = ppl.Constraint_System() # declare polytope constraints

        # p belongs to $\Delta_M$ the set of M dimensional probability vectors
        # cs.insert( sum( p[j] for j in range(M)) == 1 )
        # for j in range(M):
        #     cs.insert(p[j] >= 0)

        # <p , li2 - li > \geq 0
        # loss = []
        # for i2 in range(N):
        #     if i2 == i:
        #         pass
        #     else:
        #         for j in range(M):
        #             loss.append(  ( LossMatrix[i2][j] - LossMatrix[i][j] ) * p[j] )
        # if len(loss)>0:
        #     cs.insert( sum(loss) >= 0 )

        # # halfspace constraint: h(i,j) * (loss[i]-loss[j])^top @ p > 0 
        # halfspaceExpr = []
        # for element in halfspace:
        #     pair, sign = element[0], element[1]
        #     if sign == 0:
        #         pass
        #     else:
        #         for j in range(M):
        #             coef = sign * ( LossMatrix[ pair[0] ][j] - LossMatrix[ pair[1] ][j] )
        #             if(coef != 0):
        #                 halfspaceExpr.append( coef * p[j] )
        
        # print('halfspaceexpr', sum(halfspaceExpr) )
        # if len(halfspaceExpr)>0:
        #     cs.insert( sum(halfspaceExpr) > 0 )

        # polytope = ppl.NNC_Polyhedron(cs)
        # if polytope.is_empty() == False:
        #     P.append(i)

    return P

    # #simple constraint:
    # p = [ ppl.Variable(j) for j in range(M) ] # declare M ppl Variables
    # cs = ppl.Constraint_System() # declare polytope constraints

    # # p belongs to $\Delta_M$ the set of M dimensional probability vectors
    # cs.insert( sum( p[j] for j in range(M)) == 1 )
    # for j in range(M):
    #     cs.insert(p[j] >= 0)

    # # < p, loss(i1) - loss(i2) > = 0 
    # result = 0
    # for j in range(M):
    #     result += ( LossMatrix[i2][j] - LossMatrix[i1][j] ) * p[j]
    # cs.insert( result == 0)

    # # < p, loss(i1) - loss(i2) > = 0 
    # loss = []
    # for i3 in range(N):
    #     #print('i3',i3,'i2',i2,'i1',i1)
    #     if i3 == i1 or i3 == i2:
    #         pass
    #     else:
    #         for j in range(M):
    #             loss.append(  (  LossMatrix[i3][j] - LossMatrix[i1][j]) * p[j] )
    # #print('loss', loss, sum(loss), len(loss))
    # if len(loss) > 0:
    #     cs.insert( sum(loss) >= 0)

    # # halfspace constraint: h(i,j) * (loss[i]-loss[j])^top @ p > 0 
    # halfspaceExpr = []
    # for element in halfspace:
    #     pair, sign = element[0], element[1]
    #     if sign== 0:
    #         pass
    #     else:
    #         for j in range(M):
    #             coef = sign * ( LossMatrix[ pair[0] ][j] - LossMatrix[  pair[1] ][j] )
    #             if(coef != 0):
    #                 halfspaceExpr.append( coef * p[j] )
        
    # #print('halfspaceexpr', halfspaceExpr)
    # if len(halfspaceExpr) > 0:
    #     cs.insert( sum(halfspaceExpr) > 0 )

    # return ppl.NNC_Polyhedron(cs)



env = gp.Env( empty = True )
m = gp.Model("mip1")
m.addVar(vtype=GRB.BINARY, name="x")
m.setObjective(x + y + 2 * z, GRB.MINIMIZE)
m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
m.optimize()


FeedbackMatrix =  np.array([ [1, 1],[1, 0] ])


# class SyntheticCase:

#     def __init__(self, LossMatrix, FeedbackMatrix, horizon, outcome_dist ):
 
#         self.LossMatrix = LossMatrix 
#         self.FeedbackMatrix = FeedbackMatrix 
        
#         self.outcome_dist = outcome_dist
#         self.i_star = self.optimal_action( )
        
#         self.horizon = horizon
#         self.n_actions = len(self.LossMatrix)
#         self.n_outcomes = len(self.LossMatrix[0])

#     def optimal_action(self, ):
#         deltas = []
#         for i in range(len(self.LossMatrix)):
#             deltas.append( self.LossMatrix[i,...].T @ list( self.outcome_dist.values() ) )
#         return np.argmin(deltas)

#     def delta(self,action):
#         return ( self.LossMatrix[action,...] - self.LossMatrix[self.i_star,...] ).T @ list( self.outcome_dist.values() ) 

#     def set_outcomes(self, job_id):
#         np.random.seed(job_id)
#         #self.means = runif_in_simplex( len( LossMatrix[0] ) )
#         self.outcomes = np.random.choice( self.n_outcomes , p= list( self.outcome_dist.values() ), size= self.horizon) 

#     def get_feedback(self, FeedbackMatrix, action, outcome):
#         return FeedbackMatrix[ action ][ outcome ] 


#     def cpb_vanilla_v2(self, job_id):
#         import geometry_v3

#         action_counter = np.zeros( (self.n_actions, self.horizon) )

#         self.set_outcomes(job_id)

#         N = self.n_actions
#         M = self.n_outcomes
#         A = geometry_v3.alphabet_size(self.FeedbackMatrix, N,M)
#         print('n-actions', N, 'n-outcomes', M, 'alphabet', A)

#         SignalMatrices = geometry_v3.calculate_signal_matrices(self.FeedbackMatrix, N,M,A)

#         n = np.zeros(N)
#         nu = [  np.zeros( A ) for i in range(N)] 

#         pareto_actions = geometry_v3.getParetoOptimalActions(self.LossMatrix, N, M, [])
#         neighborhood_actions = geometry_v3.getNeighborhoodActions(self.LossMatrix, N, M, [])
#         # print('neighborhood_actions', neighborhood_actions)

#         v =  geometry_v3.getV(self.LossMatrix, N, M, A, SignalMatrices, neighborhood_actions)
#         # print('V', v)
#         W = geometry_v3.getConfidenceWidth(neighborhood_actions, v, N);
#         print(W)
#         alpha = 1.01

#         eta = []
#         for i in range(N):
#             eta.append( W[i]**2/3 )

#         for t in range(self.horizon):
          
#           if(t<N):

#             action = t
#             feedback = self.FeedbackMatrix[action][ self.outcomes[t]  ]
#             n[action] += 1
#             nu[action][ feedback ] += 1

#             for i in range(N):
#                 if i == action:
#                     action_counter[i][t] = action_counter[i][t-1] +1
#                 else:
#                     action_counter[i][t] = action_counter[i][t-1]

#           else: 

#             halfspace = []

#             for pair in neighborhood_actions:
#                 tdelta = 0
#                 c = 0
#                 for k in range(N):
#                     tdelta += v[pair[0]][pair[1]][k].dot( nu[k] ) / n[k]
#                     c += np.linalg.norm( v[pair[0]][pair[1]][k], np.inf ) * np.sqrt( alpha * np.log(t) / n[k]  )
                
#                 if( abs(tdelta) >= c):
#                     halfspace.append( ( pair, np.sign(tdelta) ) )
#                 # else:
#                 #     halfspace.append( ( pair, 0 ) )
#                 # print('pair', pair,  'tdelta', tdelta, 'c', c)

#             # print('halfspace', halfspace)
#             P_t = geometry_v3.getParetoOptimalActions(self.LossMatrix, N, M, halfspace)
#             N_t = geometry_v3.getNeighborhoodActions(self.LossMatrix, N, M, halfspace)

#             Nplus_t = []

#             for pair in N_t:
#               Nplus_t.append(pair[0])
#               Nplus_t.append(pair[1])
              
#             R_t = []
#             for k in range(N):
#               if n[k] <=  eta[k] * geometry_v3.f(t, alpha) :
#                 R_t.append(k)
#             R_t = np.unique(R_t)

#             union1= np.union1d( Nplus_t, P_t )
#             union1 = np.array(union1, dtype=int)
#             # print('union1', union1)
#             S =  np.union1d(  union1  ,  R_t)
#             S = np.array( S, dtype = int)
#             # print('S', S)
#             S = np.unique(S)
              
#             value = [ W[i]**2/n[i] for i in S]
#             # print('value', value)
#             istar = np.argmax(value)

#             print('N_t', N_t, 'Nplus_t', Nplus_t,'R_t',R_t, 'P_t', P_t , 'S', S, 'istar', istar)

#             feedback = self.FeedbackMatrix[istar][ self.outcomes[t] ]
#             n[istar] += 1
#             nu[istar][ feedback ] += 1

#             for i in range(N):
#                 if i == istar:
#                     action_counter[i][t] = action_counter[i][t-1] +1
#                 else:
#                     action_counter[i][t] = action_counter[i][t-1]

#         return action_counter

A = geometry.get_alphabet_size(FeedbackMatrix)
print('A', type(A) )

def get_signal_matrices(H):
    N, M = H.shape
    A = get_alphabet_size(H)
    signal_matrices = []
    for i in range(N):
        signal_matrix = np.zeros( (A,M) )
        for j in range(M):
            a = H[i][j]
            signal_matrix[a][j] = 1
        signal_matrices.append(signal_matrix)
    return signal_matrices

get_signal_matrices(FeedbackMatrix)



# def observer_vector(L, H, i, j, mathcal_K_plus):
#     A = np.vstack( global_signal(H) )
#     Lij = L[i,...] - L[j,...]
#     # print('Lij', Lij)
#     # print('globalsignal',global_signal(H))
#     x, res, rank, s = np.linalg.lstsq(A.T, Lij) 
#     lenght = [ len( np.unique(H[k]) ) for k in mathcal_K_plus]
#     x = iter( np.round(x) )
#     return [ np.array( list(islice( x, i)) ) for i in lenght] 



    # def ucb1(self, method, job_id):
    #     '''Play the given bandit over T rounds using the UCB1 strategy.'''

    #     self.set_outcomes(job_id)

    #     regret = []
        
    #     sum_estimators = [0] * self.n_actions
    #     counters = [0] * self.n_actions
        
    #     k_star = np.argmax(self.means)
    #     gaps = self.means[k_star] - self.means
        
    #     for t in range(self.horizon):
    #         outcome = self.outcomes[t]
            
    #         error_pbt = 1 / (self.horizon**2)
            
    #         UCB =  [ sum_estimators[k]  / counters[k] + np.sqrt(  2 * np.log( 1 / error_pbt ) / counters[k] ) if counters[k] !=0 else np.inf for k in range(self.n_actions) ] 
    #         # print(  [ sum_estimators[k]  / counters[k]  if counters[k] !=0 else np.inf for k in range(self.n_actions) ]  )
    #         action = np.argmax( UCB )
    #         reward =  self.get_feedback(action, outcome)
    #         counters[action] = counters[action] + 1
    #         sum_estimators[action] =   sum_estimators[action] + reward

    #         # policy suffers loss and regret
    #         regret.append( gaps[action ]  )

    #     return np.array(regret)


    # def W(self, mathcal_N, N_bar, observer_vector ):
    #     W = np.zeros( len(N_bar) )
    #     for pair in mathcal_N:
    #         for k in N_bar:
    #             value = np.fabs( observer_vector[ pair[0] ][ pair[1] ][k] ).max()
    #             W[k] = max( W[k], value  )
    #     return W

    # def cpb_vanilla_v2(self,alpha, job_id):
    #     np.random.seed(job_id)
    #     regret = []

    #     self.set_outcomes(job_id)

    #     N_bar = [0,1]
    #     M_bar = [0,1]
    #     e = np.eye(2)
        
    #     N = len(self.LossMatrix )
    #     n = np.zeros(N)
    #     v = [  np.zeros( len(set(i)) ) for i in self.FeedbackMatrix ] 
    #     mathcal_P = [ a for a in N_bar if geometry.isParetoOptimal(1, self.LossMatrix )] # set of pareto optimal actions
    #     mathcal_N = [ pair for pair in list( itertools.combinations([0,1], 2) ) if geometry.areNeighbours(pair[0], pair[1], self.LossMatrix ) ] #set of unordered neighboring actions
    #     # print(mathcal_N)
    #     #mathcal_N_plus = [ geometry.get_neighborhood_action_set(pair, N_bar, L) for pair in mathcal_N]  #neighborhood action set of pair 
    #     mathcal_N_plus = collections.defaultdict(dict)
    #     for pair in mathcal_N:
    #             mathcal_N_plus[ pair[0] ][ pair[1] ] = geometry.get_neighborhood_action_set(pair, N_bar, self.LossMatrix )

    #     observer_set = collections.defaultdict(dict)
    #     for pair in mathcal_N : 
    #             if geometry.ObservablePair(pair[0], pair[1], self.LossMatrix, [geometry.signal_vecs(i, self.FeedbackMatrix) for i in geometry.Neighbourhood(0, 1, self.LossMatrix )]):
    #                     observer_set [ pair[0] ][ pair[1] ] =   mathcal_N_plus[ pair[0] ][ pair[1] ] 
    #             else:
    #                     observer_set [ pair[0] ][ pair[1] ] = None
    #                     print('Observer set -- not implemented')

    #     observer_vector = collections.defaultdict(dict)
    #     for pair in mathcal_N :
    #             observer_vector[ pair[0] ][ pair[1] ] = geometry.get_observer_vector( pair ,self.LossMatrix ,self.FeedbackMatrix,observer_set) 

    #     W = self.W( mathcal_N, N_bar, observer_vector )

    #     # print('mathcal P', mathcal_P)
    #     # print('mathcal N', mathcal_N)

    #     for t in range(self.horizon):

    #         if t < N:  # initialisation
    #             action  = t
    #             outcome = self.outcomes[t]
    #             Y = geometry.signal_vecs(action, self.FeedbackMatrix) @ e[outcome]
    #             n[action] += 1
    #             v[action] += Y

    #             regret.append( self.gaps[action ]  )

    #         else: 
    #             break
                
    #     for t in range(self.horizon):
    #         outcome = self.outcomes[t]
    #         half_space = collections.defaultdict(dict)

    #         if t >= N:

    #             for pair in mathcal_N:
    #                 # print( 'inside', [  observer_vector[ pair[0] ][ pair[1] ][k].T * v[k]/n[k]   for k in mathcal_N_plus ] )
    #                 d_ij = sum( [  observer_vector[ pair[0] ][ pair[1] ][k].T @ v[k]/n[k]   for k in mathcal_N_plus ] )
    #                 c_ij = sum( [  np.fabs(  observer_vector[ pair[0] ][ pair[1] ][k] ).max()  * np.sqrt(alpha * np.log(t) / n[k] )    for k in mathcal_N_plus ] )
    #                 # print('d_ij',d_ij)
                    
    #                 if abs( d_ij ) >= c_ij:
    #                     half_space[ pair[0] ][ pair[1] ] = np.sign(d_ij)
    #                 else:
    #                     half_space[ pair[0] ][ pair[1] ] = 0

    #             #print('halfspace', half_space)

    #             #print('P before:',mathcal_P)
    #             mathcal_P = geometry.get_P_t(half_space, self.LossMatrix, mathcal_P, mathcal_N)
    #             #print('P after:',mathcal_P)

    #             #print('N before:',mathcal_N)
    #             mathcal_N = geometry.get_N_t(half_space, self.LossMatrix, mathcal_P, mathcal_N)
    #             #print('N after:',mathcal_N)

    #             #print()

    #             Q = reduce( np.union1d, [ mathcal_N_plus[ pair[0] ][ pair[1] ]  for pair in mathcal_N ]  )
    #             values = [ W[k]/n[k] for k in Q ]
    #             # print('values', values)
                
    #             action = np.argmax(values)
    #             Y = geometry.signal_vecs(action, self.FeedbackMatrix) @ e[outcome]
    #             n[action] += 1
    #             v[action] += Y

    #             regret.append( self.gaps[action ]  )

    #     return np.array(regret)


    n_cores = 16
horizon = 1000
n_folds = 15


LossMatrix = np.array( [ [1,1,0],[0,1,1],[1,0,1] ] )
FeedbackMatrix = np.array(  [ [1,0,0], [0,1,0],[0,0,1] ] )
LinkMatrix = np.linalg.inv( FeedbackMatrix ) @ LossMatrix

# task = SyntheticCase( LossMatrix, FeedbackMatrix , None, horizon) 
# result = np.cumsum(  eval_ucb1_parallel(task, n_cores, n_folds, horizon,'UCB1' ) ,1 )
# mean = np.mean(  result,0)
# std = np.std(  result,0)
# plt.plot( mean, label = 'UCB1' , color = 'purple' )
# plt.fill_between( range(horizon), mean - std / np.sqrt(n_folds), mean + std / np.sqrt(n_folds), alpha=0.2, color = 'purple') 

task = SyntheticCase(LossMatrix, FeedbackMatrix, LinkMatrix, horizon) 
result = np.cumsum( eval_feedexp_parallel(task, n_cores, n_folds, horizon,'Bianchi' ) , 1 )
mean =   np.mean(  result , 0 )
std = np.std( result , 0)
plt.plot( mean, label = 'Bianchi', color = 'green' )
plt.fill_between( range(horizon), mean -  std / np.sqrt(n_folds), mean +  std / np.sqrt(n_folds), alpha=0.2, color = 'green') 

plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.ylim( (0, horizon/2) )
plt.xlim( (0, horizon) )
plt.legend()
plt.title('{}'.format('3actions 3 outcomes') )
plt.show()
plt.clf() 

# LossMatrix = np.array( [ [1,1,0],[0,1,1],[1,0,1] ] )
# FeedbackMatrix = np.array(  [ [1,0,0], [0,1,0],[0,0,1] ] )
# LinkMatrix = np.linalg.inv( FeedbackMatrix ) @ LossMatrix

# for L_FeedbackMatrix, R_FeedbackMatrix, LinkMatrix, title in zip( Loss_FeedbackMatrix, Reward_FeedbackMatrix, LinkMatrices,['Apple Tasting', 'Bandit']):

    # task = SyntheticCase( LossMatrix, L_FeedbackMatrix , None, horizon) 
    # result = np.cumsum(  eval_ucb1_parallel(task, n_cores, n_folds, horizon,'UCB1' ) ,1 )
    # mean = np.mean(  result,0)
    # std = np.std(  result,0)
    # plt.plot( mean, label = 'UCB1' , color = 'purple' )
    # plt.fill_between( range(horizon), mean - std / np.sqrt(n_folds), mean + std / np.sqrt(n_folds), alpha=0.2, color = 'purple') 

    # task = SyntheticCase(LossMatrix, L_FeedbackMatrix, LinkMatrix, horizon) 
    # result = np.cumsum( eval_feedexp_parallel(task, n_cores, n_folds, horizon,'Bianchi' ) , 1 )
    # mean =   np.mean(  result , 0 )
    # std = np.std( result , 0)
    # plt.plot( mean, label = 'Bianchi', color = 'green' )
    # plt.fill_between( range(horizon), mean -  std / np.sqrt(n_folds), mean +  std / np.sqrt(n_folds), alpha=0.2, color = 'green') 

    # plt.xlabel('Iteration')
    # plt.ylabel('Cumulative Regret')
    # plt.ylim( (0, horizon/2) )
    # plt.xlim( (0, horizon) )
    # plt.legend()
    # plt.title('{}'.format(title) )
    # plt.show()
    # plt.clf() 
    # plt.savefig('baselines_ap.pdf', bbox_inches='tight')
    

n_cores = 16
horizon = 10000
n_folds = 25

LossMatrix = np.array( [ [0, 0], [1, -1] ] )
FeedbackMatrix =  np.array([ [0, 0],[1, -1] ])
LinkMatrix = np.identity( len(LossMatrix) )

task = SyntheticCase(LossMatrix, FeedbackMatrix, horizon) 
# result = np.cumsum( task.feedexp3( True , 10)  )
# plt.plot(result)

result =  eval_feedexp_parallel(task, n_cores, n_folds, horizon, True, 'Piccolboni' ) 
plt.plot(  np.mean( result , 0 ) , label = 'Piccolboni', color = 'green' )

result =  eval_feedexp_parallel(task, n_cores, n_folds, horizon, True, 'Bianchi' ) 
plt.plot(  np.mean( result , 0 ) , label = 'Bianchi', color = 'orange' )

plt.legend()

n_cores = 1
horizon = 2000
n_folds = 25



task = SyntheticCase(LossMatrix, FeedbackMatrix, horizon) 

# task.feedexp3(False,'Piccolboni',1)

result =  eval_feedexp_parallel(task, n_cores, n_folds, horizon, False, 'Piccolboni' ) 
plt.plot(   np.mean( result , 0 )  , label = 'Piccolboni', color = 'green' )

result =  eval_feedexp_parallel(task, n_cores, n_folds, horizon, False, 'Bianchi' ) 
plt.plot(   np.mean( result , 0 )  , label = 'Bianchi', color = 'orange' )

plt.legend()