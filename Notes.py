import gurobipy as gp
from gurobipy import GRB


env = gp.Env( empty = True )
m = gp.Model("mip1")
m.addVar(vtype=GRB.BINARY, name="x")
m.setObjective(x + y + 2 * z, GRB.MINIMIZE)
m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
m.optimize()


FeedbackMatrix =  np.array([ [1, 1],[1, 0] ])

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


def two_cell_intersection(pair, LossMatrix, halfspace, mathcal_N, mathcal_P):

    N, M = LossMatrix.shape
    # declare M ppl Variables
    p = [ ppl.Variable(j) for j in range(M) ]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()

    # p belongs to $\Delta_M$ the set of M dimensional probability vectors
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)

    # strict Loss domination constraints for both a and b
    Doma = scale_to_integers(domination_matrix( pair[0],LossMatrix))
    Domb = scale_to_integers(domination_matrix(pair[1],LossMatrix)) 
    for i in range(N):
        if i!=pair[0]:
            # p is such that for any action i Loss[a,...]*p <= Loss[a,...]*p
            cs.insert( sum( (Doma[i,j]*p[j] for j in range(M)) ) <= 0 )
        if i!=pair[1]:
            # p is such that for any action i Loss[b,...]*p <= Loss[a,...]*p
            cs.insert( sum( (Domb[i,j]*p[j] for j in range(M)) ) <= 0 )

    # intersection from the halfspaces:
    for pair in get_halfspace_pairs(halfspace):
        substract = LossMatrix[ pair[0] ] - LossMatrix[ pair[1] ]  
        cs.insert(  halfspace[  pair[0] ][ pair[1] ] * sum( ( substract[a] * p[a] for a in range(M) ) )  > 0 )

    return ppl.NNC_Polyhedron(cs)


# def observer_vector(L, H, i, j, mathcal_K_plus):
#     A = np.vstack( global_signal(H) )
#     Lij = L[i,...] - L[j,...]
#     # print('Lij', Lij)
#     # print('globalsignal',global_signal(H))
#     x, res, rank, s = np.linalg.lstsq(A.T, Lij) 
#     lenght = [ len( np.unique(H[k]) ) for k in mathcal_K_plus]
#     x = iter( np.round(x) )
#     return [ np.array( list(islice( x, i)) ) for i in lenght] 

def get_P_t(halfspace, L, mathcal_P, mathcal_N):
    P_t  = []
    for pair in mathcal_P:
        result = single_cell_intersection(pair, L, halfspace, mathcal_N, mathcal_P)
        #print(result)
        if result.is_empty() == False:
            P_t.append(pair)
    return P_t

def get_N_t(halfspace, L, mathcal_P, mathcal_N):
    N_t  = []
    for i in mathcal_N:
        result = two_cell_intersection(i, L, halfspace, mathcal_N, mathcal_P)
        #print(result)
        if result.is_empty() == False:
            N_t.append(i)
    return N_t

def get_halfspace_pairs(dictionary):
    result = []
    for e1 in dictionary.keys():
        for e2 in dictionary[e1].keys():
            result.append( [e1,e2] )
    return result

def single_cell_intersection(i, LossMatrix, halfspace, mathcal_N, mathcal_P):

    N, M = LossMatrix.shape
    
    p = [ ppl.Variable(j) for j in range(M) ] # declare M ppl Variables
    cs = ppl.Constraint_System() # declare polytope constraints

    # p belongs to $\Delta_M$ the set of M dimensional probability vectors
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)

    # strict Loss domination constraints
    Dom = scale_to_integers(domination_matrix(i,LossMatrix))
    
    for a in range(N):
        if a != i:
            # p is such that for any action a Loss[i,...]*p <= Loss[a,...]*p
            #print "Domination line:", Dom[a,...], "inequality:", sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0
            cs.insert( sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0 )
    
    # intersection from the halfspaces:
    for pair in get_halfspace_pairs(halfspace):
        substract = LossMatrix[ pair[0] ] - LossMatrix[ pair[1] ]  
        cs.insert(  halfspace[  pair[0] ][ pair[1] ] * sum( ( substract[a] * p[a] for a in range(M) ) )  > 0 )

    return ppl.NNC_Polyhedron(cs)
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