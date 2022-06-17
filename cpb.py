import numpy as np
import geometry_v3
import geometry
import collections

class CPB():

    def __init__(self, game, horizon):

        self.game = game
        self.horizon = horizon

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
        # print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

        self.SignalMatrices = game.SignalMatrices

        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)] 
        # print('nu', self.nu)

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N #geometry_v3.getNeighborhoodActions(game.LossMatrix, self.N, self.M, [])
        # print('mathcal_N', self.mathcal_N)

        # self.N_plus =  geometry.get_neighborhood_action_set(self.N, game.LossMatrix)
        self.N_plus =  collections.defaultdict(dict)
        # self.N_plus[0][1] = [ 0, 1, 2 ]
        # self.N_plus[0][2] = [ 0, 1, 2 ]
        # self.N_plus[1][0] = [ 0, 1, 2 ]
        # self.N_plus[2][0] = [ 0, 1, 2 ]
        self.N_plus[2][1] = [ 1, 2 ]
        self.N_plus[1][2] = [ 1, 2 ]
        # print('N_plus', self.N_plus)

        self.V = collections.defaultdict(dict)
        # self.V[0][1] = self.N_plus[0][1]
        # self.V[0][2] = self.N_plus[0][2]
        # self.V[1][0] = self.N_plus[1][0]
        # self.V[2][0] = self.N_plus[2][0]
        self.V[2][1] = [ 0, 1, 2 ]
        self.V[1][2] = [ 0, 1, 2 ]

        self.v = {1: {2: [ np.array([-1.,  1.]), np.array([0]), np.array([0])]}, 2: {1: [np.array([ 1., -1.]), np.array([0.]), np.array([0.])]}} #geometry_v3.getV(game.LossMatrix, 3, 2, game.FeedbackMatrix, game.SignalMatrices, game.mathcal_N, self.V)# game.v #geometry_v3.getV(game.LossMatrix, self.N, self.M, self.A, self.SignalMatrices, self.neighborhood_actions)
        # print('observer vectors', self.v)


        self.W = geometry_v3.getConfidenceWidth(self.mathcal_N, self.V, self.v, self.N)
        # print('W', self.W)

        self.alpha = 1.01

        self.eta =  self.W **2/3 
 
    def get_action(self, t):

        if(t<self.N):

            action = t

        else:

            halfspace = []

            for pair in self.mathcal_N:
                tdelta = 0
                c = 0
                # print('pair', pair, 'N_plus', self.N_plus[ pair[0] ][ pair[1] ] )
                for k in  self.V[ pair[0] ][ pair[1] ] :
                    # print( 'proba', self.nu[k]  / self.n[k]  )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k] @ self.nu[k]  / self.n[k] 
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * np.sqrt( self.alpha * np.log(t) / self.n[k]  )
                # print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta)[0] ) ) #[0]
                # else:
                #     halfspace.append( ( pair, 0 ) )
                

            # print('halfspace', halfspace)
            P_t = geometry_v3.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
            N_t = geometry_v3.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.mathcal_N )

            Nplus_t = []
            for pair in N_t:
                Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )

            R_t = []
            for k in range(self.N):
              if self.n[k] <=  self.eta[k] * geometry_v3.f(t, self.alpha) :
                R_t.append(k)

            V_t = []
            for pair in N_t:
                V_t.extend( self.V[ pair[0] ][ pair[1] ] )

            intersect = np.intersect1d(V_t, R_t)

            union1= np.union1d(  P_t, Nplus_t )
            union1 = np.array(union1, dtype=int)
            # print('union1', union1)
            S =  np.union1d(  union1  , intersect ) #intersect
            S = np.array( S, dtype = int)
            # print('S', S)
            S = np.unique(S)
            # print('outcome frequency', self.nu, 'action frequency', self.n )
            

            values = { i:self.W[i]**2/self.n[i] for i in S}
            # print('value', values)
            action = max(values, key=values.get)
            # print('P_t',P_t,'N_t', N_t,'Nplus_t',Nplus_t,'V_t',V_t, 'R_t',R_t, 'S',S,'values', values, 'action', action)
            # print('n', self.n,'nu', self.nu)

        return action

    def update(self, action, feedback, outcome):
        self.n[action] += 1
        Y_t = np.array([ self.game.SignalMatrices[action] @ np.eye(self.M)[outcome] ] )
        # print('Y_t', Y_t, 'shape', Y_t.shape, 'nu[action]', self.nu[action], 'shape', self.nu[action].shape)
        self.nu[action] += Y_t.T
