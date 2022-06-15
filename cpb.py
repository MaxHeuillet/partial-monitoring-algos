import numpy as np
import geometry_v3
import geometry

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
        self.neighborhood_actions = [ [0, 1], [1, 0] ]#geometry_v3.getNeighborhoodActions(game.LossMatrix, self.N, self.M, [])
        # print('mathcal_N', self.neighborhood_actions)

        self.N_plus =  geometry.get_neighborhood_action_set(self.N, game.LossMatrix)
        # print('N_plus', self.N_plus)

        self.v = geometry_v3.getV(game.LossMatrix, self.N, self.M, self.A, self.SignalMatrices, self.neighborhood_actions)

        self.W = geometry_v3.getConfidenceWidth(self.neighborhood_actions,self.N_plus, self.v, self.N)
        # print('W', self.W)

        self.alpha = 1.01

        self.eta =  self.W **2/3 
 
    def get_action(self, t):

        if(t<self.N):

            action = t

        else:

            halfspace = []

            for pair in self.neighborhood_actions:
                tdelta = 0
                c = 0
                for k in  self.N_plus[ pair[0] ][ pair[1] ] :
                    # print( 'pair', pair, self.v[ pair[0] ][ pair[1] ][k].shape, self.nu[k].shape )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k] @ self.nu[k]  / self.n[k] 
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * np.sqrt( self.alpha * np.log(t) / self.n[k]  )
                # print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta)[0][0] ) )
                # else:
                #     halfspace.append( ( pair, 0 ) )
                # print('pair', pair,  'tdelta', tdelta, 'c', c)

            # print('halfspace', halfspace)
            P_t = geometry_v3.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
            N_t = geometry_v3.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.neighborhood_actions )

            Nplus_t = []
            for pair in N_t:
                Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )
            R_t = []
            for k in range(self.N):
              if self.n[k] <=  self.eta[k] * geometry_v3.f(t, self.alpha) :
                R_t.append(k)
            union1= np.union1d( Nplus_t, P_t )
            union1 = np.array(union1, dtype=int)
            # print('union1', union1)
            S =  np.union1d(  union1  ,  R_t)
            S = np.array( S, dtype = int)
            # print('S', S)
            S = np.unique(S)
            # print('outcome frequency', self.nu, 'action frequency', self.n )
            # print('P_t',P_t,'N_t', N_t,'Nplus_t',Nplus_t,'R_t',R_t,'S',S)

            values = { i:self.W[i]**2/self.n[i] for i in S}
            # print('value', value)
            action = max(values, key=values.get)

        return action

    def update(self, action, feedback, outcome):
        self.n[action] += 1
        Y_t = np.array([ self.game.SignalMatrices[action] @ np.eye(self.M)[outcome] ] )
        # print('Y_t', Y_t, 'shape', Y_t.shape, 'nu[action]', self.nu[action], 'shape', self.nu[action].shape)
        self.nu[action] += Y_t.T
