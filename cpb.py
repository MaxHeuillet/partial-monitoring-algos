import numpy as np
import geometry_v3

class CPB():

    def __init__(self, game, horizon):

        self.game = game
        self.horizon = horizon

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
        print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

        self.SignalMatrices = geometry_v3.calculate_signal_matrices(game.FeedbackMatrix, self.N, self.M, self.A)

        self.n = np.zeros(self.N)
        self.nu = [ np.zeros( self.A ) for i in range(self.N)] 

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.neighborhood_actions = geometry_v3.getNeighborhoodActions(game.LossMatrix, self.N, self.M, [])

        self.v = geometry_v3.getV(game.LossMatrix, self.N, self.M, self.A, self.SignalMatrices, self.neighborhood_actions)

        self.W = geometry_v3.getConfidenceWidth(self.neighborhood_actions, self.v, self.N)

        self.alpha = 1.01

        self.eta = []
        for i in range(self.N):
            self.eta.append( self.W[i]**2/3 )
 
    def get_action(self, t):

        if(t<self.N):

            action = t

        else:

            halfspace = []

            for pair in self.neighborhood_actions:
                tdelta = 0
                c = 0
                for k in range(self.N):
                    tdelta += self.v[pair[0]][pair[1]][k].dot( self.nu[k] ) / self.n[k]
                    c += np.linalg.norm( self.v[pair[0]][pair[1]][k], np.inf ) * np.sqrt( self.alpha * np.log(t) / self.n[k]  )
                
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta) ) )
                # else:
                #     halfspace.append( ( pair, 0 ) )
                # print('pair', pair,  'tdelta', tdelta, 'c', c)

            # print('halfspace', halfspace)
            P_t = geometry_v3.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
            N_t = geometry_v3.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace)

            Nplus_t = []

            for pair in N_t:
              Nplus_t.append(pair[0])
              Nplus_t.append(pair[1])
              
            R_t = []
            for k in range(self.N):
              if self.n[k] <=  self.eta[k] * geometry_v3.f(t, self.alpha) :
                R_t.append(k)
            R_t = np.unique(R_t)

            union1= np.union1d( Nplus_t, P_t )
            union1 = np.array(union1, dtype=int)
            # print('union1', union1)
            S =  np.union1d(  union1  ,  R_t)
            S = np.array( S, dtype = int)
            # print('S', S)
            S = np.unique(S)
              
            value = [ self.W[i]**2/self.n[i] for i in S]
            # print('value', value)
            action = np.argmax(value)

        return action

    def update(self, action, feedback):
        self.n[action] += 1
        self.nu[action][ feedback ] += 1
