import numpy as np
import geometry_v3

from sklearn.linear_model import LogisticRegression

class CPB_side():

    def __init__(self, game, horizon,):

        self.game = game
        self.horizon = horizon

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
        # print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

        self.SignalMatrices = game.SignalMatrices
        # print('signalmatrices', self.SignalMatrices)

        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)] 
        # print('nu', self.nu)

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        # print('mathcal_N', self.mathcal_N)

        self.N_plus =  game.N_plus

        self.V = game.V

        self.v = game.v 

        self.W = geometry_v3.getConfidenceWidth(self.mathcal_N, self.V, self.v, self.N)
        # print('weights', self.W)
        self.alpha = 1.01

        self.eta =  self.W **2/3 

        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':[], 'labels':[],'weights':None} )
 
    def get_action(self, t, X):

        if(t<self.N):

            action = t

        else:

            halfspace = []
            q = []
            w = []
            
            # X = np.atleast_2d(X)
            for i in range(self.N):
                q.append( self.contexts[i]['weights'] @ X  )
                X_it =  np.array( self.contexts[i]['features'] )

                n,D = X_it.shape
                X_it = X_it.reshape( (D, n) )
                D = 784
                lambda_i = 1
                formule = D * (  np.sqrt( D * np.log(t+1) ) + len(self.SignalMatrices[i]) ) * np.sqrt( X.T @ np.linalg.inv( lambda_i * np.identity(D) + X_it @ X_it.T  ) @ X )
                w.append( formule  )

            print( 'q   ', q )
            print('conf   ', w )

            for pair in self.mathcal_N:
                tdelta = 0
                c = 0
                # print('pair', pair, 'N_plus', self.N_plus[ pair[0] ][ pair[1] ] )
                for k in  self.V[ pair[0] ][ pair[1] ]:
                    # print( 'pair ', pair, 'action ', k, 'proba ', self.nu[k]  / self.n[k]  )
                    # print('k', k, 'pair ', pair, 'v ', self.v[ pair[0] ][ pair[1] ][k].T.shape , 'q[k] ', q[k].shape  )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k]
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta)[0] ) ) #[0]
                # else:
                #     halfspace.append( ( pair, 0 ) )
                

            # print('halfspace', halfspace)
            P_t = self.pareto_halfspace_memory(halfspace)
            N_t = self.neighborhood_halfspace_memory(halfspace)

            Nplus_t = []
            for pair in N_t:
                Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )
            Nplus_t = np.unique(Nplus_t)

            R_t = []
            for k in range(self.N):
              if self.n[k] <=  self.eta[k] * geometry_v3.f(t, self.alpha) :
                R_t.append(k)

            V_t = []
            for pair in N_t:
                V_t.extend( self.V[ pair[0] ][ pair[1] ] )
            V_t = np.unique(V_t)

            intersect = np.intersect1d(V_t, R_t)

            union1= np.union1d(  P_t, Nplus_t )
            union1 = np.array(union1, dtype=int)
            # print('union1', union1)
            S =  np.union1d(  union1  , intersect )
            S = np.array( S, dtype = int)
            # print('S', S)
            S = np.unique(S)
            # print('outcome frequency', self.nu, 'action frequency', self.n )
            
            values = { i:self.W[i]*w[i] for i in S}
            # print('value', values)
            action = max(values, key=values.get)
            # print('P_t',P_t,'N_t', N_t,'Nplus_t',Nplus_t,'V_t',V_t, 'R_t',R_t, 'S',S,'values', values, 'action', action)
            # print('n', self.n,'nu', self.nu)

        return action

    def update(self, action, feedback, outcome, X):

        self.n[action] += 1
        e_y = np.zeros( (self.M, 1) )
        e_y[outcome] = 1
        Y_t =  self.game.SignalMatrices[action] @ e_y 
        
        # print('e_y', e_y)
        self.contexts[action]['labels'].append( Y_t )
        self.contexts[action]['features'].append( X ) 
        X_it =  np.array( self.contexts[action]['features'] )
        n,D = X_it.shape
        sigma, _ = Y_t.shape
        Y_it =  np.array( self.contexts[action]['labels'] ).reshape( (sigma, 1, n) )
        print('Y_it', Y_it.shape )
        X_it = X_it.reshape( (D, n) )
        lambda_i = 1
        
        # print('Y_it', Y_it.shape )
        # print('X_it', X_it.shape )
        self.contexts[action]['weights'] = Y_it @ X_it.T @ np.linalg.inv( lambda_i * np.identity( D ) + X_it @ X_it.T )
        # print( 'weigts',  Y_it @ X_it.T @ np.linalg.inv( lambda_i * np.identity( D ) + X_it @ X_it.T ) )
        # print('action', action, 'Y_t', Y_t, 'shape', Y_t.shape, 'nu[action]', self.nu[action], 'shape', self.nu[action].shape)
        self.nu[action] += Y_t

        

    def halfspace_code(self, halfspace):
        string = ''
        for element in halfspace:
            pair, sign = element
            string += '{}{}{}'.format(pair[0],pair[1], sign)
        return string 


    def pareto_halfspace_memory(self,halfspace):

        code = self.halfspace_code(  sorted( halfspace) )
        known = False
        for mem in self.memory_pareto.keys():
            if code  == mem:
                known = True

        if known:
            result = self.memory_pareto[ code ]
        else:
            result =  geometry_v3.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
            self.memory_pareto[code ] =result
 
        return result

    def neighborhood_halfspace_memory(self,halfspace):

        code = self.halfspace_code(  sorted( halfspace) )
        known = False
        for mem in self.memory_neighbors.keys():
            if code  == mem:
                known = True

        if known:
            result = self.memory_neighbors[ code ]
        else:
            result =  geometry_v3.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.mathcal_N )
            self.memory_neighbors[code ] =result
 
        return result
