
import numpy as np
from scipy.special import logsumexp

class FeedExp3():

    def __init__(self, game, horizon):

        self.game = game
        self.horizon = horizon
        self.L_tilde = np.zeros(game.n_actions, dtype = np.float128 )
        self.k_star = max( 1,  np.fabs(game.LinkMatrix).max() )
        # print('kstar', self.k_star)

    def get_action(self, t):

        self.eta, self.gamma = self.parameters_Bianchi(t+1) 
        
        LogZ = logsumexp( - self.eta * self.L_tilde)
        Q = np.exp( -self.eta * self.L_tilde - LogZ ) #
        Q = Q / Q.sum()
        Q = Q.astype(np.float)
        # print('logZ', LogZ, 'Q', Q)
        self.pbt  =  (1 - self.gamma) * Q  + self.gamma / self.game.n_actions
        # print('pbt', self.pbt, 'gamma', self.gamma, 'eta', self.eta, 'cumulatif loss',  self.L_tilde)

        action = np.random.choice(self.game.n_actions, 1,  p = self.pbt )[0]
        return action

    def update(self, action, feedback, outcome):
        # print('action', action, 'feedback', feedback)

        for i in range( self.game.n_actions ):
            l =  ( self.game.LinkMatrix[i][action] * feedback ) / self.pbt[action]
            self.L_tilde[i] += self.L_tilde[i] + l


    def parameters_Bianchi(self, t):
        # [Bianchi et al. 2006 "Regret minimization under partial monitoring"]
        eta = (self.k_star)**(2/3) * ( np.log(self.game.n_actions)/self.game.n_actions )**(2/3) * t**(-2/3)  #1 / C * pow( np.log( self.game.n_actions ) / ( self.game.n_actions * t ) , 2./3.) 
        gamma = min(1, (self.k_star)**(1/3) * self.game.n_actions**(2/3) * t**(-1/3) )  #min(1,  C * pow( ( np.log( self.game.n_actions ) * self.game.n_actions **2) / t , 1./3.) )
        return eta, gamma 


            
    # def feedexp3(self, general_algo, method, job_id):

    #     action_counter = np.zeros(self.n_actions)
    #     outcome_counter = np.zeros(self.n_outcomes)
        
        
    #     if general_algo:
    #         LossMatrix, FeedbackMatrix = self.general_algorithm( self.FeedbackMatrix, self.LossMatrix )
    #     else:
    #         LossMatrix, FeedbackMatrix = self.FeedbackMatrix, self.LossMatrix 

    #     self.set_outcomes(LossMatrix, job_id)

    #     k_star = max( 1, np.fabs(LinkMatrix).max() )
    #     C = pow( k_star * np.sqrt(np.exp(1.) - 2.), 2./3.)
    #     if method == 'Piccolboni':
    #         eta, gamma = self.parameters_Piccolboni()

    #     u = np.ones(self.n_actions)/self.n_actions
    #     pbt = np.ones(self.n_actions)/self.n_actions

    #     for t in range(self.horizon):

    #         if method == 'Bianchi':
    #             eta, gamma = self.parameters_Bianchi(C, t+1) 
            
    #         pbt_hat =  (1 - gamma) * pbt  + gamma * u 

    #         action = np.random.choice(self.n_actions,1,  p = pbt )[0]

    #         feedback =  self.get_feedback(FeedbackMatrix, action, self.outcomes[t] )

    #         x = np.array( [ feedback * LinkMatrix[i][action] / pbt_hat[i] for i in range(self.n_actions) ] )

    #         Z = sum( pbt / np.exp( eta * x ) )

    #         pbt = pbt / ( Z * np.exp( eta * x  ) )

    #         cumAllLosses += LossMatrix[..., self.outcomes[t] ]
    #         cumSufferedLoss += LossMatrix[action, self.outcomes[t] ]
    #         cumRegret[t] = cumSufferedLoss - min(cumAllLosses)

    #         action_counter[action] += 1
    #         outcome_counter[ self.outcomes[t] ] += 1

    #     return np.array(cumRegret)

