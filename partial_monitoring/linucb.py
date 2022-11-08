import numpy as np

class LinUCB():

    def __init__(self, game, horizon, alpha):

        self.game = game
        self.horizon = horizon
        self.alpha = alpha
        self.N = game.n_actions

    def calc_UCB(self, arm, X, t):
        
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.As[arm])
        
        # Perform ridge regression to obtain estimate of covariate coefficients theta theta is (d x 1) dimension vector
        theta = np.dot(A_inv, self.bs[arm])
        
        # Find ucb based on p formulation (mean + std_dev) 
        # p is (1 x 1) dimension vector
        ucb = np.dot(theta.T,X) +  self.alpha * np.sqrt(np.dot( X.T, np.dot(A_inv,X)))
        
        return ucb

    def get_action(self, t, X):

        if(t < self.N ):
            self.d = len(X)
            self.As = [ np.identity(self.d) for _ in range(self.N) ]
            self.bs = [ np.zeros( (self.d,1) ) for _ in range(self.N) ] 
            action = t
            
        else:
            ucbs = [ self.calc_UCB(arm, X, t)  for arm in range(self.N) ] 
            action = np.argmax(ucbs)
        
        return action

    def update(self, action, feedback, bandit_feedback, outcome, t, X):

        X = X.reshape([-1,1])
    
        # Update A which is (d * d) matrix.
        self.As[action] += np.dot(X, X.T)
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.bs[action] += bandit_feedback * X

    def reset(self,):
        self.As = [ np.identity(self.d) for _ in range(self.N) ]
        self.bs = [ np.zeros( (self.d,1) ) for _ in range(self.N) ] 