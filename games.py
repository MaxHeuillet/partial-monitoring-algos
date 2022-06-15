from math import log, exp, pow
import numpy as np
import geometry




class Game():
    
    def __init__(self, LossMatrix, FeedbackMatrix, LinkMatrix, SignalMatrices, outcome_distribution ):
        self.LossMatrix = LossMatrix
        self.FeedbackMatrix = FeedbackMatrix
        self.LinkMatrix = LinkMatrix
        self.SignalMatrices = SignalMatrices
        self.n_actions = len(self.LossMatrix)
        self.n_outcomes = len(self.LossMatrix[0])
        # self.N = len(self.LossMatrix)
        # self.M = len(self.LossMatrix[0])
        # self.Actions_dict = { a : "{0}".format(a) for a in range(self.N)} # Actions semantic
        # self.Outcomes_dict = { a : "{0}".format(a) for a in range(self.M)} # Outcomes semantic

        self.outcome_dist = outcome_distribution
        self.i_star = self.optimal_action(  )

    def optimal_action(self, ):
        deltas = []
        for i in range(len(self.LossMatrix)):
            deltas.append( self.LossMatrix[i,...].T @ list( self.outcome_dist.values() ) )
        return np.argmin(deltas)

    def delta(self, action):
        return ( self.LossMatrix[action,...] - self.LossMatrix[self.i_star,...] ).T @ list( self.outcome_dist.values() ) 


def apple_tasting( restructure_game, outcome_distribution ):
    init_LossMatrix = np.array( [ [1, 0], [0, 1] ] )
    init_FeedbackMatrix =  np.array([ [1, 1],[1, 0] ])
    signal_matrices =  [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ] ) ]

    if restructure_game:
        FeedbackMatrix, LossMatrix = general_algorithm( init_FeedbackMatrix, init_LossMatrix )
    else:
        FeedbackMatrix, LossMatrix = init_FeedbackMatrix, init_LossMatrix


    # if (FeedbackMatrix == LossMatrix).all():
    #     LinkMatrix = np.identity( len(init_LossMatrix[1] ) )
    # else:
    
    LinkMatrix = np.linalg.inv( init_FeedbackMatrix ) @ LossMatrix 

    game = Game( LossMatrix, FeedbackMatrix, LinkMatrix, signal_matrices, outcome_distribution )

    return game

def bandit( restructure_game, outcome_distribution ):
    init_LossMatrix = np.array( [ [0, 0], [-1, 1] ] )
    init_FeedbackMatrix =  np.array([ [0, 0],[-1, 1] ])
    signal_matrices = [  np.array( [ [1,1] ] ), np.array( [ [0,1],[1,0] ] ) ] 

    if restructure_game:
        FeedbackMatrix, LossMatrix = general_algorithm( init_FeedbackMatrix, init_LossMatrix )
    else:
        FeedbackMatrix, LossMatrix = init_FeedbackMatrix, init_LossMatrix

    if (FeedbackMatrix == LossMatrix).all():
        LinkMatrix = np.identity( len(init_LossMatrix[1] ) )
    else:
        LinkMatrix = np.linalg.lstsq(FeedbackMatrix.transpose(), LossMatrix.transpose(), rcond=None )[0].transpose()

    game = Game( LossMatrix, FeedbackMatrix, LinkMatrix,signal_matrices, outcome_distribution )

    return game


def label_efficient( outcome_distribution ):
    LossMatrix = np.array( [ [1, 1],[1, 0],[0, 1] ] )
    FeedbackMatrix = np.array(  [ [1, 1/2], [1/4, 1/4], [1/4, 1/4] ] )
    LinkMatrix = np.array( [ [0, 2, 2],[2, -2, -2],[-2, 4, 4] ] )
    signal_matrices = [ np.array( [ [0,1],[1,0] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 
    return Game( LossMatrix, FeedbackMatrix, LinkMatrix, signal_matrices, outcome_distribution )


def general_algorithm(F, L):
    # print(F)

    N, M = F.shape

    Fdash_list = []
    Ldash_list = []
    # We use the lists sizes for z in paper
    h = {} # pseudo-action to action map
    s = {} # pseudo-action to symbol map

    for j in range(N):

        for v in set(F[j,...]):
                
            Fiv = geometry.signal_vec( j, v, F )

            if not Fdash_list or not geometry.is_linear_comb(Fiv, Fdash_list):
                h[len(Fdash_list)] = j # h(z)=i in FeedExp3 paper
                s[len(Fdash_list)] = v # not in FeedExp3 paper ??
                Fdash_list.append(Fiv)
                Ldash_list.append(L[j,...])
                bool_fiv_added = True

        if not bool_fiv_added:
            h[len(Fdash_list)] = j # h(z)=j in the paper
            s[len(Fdash_list)] = v # not in FeedExp3 paper ??
            Fdash_list.append( np.zeros(M) )
            Ldash_list.append(L[j,...])

    # Build F' and H' matrices
    FdashMatrix = np.vstack(Fdash_list)
    LdashMatrix = np.vstack(Ldash_list)

    Ndash, Mdash = FdashMatrix.shape
    assert FdashMatrix.shape == LdashMatrix.shape # just in case

    # Search for strictly-dominating pseudo-actions 
    NonEmptyCells = []
    EmptyCells = []
    for iv in range(Ndash):
        if geometry.isStrictlyNonDominated(iv, LdashMatrix):
            NonEmptyCells.append(iv)
        else:
            EmptyCells.append(iv)
            
    if len(NonEmptyCells)>0: # An empty nonEmptyCells is a problem!
        # Pick one non-dominated action
        b = np.random.choice(NonEmptyCells)      # Choose any action from the set of actions with nonempty cells
    else:
        print("WARNING: no strictly dominant cell found")
        #b = random.choice(range(Ndash))  # Choose any action
        b = 0

    # Translate the loss relatively to pseudo-action b. Recall that loss transposition does not impact the policy regret.
    LdashMatrix = LdashMatrix - LdashMatrix[b,...]
    
    # Makes the dominated actions as bad as possible i.e. with worst possible loss
    for iv in EmptyCells:
        if not geometry.isNonDominated(iv, LdashMatrix):
            LdashMatrix[iv,...] =  max(LdashMatrix[iv,...])
        
    return FdashMatrix, LdashMatrix