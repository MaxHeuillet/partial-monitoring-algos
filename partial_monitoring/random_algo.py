
from re import L
import numpy as np


class Random():

    def __init__(self, game, horizon):

        self.game = game
        self.horizon = horizon

    def get_action(self, t):
        
        pbt = np.ones( self.game.n_actions ) / self.game.n_actions
        action = np.random.choice(self.game.n_actions, 1,  p = pbt )[0]
        return action

    def update(self, action, feedback, outcome):
        pass

    def reset(self,):
        pass
