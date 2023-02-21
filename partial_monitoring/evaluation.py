import numpy as np


# def evaluate_parallel(nbCores, n_folds, horizon, alg, game, type):
#     print("nbCores:", nbCores, "nbFolds:", n_folds, "Horizon:", horizon)
#     pool = Pool(processes = nbCores) 
#     task = Evaluation(horizon, type)

#     np.random.seed(1)
#     distributions = []

#     for jobid in range(n_folds):
        
#         if type == 'easy' :
#             p = np.random.uniform(0, 0.2) 
#         else:
#             p = np.random.uniform(0.4,0.5)
#         distributions.append( [p, 1-p] )

#     return np.asarray(  pool.map( partial( task.eval_policy_once, alg, game ), zip(distributions ,range(n_folds)) ) ) 



class Evaluation:

    def __init__(self, horizon, ):
        self.horizon = horizon

    def get_outcomes(self, game):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        alg.reset()

        distribution, jobid = job

        np.random.seed(jobid)

        outcome_distribution =  {'spam':distribution[0],'ham':distribution[1]}

        game.set_outcome_distribution( outcome_distribution, jobid )
        outcomes = self.get_outcomes(game)

        action_counter = np.zeros( (game.n_actions, self.horizon) )        

        for t in range(self.horizon):

            # policy chooses one action
            action = alg.get_action(t, None)

            # Environment chooses one outcome
            outcome = outcomes[t]

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, None, t)

            for i in range(game.n_actions):
                if i == action:
                    action_counter[i][t] = action_counter[i][t-1] +1
                else:
                    action_counter[i][t] = action_counter[i][t-1]

        regret = np.array( [ game.delta(i) for i in range(game.n_actions) ] ).T @ action_counter

        return regret
