from sklearn.preprocessing import PolynomialFeatures
import numpy as np



# def evaluate_parallel(nbCores, n_folds, horizon, alg, game, type, context_type):
#     print("nbCores:", nbCores, "nbFolds:", n_folds, "Horizon:", horizon)
#     pool = Pool(processes = nbCores) 
#     task = Evaluation(horizon, type)

#     np.random.seed(1)
#     distributions = []
#     context_generators = []

#     for jobid in range(n_folds):
        
#         p = np.random.uniform(0, 0.2) if type == 'easy' else np.random.uniform(0.4,0.5)
#         distributions.append( [p, 1-p] )

#         contexts = synthetic_data.QuinticContexts( 2, 0.01)
#         context_generators.append( contexts )

#         # d = 2
#         # margin =0.01
#         # contexts = synthetic_data.LinearContexts( np.array([0.5,0.5]), 0, d, margin) #synthetic_data.ToyContexts( )

#     return np.asarray(  pool.map( partial( task.eval_policy_once, alg, game ), zip(distributions , context_generators ,range(n_folds)) ) ) 



class Evaluation_contextual:

    def __init__(self, horizon, ):
        self.horizon = horizon

    def get_outcomes(self, game, ):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def get_bandit_feedback(self, game, action, outcome):
        return game.banditFeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        distribution, context_generator, jobid = job

        np.random.seed(jobid)
        outcome_distribution =  {'spam':distribution[0],'ham':distribution[1]}
        game.set_outcome_distribution( outcome_distribution,jobid )

        outcomes = self.get_outcomes(game, )
        contexts = [ context_generator.get_context(outcome) for outcome in outcomes ]
        # context_generator.generate_unique_context()
        # contexts = [ context_generator.get_same_context(outcome) for outcome in outcomes ]
        
        if context_generator.type == 'polynomial':
            contexts = np.array(contexts).squeeze()
            contexts = PolynomialFeatures(6).fit_transform( contexts)
            contexts = contexts.tolist()
            dim = len(contexts[0])
            contexts = [ np.array(elmt).reshape( (dim,1) ) for elmt in contexts]

        cumRegret =  np.zeros(self.horizon, dtype =float)

        alg.reset()

        for t in range(self.horizon):

            outcome = outcomes[t]
            context = contexts[t]

            action = alg.get_action(t, context)
            
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )
            
            regret = game.LossMatrix[action, outcome] - np.min( game.LossMatrix[...,outcome] )

            cumRegret[t] =  regret

        return  np.cumsum( cumRegret ) 