
import numpy as np
import os

from multiprocess import Pool
from functools import partial
import pickle as pkl
import gzip
import plotly.graph_objects as go

import games

import cpb
import bpm
import random_algo
import TSPM
import cpb_gaussian
import PM_DMED

######################
######################

def evaluate_parallel(nbCores, n_folds, horizon, alg, game, type):

    print("nbCores:", nbCores, "nbFolds:", n_folds, "Horizon:", horizon)
    pool = Pool(processes = nbCores) 
    task = Evaluation(horizon, type)

    np.random.seed(1)
    distributions = []

    for jobid in range(n_folds):
        
        if type == 'easy' :
            p = np.random.uniform(0, 0.2) if np.random.uniform(0,1)>0.5 else np.random.uniform(0.8, 1)
        #elif type == 'easy' and jobid > 100:
        #    p = np.random.uniform(0.8, 1)
        else:
            p = np.random.uniform(0.4,0.6)
        distributions.append( [p, 1-p] )

    return np.asarray(  pool.map( partial( task.eval_policy_once, alg, game ), zip(distributions ,range(n_folds)) ) ) 

class Evaluation:

    def __init__(self, horizon, type ):
        self.horizon = horizon
        self.type = type
        # self.outcome_distribution = outcome_distribution

    def get_outcomes(self, game, job_id):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):

        alg.reset()

        distribution, jobid = job

        #print('seed {} distribution {}'.format(jobid, distribution)) 
        np.random.seed(jobid)

        outcome_distribution = {'spam':distribution[0],'ham':distribution[1]}

        game.set_outcome_distribution( outcome_distribution, jobid )
        optimal_action = game.i_star

        action_counter = np.zeros( (game.n_actions, self.horizon) )
        cum_regret = []

        # generate outcomes obliviously
        outcomes = self.get_outcomes(game, jobid)
        # outcomes, summary = self.distribution_shift(game )

        for t in range(self.horizon):

            # policy chooses one action
            action = alg.get_action(t)

            # Environment chooses one outcome
            outcome = outcomes[t]

            #print('t', t, 'action', action, 'outcome', outcome, )

            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, None, t)
            
            # print('nu', alg.nu / alg.n )

            for i in range(game.n_actions):
                if i == action:
                    action_counter[i][t] = action_counter[i][t-1] + 1
                else:
                    
                    action_counter[i][t] = action_counter[i][t-1]

        # regret = []
        # for t in range(self.horizon):
        #     regret.append(  self.delta_t( game, summary, t )  @ action_counter[:,t] )
            # cum_regret.append(  game.LossMatrix[action,outcome] - min( game.LossMatrix[...,outcome ] )  )
            
        regret = np.array( [ game.delta(i) for i in range(game.n_actions) ]).T @ action_counter
        
        return regret #np.cumsum(cum_regret)

def run_experiment(name, task, n_cores, n_folds, horizon, game, algos, colors, labels):

    fig = go.Figure( )

    final_regrets = []

    for alg, color, label in zip( algos, colors, labels):

        r,g,b = color
        result = evaluate_parallel(n_cores, n_folds, horizon, alg, game, '{}'.format(task) )


        with gzip.open(  os.path.join( './partial_monitoring/results', '{}_{}_{}_{}_{}.pkl.gz'.format(name, task, horizon, n_folds, label)  ), "wb" ) as f:
            pkl.dump(result, f)

        # with gzip.GzipFile(  './results/label_efficient/{}_{}_{}_{}_{}.pkl.gz'.format(name, task, horizon, n_folds, label)  ,'wb') as f:
        #     pkl.dump(result,f)
        
        # np.save('./results/label_efficient/easy_{}_{}_{}'.format(horizon,n_folds, label), result)
        # result = np.load( './results/label_efficient/easy_{}_{}_{}.npy'.format(horizon,n_folds, label) )

        final_regrets.append( result[:,-1] )
        regret =  np.mean(result, 0) 
        xcoords = np.arange(0,horizon,1).tolist()
        std =  np.std(result,0) 
        upper_regret = regret + std

        fig.add_trace(go.Scatter(x=xcoords, y=regret, line=dict(color='rgb({},{},{})'.format(r,g,b)), mode='lines',  name=label )) # 

        fig.add_trace(   go.Scatter( x=xcoords+xcoords[::-1], y=upper_regret.tolist()+regret.tolist()[::-1],  fill='toself', fillcolor='rgba({},{},{},0.2)'.format(r,g,b), 
                            line=dict(color='rgba(255,255,255,0)'),   hoverinfo="skip",  showlegend=False )  )
        
    fig.update_xaxes( type="linear")
    fig.update_yaxes( type="log",range=[0, 5] )
    fig.update_layout(legend= dict(yanchor="top",y=0.98,xanchor="left",x=0.1), autosize=False,
                    xaxis_title="Sequence", yaxis_title="Regret",  margin=go.layout.Margin( l=0,   r=0,   b=0,    t=0, ),   font=dict(size=13,) )
    fig.write_image("./{}_{}.pdf".format(name, task) )

    fig.show()

    final_regrets = np.array(final_regrets)
    final = [ ( np.argmin(final_regrets[:,i]), i) for i in range(n_folds) ]

    return True


###################################
###################################

### Label Efficient game:

horizon = 10
n_cores = 1
n_folds = 1

game = games.label_efficient(  )

algos = [ random_algo.Random(  game, horizon, ), 
          cpb.CPB(  game, horizon, 1.01),  
          cpb_gaussian.CPB_gaussian(  game, horizon, 1.01, True, 1/16, 10, False),
          PM_DMED.PM_DMED(  game, horizon, 100), 
          PM_DMED.PM_DMED(  game, horizon, 10), 
          PM_DMED.PM_DMED(  game, horizon, 1),
          PM_DMED.PM_DMED(  game, horizon, 0.1),  
          TSPM.TSPM_alg(  game, horizon, 1),
          TSPM.TSPM_alg(  game, horizon, 0),
          bpm.BPM(game, horizon) ]

colors = [  [0,0,0], [250,0,0], [0,250,0], [0,150,0], [0,0,250], [0,0,200], [0,0,150], [0,0,100],  [225,0,225], [150 , 0, 150], [0 , 250, 250] ] 
labels = [   'random', 'CBP', 'RandCBP', 'PM_DMEDc100', 'PM_DMEDc10', 'PM_DMEDc1', 'PM_DMEDc01', 'TSPM_R0', 'TSPM_R1', 'BPM_LEAST'  ]  

run_experiment('LE', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels)

run_experiment('LE', 'difficult', n_cores, n_folds, horizon, game, algos, colors, labels)


### Apple Tasting game:

horizon = 10
n_cores = 1
n_folds = 1

game = games.apple_tasting(False)

algos = [ random_algo.Random(  game, horizon, ), 
          cpb.CPB(  game, horizon, 1.01),  
          cpb_gaussian.CPB_gaussian(  game, horizon, 1.01, True, 1/16, 10, False),
          PM_DMED.PM_DMED(  game, horizon, 100), 
          PM_DMED.PM_DMED(  game, horizon, 10), 
          PM_DMED.PM_DMED(  game, horizon, 1),
          PM_DMED.PM_DMED(  game, horizon, 0.1),  
          TSPM.TSPM_alg(  game, horizon, 1),
          TSPM.TSPM_alg(  game, horizon, 0),
          bpm.BPM(game, horizon) ]

colors = [  [0,0,0], [250,0,0], [0,250,0], [0,150,0], [0,0,250], [0,0,200], [0,0,150], [0,0,100],  [225,0,225], [150 , 0, 150], [0 , 250, 250] ] 
labels = [   'random', 'CBP', 'RandCBP', 'PM_DMEDc100', 'PM_DMEDc10', 'PM_DMEDc1', 'PM_DMEDc01', 'TSPM_R0', 'TSPM_R1', 'BPM_LEAST'  ]  

run_experiment('AT', 'easy', n_cores, n_folds, horizon, game, algos, colors, labels)

run_experiment('AT', 'difficult', n_cores, n_folds, horizon, game, algos, colors, labels)
