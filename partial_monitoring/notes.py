n_cores = 8
n_folds = 8
horizon = 20000

game = games.apple_tasting(False)
algos = [ TSPM.TSPM_alg(  game, horizon, 0) , TSPM.TSPM_alg(  game,horizon, 1) ]
labels = [  'TSPMGaussian', 'TSPM'  ]  

for alg, label in zip( algos, labels):
    print('AT easy')
    result1 = evaluate_parallel(n_cores, n_folds, horizon, alg, game, 'easy')
    np.save('./results/AT/easy_{}_{}_{}'.format(horizon,n_folds, label), result1)
    print('AT hard')
    result2 = evaluate_parallel(n_cores, n_folds, horizon, alg, game, 'hard')
    np.save('./results/AT/hard_{}_{}_{}'.format(horizon,n_folds, label), result2)

# game = games.label_efficient()
# algos = [ TSPM.TSPM_alg(  game, horizon, 0) , TSPM.TSPM_alg(  game,horizon, 1) ]
# labels = [  'TSPMGaussian', 'TSPM'  ]  

# for alg, label in zip( algos,  labels):
#     print('LE easy')
#     result3 = evaluate_parallel(n_cores, n_folds, horizon, alg, game, 'easy')
#     np.save('./results/LE/easy_{}_{}_{}'.format(horizon,n_folds, label), result3)
#     print('LE hard')
#     result4 = evaluate_parallel(n_cores, n_folds, horizon, alg, game, 'hard')
#     np.save('./results/LE/hard_{}_{}_{}'.format(horizon,n_folds, label), result4)


import PGIDSratio

n_cores = 8
n_folds = 15
horizon = 20000

game =  games.apple_tasting( False ) 

alg = PGIDSratio.PGIDSratio( game, horizon, 28 )
task = Evaluation(horizon, 'difficult')

result = evaluate_parallel(n_cores, n_folds, horizon, alg, game, 'difficult', 'quintic')

# n_cores = 8
# n_folds = 8
# horizon = 1000

# result = evaluate_parallel(n_cores, n_folds, horizon, alg, game, 'easy')
# regret =  np.mean(result, 0) 
# xcoords = np.arange(0,horizon,1).tolist()
# std =  np.std(result,0) 

# plt.plot( regret )

import PM_DMED

n_cores = 1
n_folds = 1
horizon = 100

# np.seterr(all='raise')

# game = games.apple_tasting(False, outcome_distribution) 

outcome_distribution = [0.8,0.2]
job = (outcome_distribution, 1 )


game =  games.label_efficient(  ) 
game.set_outcome_distribution( {'spam':outcome_distribution[0],'ham':outcome_distribution[1]} )
print('optimal action', game.i_star)


# print('optimal action', game.i_star)
alg = cpb.CPB(  game, horizon,1.01) #TSPM.TSPM_alg(  game, horizon, 1)
task = Evaluation(horizon, 'easy')

result = task.eval_policy_once(alg,game, job)
#plt.plot(range(horizon), result)
# fig = go.Figure( )
# regret = np.array([ game.delta(i) for i in range(game.n_actions) ]).T @ np.mean(result,0) 
# xcoords = np.arange(0,horizon,1).tolist()

# fig.add_trace(go.Scatter(x=xcoords, y=regret, line=dict(color='blue'), mode='lines',  name='TPSM' )) # 


# game_name = 'AT'
# task = 'difficult'
# context_type = 'quintic'
# horizon = 20000
# n_folds = 15
# import subprocess
# label ='PGIDSratio'

# with gzip.open( './contextual_results/{}/{}_{}_{}_15_{}.pkl.gz'.format(game_name, task, context_type, horizon, label) ,'wb') as g:

#     for jobid in [71,72,74,75,77,82,83,84,88,89,90,92,93,94,95]:

#         with gzip.open(  './contextual_results/{}/{}_{}_{}_96_{}_{}.pkl.gz'.format(game_name, task, context_type, horizon, label, jobid) ,'rb') as f:
#             r = pkl.load(f)

#         pkl.dump( r, g)