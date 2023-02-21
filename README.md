# partial_monitoring_algorithms

---
title: "My Document"
bibliography: references.bib
link-citations: true
---

This repository contains implementations of multiple stochastic partial monitoring agents.

Requirements:
```
pip install numpy
pip install scipy
```

**First step:** identify the partial monitoring game you need. We have implemented the Apple Tasting @helmbold2000apple and the Label Efficient @helmbold1997some ganes as well as all the quantities required by the proposed agents. 

```python
import games
game = games.apple_tasting(False)
game = games.label_efficient()
```

**Second step:** define an evaluation instance, called ```job```, where ```[p, 1-p]``` is a 2-dimensional outcome distribution ( Apple Tasting and Label Efficient games are 2-outcomes games):
```python
seed = 3
p = 0.8
job = [p, 1-p], seed 
```

**Third step:** evaluate a partial monitoring agent on the game instance:

```python
import evaluation

horizon = 100
evaluation.Evaluation(horizon)


###### PM-DMED:
c =1 
alg = PM_DMED.PM_DMED( game, horizon, c) 
result_pmdmed = ev.eval_policy_once(alg, game, job)

###### CBP:
alpha = 1.01
alg = cbp.CBP(  game, horizon, alpha )
result_cbp = ev.eval_policy_once(alg, game, job)

###### RandCBP:
alpha = 1.01
sigma = 1
K = 20
epsilon = 10e-7
alg = randcbp.RandCBP(  game, horizon, alpha, sigma, K, epsilon) 
result_randcbp = ev.eval_policy_once(alg, game, job)

###### BPMleast:
alg = bpm.BPM(  game, horizon )
result_bpmleast = ev.eval_policy_once(alg, game, job)

##### BPM-TS:
R = 0
alg = TSPM.TSPM_alg(  game, horizon, R) 
result_bpmts = ev.eval_policy_once(alg, game, job)

###### TSPM:
R = 1
alg = TSPM.TSPM_alg(  game,horizon, R) 
result_tspm = ev.eval_policy_once(alg, game, job)
```

**Fourth step:** plot the cumulative regret

```python
import matplotlib.pyplot as plt
plt.plot(range(horizon), result_pmdmed, label = 'pm-dmed' )
plt.plot(range(horizon), result_cbp, label = 'cbp' )
plt.plot(range(horizon), result_randcbp, label = 'randcbp' )
plt.plot(range(horizon), result_bpmleast, label = 'bpm-least' )
plt.plot(range(horizon), result_bpmts, label = 'bpm-ts' )
plt.plot(range(horizon), result_tspm, label = 'tspm' )
```

![Alt text](./partial_monitoring/tutorial.png "Example")
