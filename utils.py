from sklearn.linear_model import SGDClassifier
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import geometry
import itertools
from collections import defaultdict
import collections
import target
import controler
import utils

def cumsum(vector):
    cumulative = []
    sum =0
    for i in vector:
        if i == None:
            cumulative.append(None)
        else:
            sum += i
            cumulative.append(sum)
    return cumulative



def STAP(clf, controler_input, nb_mistakes, i):    

    prediction = clf(controler_input)
    initial_action = torch.round( nn.Sigmoid()(prediction) )

    r = np.random.uniform(0,1)
    flip_proba = np.sqrt( (1+nb_mistakes)/(i+1) )

    if (initial_action == 1 ) or (initial_action == 0 and r <= flip_proba):
        STAP_action = 1
    else:
        STAP_action = 0
    
    return initial_action, STAP_action

def CesaBianchi(clf, controler_input, beta, K):

    prediction = clf(controler_input)
    initial_action = torch.round( nn.Sigmoid()(prediction) )

    p = beta / ( beta + abs( prediction.item() ) )
    cesa_action = np.random.binomial(1, p)
    
    
    return initial_action, cesa_action
    

def PLOT(epsilon,):
    q = np.random.binomial(1, epsilon)



def fgsm_attack(target, epsilon, x, y, n_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    if n_classes <=2:
        dlt =  target.clf(x)[0] - y
    else:
        label = torch.zeros(n_classes)
        label[y] = 1
        dlt =  target.clf( x ) - label

    direction = torch.sign( torch.matmul( dlt, target.clf[0].weight ) ).to(device)
    x_attacked = x + epsilon * direction
    return x_attacked