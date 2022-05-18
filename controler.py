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

class Controler:

    def __init__(self,  n_classes, n_features, seed = 42):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.clf = nn.Sequential( nn.Linear(n_features, 1) ).to(self.device)
        self.opt = optim.SGD( self.clf.parameters(), lr=1e-1 )
        self.X_train = torch.Tensor([]).to(self.device)
        self.y_train = torch.Tensor([]).to(self.device)

    def epoch(self, loader, model, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        
        #total_loss, total_err = 0., 0.
        for X,y in loader:
            X,y = X.to(self.device), y.to(self.device)
            yp = model(X)

            loss = nn.BCELoss()(  nn.Sigmoid()(yp) ,y)

            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            #total_err += (yp.max(dim=1)[1] != y).sum().item()
            #total_loss += loss.item() * X.shape[0]
        #return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def train(self, loader, n_epochs=1):
        for _ in range(n_epochs):
            self.epoch(loader, self.clf, self.opt)

    def append(self, x, y):
        self.X_train = torch.cat( (self.X_train,x[0]) )
        self.y_train = torch.cat( ( self.y_train,  torch.Tensor([[y]]).to(self.device) ) )

    def get_loader(self, batch_size=32):
        return DataLoader(  TensorDataset( self.X_train, self.y_train  ), batch_size = batch_size, shuffle=False) 

    def collect_fit(self, x, y):
        if y != None:
            self.append(x, y)
            self.clf.apply(reset)
            self.train(  self.get_loader(), n_epochs=1) 
        else:
            pass 

    def confidence(self,):
        return 0.1
