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

class Target:

    def __init__(self,  n_classes, n_features, seed = 42):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.clf = nn.Sequential( nn.Linear(n_features, n_classes)).to(device)
        self.opt = optim.SGD( self.clf.parameters(), lr=1e-1 )

    def epoch(self, loader, model, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        total_loss, total_err = 0.,0.
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def train(self, loader, n_epochs=1):
        for t in range(n_epochs):
            train_err, train_loss = self.epoch(loader, self.clf, self.opt)
