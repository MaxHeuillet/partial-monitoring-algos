import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import attacks
import models
from tqdm.notebook import tqdm
import numpy as np

def create_task( input, model,probas, name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = []
    label = []
    status = []
    for i in tqdm( range( len(input) ) ): #
        is_corrupt = np.random.choice( [1, 0], p=probas)
        X,y = input[i]
        X = X.reshape( (1,1,28,28) ).to(device)
        y = torch.Tensor([ y ]).type(torch.LongTensor).to(device)
        if is_corrupt==1:
            delta = attacks.pgd_linf(model, X , y).to(device)
            X = X + delta
        data.append(  X[0] )
        label.append(y)
        status.append([is_corrupt])
    data = torch.stack(data)
    label = torch.stack(label)
    status = torch.Tensor(status)
    
    dataset = TensorDataset( data,label,status )
    
    if name == 'train':
        
        online = dataset[30000:]
        online = TensorDataset( online[0], online[1],online[2] )
        torch.save(online,'./{}_online.pt'.format(name) )
    
        offline = dataset[:30000]
        offline = TensorDataset( offline[0], offline[1],offline[2] )
        torch.save(offline,'./{}_offline.pt'.format(name) )
        
    else:
        
        torch.save(offline,'./{}.pt'.format(name) )

def create_train_val():
    
    target = models.load_target()

    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor() )
    create_task(mnist_train,target,[0.5,0.5], 'train')

    mnist_val = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor() )
    create_task(mnist_val,target,[0.5,0.5], 'val')


def task_loader():
    
    target = models.load_target()

    train_data = torch.load('./train.pt')
    train_ratio = 0.5
    train_size = int(len(train_data) * train_ratio)
    dump_size = len(train_data) - train_size
    train_data = torch.utils.data.random_split(train_data, [train_size, dump_size])[0]

    batch_size = 128

    dataloaders = {'train': DataLoader( train_data , batch_size = batch_size, shuffle=True),
               'val': DataLoader( torch.load('./val.pt'), batch_size = batch_size, shuffle=True)  }

    dataset_sizes = {'train': 0.70, 'val': 0.30}

    return dataloaders,dataset_sizes   
 