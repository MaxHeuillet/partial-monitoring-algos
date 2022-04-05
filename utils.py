import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def data_loader():

    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_val = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())

    train_ratio = 0.4
    train_size = int(len(mnist_train) * train_ratio)
    dump_size = len(mnist_train) - train_size
    mnist_train = torch.utils.data.random_split(mnist_train, [train_size, dump_size])[0]

    batch_size = 128

    dataloaders = {
        'train': DataLoader(mnist_train, batch_size = batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(mnist_val, batch_size = 100, shuffle=False, num_workers = 4)
    }

    dataset_sizes = {'train': 0.85, 'val': 0.15}
    print(len(mnist_train))
    print(len(mnist_val))

    return dataloaders,dataset_sizes