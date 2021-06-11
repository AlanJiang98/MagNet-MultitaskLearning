import os
import torch
from torch.utils.data import Subset
import random
from typing import TypeVar
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
import numpy as np
import logging
from sklearn.model_selection import StratifiedShuffleSplit

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_logger(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


def get_dataset_pre(data='mnist' ,datapath='../data', size=32, is_train=True):
    if data == 'mnist':
        return MNIST(datapath, train=is_train, download=True, transform=transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ))
    elif data == 'fmnist':
        return FashionMNIST(datapath, train=is_train, download=True, transform=transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ))
    elif data == 'cifar10':
        return CIFAR10(datapath, train=is_train, download=True, transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(),transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ))
    else:
        raise ValueError('Invalid Dataset name in get dataset!')

def trainset_spilt(dataset, rate):
    labels = [dataset[i][1] for i in range(len(dataset))]
    ss = StratifiedShuffleSplit(n_splits=1, train_size=1-rate, test_size=rate, random_state=0)
    train_indices, test_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    trainset = Subset(dataset, train_indices)
    testset = Subset(dataset, test_indices)
    return trainset, testset

def get_dataset(data='mnist', datapath='../data', size=32, is_train=True, rate=0.1):
    dataset = get_dataset_pre(data, datapath, size, is_train)
    if is_train:
        return trainset_spilt(dataset, rate)
    return dataset

T = TypeVar('T')

def to(thing: T, device=None) -> T:
    if thing is None:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(thing, (list, tuple)):
        return [to(item, device) for item in thing]
    if isinstance(thing, dict):
        return {k: to(v, device) for k,v in thing.items()}
    return thing.to(device)

def setup_run(deterministic=False, given_seed=None):
    manual_seed = random.randint(0, 1023) if given_seed is None else given_seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    return manual_seed

