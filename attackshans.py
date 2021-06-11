import os
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from modules import MnistMolel, FMnistMolel, Cifar10Molel


from utils import get_dataset, mkdir, setup_run, to, get_logger
from art.attacks import FastGradientMethod, CarliniL2Method, DeepFool, SaliencyMapMethod
from art.classifiers import PyTorchClassifier

class AdversarialAttack:
    def __init__(self, model, name, attackmethod=None, noattack=False):
        self.model = model.eval()
        self.x = None
        self.y = None
        self.name = name
        self.attackmethod = attackmethod
        self.noattack = noattack

    def attack(self, data_loader):
        correct = 0
        total = 0
        all_x_adv = []
        all_y = []
        for i, (x, y) in enumerate(tqdm(data_loader, desc='Attack')):
            all_y.append(y)
            x, y = to(x), to(y)
            if self.noattack:
                x_adv = x
            else:
                x_adv = self.attackmethod.generate(x=x.cpu().numpy())
                x_adv = to(torch.from_numpy(x_adv))
            y_adv = self.model(x_adv)
            pred = y_adv.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += y.size(0)
            all_x_adv.append(x_adv.cpu())
        self.x = torch.cat(all_x_adv, dim=0)
        self.y = torch.cat(all_y, dim=0)
        return correct / total

    def save(self, save_dir):
        torch.save({'x': self.x.cpu().detach(), 'y': self.y.cpu().detach()},
                   os.path.join(save_dir, self.name + '.pth'))


def main():
    parser = argparse.ArgumentParser(description='Attacks PyTorch')
    parser.add_argument('--min_value', type=float, default=0.0)
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--T', type=float, default=2.0)
    args = parser.parse_args()

    setup_run(given_seed=1234)
    base_dir = './save_attacks/'
    mkdir(base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.dataset
    modelname = args.dataset

    test_data_loader_all = DataLoader(get_dataset(datasets, '../data', 32, False, 0.1), batch_size=64, shuffle=True)
    xs, ys = [], []
    for i, (x, y) in enumerate(test_data_loader_all):
        xs.append(x)
        ys.append(y)
        if i==10:
            holdout = (torch.cat(xs, dim=0), torch.cat(ys, dim=0))
    test_data_loader = DataLoader(TensorDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0)), batch_size=64)

    models = {'mnist':MnistMolel, 'fmnist':FMnistMolel, 'cifar10':Cifar10Molel}
    model = models[modelname](4, args.T,3 if args.dataset == 'cifar10' else 1)
    model.load_state_dict(torch.load('./train_models/{}.pth'.format(args.dataset)))
    model = to(model).eval()
    model.set_id(4)
    #model.reset_fix_param()

    logfile = './save_attacks/{}_log_attacks_for_paper.txt'.format(datasets)
    logger = get_logger(logfile)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        clip_values=(args.min_value, args.max_value),
        input_shape=(1, 32, 32),
        nb_classes=10,
    )

    attackmethods = [
        FastGradientMethod(classifier=classifier, eps=0.3, batch_size=64),
        FastGradientMethod(classifier=classifier, eps=0.15, batch_size=64),
        SaliencyMapMethod(classifier=classifier, batch_size=64),
        DeepFool(classifier=classifier, batch_size=64),
        CarliniL2Method(classifier=classifier, confidence=0.0, batch_size=64),
        CarliniL2Method(classifier=classifier, confidence=10.0, batch_size=64)
    ]

    attackers = [
        AdversarialAttack(model, '{}_Noattack'.format(datasets), noattack=True),
        AdversarialAttack(model, '{}_FGSM0.3'.format(datasets), attackmethods[0]),
        #AdversarialAttack(model, '{}_FGSM0.15'.format(datasets), attackmethods[1]),
        AdversarialAttack(model, '{}_JSMA'.format(datasets), attackmethods[2]),
        AdversarialAttack(model, '{}_DeepFool'.format(datasets), attackmethods[3]),
        AdversarialAttack(model, '{}_CW2_k0'.format(datasets), attackmethods[4]),
        #AdversarialAttack(model_cls, '{}_{}_CW2_k10'.format(datasets, model), attackmethods[5])
    ]

    plt.figure(figsize=(40, 20))
    for id, attacker in enumerate(attackers):
        acc = attacker.attack(test_data_loader)
        print('\t{} Acc:'.format(attacker.name), acc)
        logger.info('\n {} Acc: {}'.format(attacker.name, acc))
        #attacker.save(base_dir)
        plt.subplot(1, 5, id + 1)
        plt.axis('off')
        plt.title(attacker.name)
        plt.imshow(np.transpose(vutils.make_grid(attacker.x[:9], nrow=3, range=(-1.0, 1.0), padding=5), (1, 2, 0)))
    plt.savefig('./save_attacks/attackers_{}_for_paper.jpg'.format(attacker.name))

if __name__ == '__main__':
    main()




