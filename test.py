# import argparse
import easydict
import random 
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader as dataloader
import torchvision.datasets as datasets

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"
print(device)

from sklearn.metrics import roc_auc_score

from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
from RotDataset import RotDataset
from utils import *

def arg_parser():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--method', type=str, default='rot', help='rot, msp')
    parser.add_argument('--ood_dataset', type=str, default='cifar100', help='cifar100 | svhn')
    parser.add_argument('--num_workers', type=int, default=8)

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--rot-loss-weight', type=float, default=0.5, help='Multiplicative factor on the rot losses')

    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    args = parser.parse_args()

    return args

def easy_dict():
    args = easydict.EasyDict({
        "seed": 0,

        "method": 'rot',
        'ood_dataset': 'cifar100',
        'num_workers': 8,

        'epochs': 100,
        'learning_rate': 0.1,
        'batch_size': 128,
        'test_bs': 200,
        'momentum': 0.9,
        'decay': 0.0005,
        'rot-loss-weight': 0.5,

        'layers': 40,
        'widen_factor': 2,
        'droprate': 0.3
    })

    return args

def main():
    # arg parser
    # args = arg_parser()
    args = easy_dict()

    # set seed
    set_seed(args.seed)
    
    # dataset 
    id_testdata = datasets.CIFAR10('./data/', train=False, download=True)
    id_testdata = RotDataset(id_testdata, train_mode=False)

    if args.ood_dataset == 'cifar100':
        ood_testdata = datasets.CIFAR100('./data/', train=False, download=True)
    elif args.ood_dataset == 'svhn':
        ood_testdata = datasets.SVHN('./data/', split='test', download=True)
    else:
        raise ValueError(args.ood_dataset)
    ood_testdata = RotDataset(ood_testdata, train_mode=False)
    
    # data loader  
    id_test_loader = dataloader(id_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    ood_test_loader = dataloader(ood_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
  
    # load model
    num_classes = 10
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    model.rot_head = nn.Linear(128, 4)
    model = model.to(device)
    model.load_state_dict(torch.load('./models/trained_model_{}.pth'.format(args.method), map_location = device))

    # TODO:
    ## 1. calculate ood score by two methods(MSP, Rot)
    model.eval()

    id_testdata_score, ood_testdata_score = [], []

    for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(id_test_loader):
        batch_size = x_tf_0.shape[0]
        batch_x = torch.cat([x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).to(device)
        batch_y = batch_y.to(device)
        batch_rot_y = torch.cat((
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long().to(device)
        
        logits, pen = model(batch_x)

        classification_logits = logits[:batch_size]
        rot_logits = model.rot_head(pen)

        classification_loss = torch.max(classification_logits, dim = -1)[0].data
        rotation_loss = F.cross_entropy(rot_logits, batch_rot_y, reduction = 'none').data
 
        uniform_distribution = torch.zeros_like(classification_logits).fill_(1 / num_classes)
        kl_divergence_loss = nn.KLDivLoss(reduction = 'none')(classification_logits, uniform_distribution).data

        for i in range(batch_size):
            msp_score = - classification_loss[i]
            rot_score = - torch.sum(kl_divergence_loss[i]) + 1 / 4 * (rotation_loss[i] + rotation_loss[i + batch_size] + rotation_loss[i + 2 * batch_size] + rotation_loss[i + 3 * batch_size])
            if args.method == 'msp':
                score = msp_score
            elif args.method == 'rot':
                score = rot_score

            id_testdata_score.append(score)

    for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(ood_test_loader):
        batch_size = x_tf_0.shape[0]
        batch_x = torch.cat([x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).to(device)
        batch_y = batch_y.to(device)
        batch_rot_y = torch.cat((
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long().to(device)
        
        logits, pen = model(batch_x)

        classification_logits = logits[:batch_size]
        rot_logits = model.rot_head(pen)

        classification_loss = torch.max(classification_logits, dim = -1)[0].data
        rotation_loss = F.cross_entropy(rot_logits, batch_rot_y, reduction = 'none').data

        uniform_distribution = torch.zeros_like(classification_logits).fill_(1 / num_classes)
        kl_divergence_loss = nn.KLDivLoss(reduction = 'none')(classification_logits, uniform_distribution).data

        for i in range(batch_size):
            msp_score = - classification_loss[i]
            rot_score = - torch.sum(kl_divergence_loss[i]) + 1 / 4 * (rotation_loss[i] + rotation_loss[i + batch_size] + rotation_loss[i + 2 * batch_size] + rotation_loss[i + 3 * batch_size])
            if args.method == 'msp':
                score = msp_score
            elif args.method == 'rot':
                score = rot_score

            ood_testdata_score.append(score)

    y_true = torch.cat((
        torch.zeros(len(id_testdata_score)),
        torch.ones(len(ood_testdata_score))
    ), 0)

    y_score = torch.cat((
        torch.tensor(id_testdata_score),
        torch.tensor(ood_testdata_score)
    ), 0).long()

    ## 2. calculate AUROC by using ood scores
    print(roc_auc_score(y_true, y_score))

if __name__ == "__main__":
    main()