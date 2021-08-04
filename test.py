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

from sklearn.metrics import roc_auc_score

from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
from RotDataset import RotDataset
from utils import *

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

def main(args):
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

    for idx, loader in enumerate([id_test_loader, ood_test_loader]):
        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(loader):
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

            classification_probabilities = F.softmax(logits[:batch_size], dim = -1)
            rot_logits = model.rot_head(pen)

            classification_loss = torch.max(classification_probabilities, dim = -1)[0].data.cpu()
            rotation_loss = F.cross_entropy(rot_logits, batch_rot_y, reduction = 'none').data
    
            uniform_distribution = torch.zeros_like(classification_probabilities).fill_(1 / num_classes)
            kl_divergence_loss = nn.KLDivLoss(reduction = 'none')(classification_probabilities.log(), uniform_distribution).data

            for i in range(batch_size):
                if args.method == 'msp':
                    score = - classification_loss[i]
                elif args.method == 'rot':
                    rotation_loss_tensor = torch.tensor([rotation_loss[i], rotation_loss[i + batch_size], rotation_loss[i + 2 * batch_size], rotation_loss[i + 3 * batch_size]])
                    score = - torch.sum(kl_divergence_loss[i]) + torch.mean(rotation_loss_tensor)

                if idx == 0:
                    id_testdata_score.append(score)
                elif idx == 1:
                    ood_testdata_score.append(score)

    y_true = torch.cat((
        torch.zeros(len(id_testdata_score)),
        torch.ones(len(ood_testdata_score))
    ), 0)

    y_score = torch.cat((
        torch.tensor(id_testdata_score),
        torch.tensor(ood_testdata_score)
    ), 0).float()

    ## 2. calculate AUROC by using ood scores
    print(f"dataset : {args.ood_dataset}, method : {args.method}")
    print(roc_auc_score(y_true, y_score))

if __name__ == "__main__":
    # easy dict
    args = easy_dict()

    args.ood_dataset = "cifar100"
    args.method = "msp"
    main(args)

    args.ood_dataset = "cifar100"
    args.method = "rot"
    main(args)

    args.ood_dataset = "svhn"
    args.method = "msp"
    main(args)

    args.ood_dataset = "svhn"
    args.method = "rot"
    main(args)