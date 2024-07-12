import pandas as pd
import math
import time
import os
import sys
import pickle
import argparse
import numpy as np
import random

import tensorboard_logger as tb_logger
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from dataset import Nina1Dataset, Nina2Dataset
from networks.STCNet import STCNetSAC
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, get_data
from augmentations import GaussianNoise, MagnitudeWarping, WaveletDecomposition, Permute
from losses import MultiSupConLoss


# seed
seed = 42
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,140',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='STCNet')
    parser.add_argument('--dataset', type=str, default='nina1',
                        choices=['nina1', 'nina2', 'nina4'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--sampled', action='store_true',
                        help='using sampled dataset (10000 -> 500)')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='choose gamma')
    
    # kfold
    parser.add_argument('--kfold', type=int, default=-1,
                        help='if you want to use kfold val, choose 0~4')
    
    # augmentations
    parser.add_argument('--prob', type=float, default=0.5,
                        help='probablity for augmentations')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/Multi/{}_models'.format(opt.dataset)
    opt.tb_path = './save/Multi/{}_tensorboard'.format(opt.dataset)
    opt.pkl_path = './save/Multi/{}_pkl'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'lr_{}_decay_{}_bsz_{}_temp_{}_tri_{}_gamma_{}'.\
        format(opt.learning_rate, opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.gamma)
    
    if opt.kfold in range(5):
        opt.model_name = 'kfold{}_{}'.format(opt.kfold, opt.model_name)
    
    if opt.cosine:
        opt.model_name = '{}_cos'.format(opt.model_name)

    if opt.sampled:
        opt.model_name = '{}_samp'.format(opt.model_name)


    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.pkl_folder = os.path.join(opt.pkl_path, opt.model_name)
    if not os.path.isdir(opt.pkl_folder):
        os.makedirs(opt.pkl_folder)

    return opt


def set_loader(opt):

    train, _ = get_data(opt.dataset, opt.kfold)


    train_transform = transforms.Compose([
        GaussianNoise(p=opt.prob),
        MagnitudeWarping(p=opt.prob),
        WaveletDecomposition(p=opt.prob),
        Permute(data=opt.dataset, model=opt.model)
    ])

    if opt.dataset == 'nina1':
        train_dataset = Nina1Dataset(train, mode='multi', transform=TwoCropTransform(train_transform))
    elif opt.dataset in ['nina2', 'nina4']:
        train_dataset = Nina2Dataset(train, sampled = opt.sampled, mode='multi', transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    criterion = MultiSupConLoss(temperature=(opt.temp, opt.temp), gamma=opt.gamma)
    if opt.sampled:
        model = STCNetSAC(data = f'{opt.dataset}_sampled')
    else:
        model = STCNetSAC(data = opt.dataset)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    losses = AverageMeter()

    for idx, (inputs, labels, subjects) in enumerate(train_loader):

        inputs = torch.cat([inputs[0], inputs[1]], dim=0)
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        labels_features, subjects_features = model(inputs)

        f1, f2 = torch.split(labels_features, [bsz, bsz], dim=0)
        labels_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        f3, f4 = torch.split(subjects_features, [bsz, bsz], dim=0)
        subjects_features = torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)

        loss = criterion(labels_features, subjects_features, labels, subjects)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    lrs = []
    losses = []

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # pkl
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # save the figure pkl
    save_pkl = os.path.join(
        opt.pkl_folder, 'figure.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump({
            'lrs': lrs, 
            'losses': losses
            }, f)


if __name__ == '__main__':
    main()