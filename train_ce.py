import pandas as pd
import copy
import math
import time
import os
import argparse
import pickle
import numpy as np
import random

import tensorboard_logger as tb_logger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from dataset import NinaDataset
# from networks.STCNet import STCNetCE
from util import AverageMeter, AccuracyMeter
from util import save_model, get_data, get_model
from augmentations import GaussianNoise, MagnitudeWarping, WaveletDecomposition, Permute

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


    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # model dataset
    parser.add_argument('--model', type=str, default='STCNet', 
                        choices=['STCNet', 'baseline'], help='model')
    parser.add_argument('--stc', action='store_true')
    parser.add_argument('--encoder', type=str, default=None, help='path to encoder weights')
    parser.add_argument('--freeze', action='store_true', help='freeze encoder weights')
    parser.add_argument('--dataset', type=str, default='nina1',
                        choices=['nina1', 'nina2', 'nina4'], help='dataset')

    # augmentations
    parser.add_argument('--aug', action='store_true',
                        help='using augmentations')
    parser.add_argument('--prob', type=float, default=0.5,
                        help='probablity for augmentations')
    
    # kfold
    parser.add_argument('--kfold', type=int, default=-1,
                        help='if you want to use kfold val, choose 0~4')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    parser.add_argument('--seg', type=int, default=25,
                        help='seg')
    

    opt = parser.parse_args()
    stc_cond = ''
    if opt.stc and opt.model == 'baseline':
        stc_cond = 'stc'
    opt.model_path = './save/CE/{}_{}_{}models'.format(opt.dataset, opt.model, stc_cond)
    opt.tb_path = './save/CE/{}_{}_{}tensorboard'.format(opt.dataset, opt.model, stc_cond)
    opt.pkl_path = './save/CE/{}_{}_{}pkl'.format(opt.dataset, opt.model, stc_cond)

    opt.model_name = 'lr_{}_decay_{}_bsz_{}_tri_{}'.\
        format(opt.learning_rate, opt.weight_decay, opt.batch_size, opt.trial)
    
    if opt.kfold in range(5):
        opt.model_name = 'kfold{}_{}'.format(opt.kfold, opt.model_name)
    
    if opt.encoder is not None:
        enc_cfg = opt.encoder
        enc_cfg = enc_cfg.split('/')[-2]
        opt.model_name = '{}_enc_{}'.format(opt.model_name, enc_cfg)

    if opt.cosine:
        opt.model_name = '{}_cos'.format(opt.model_name)

    if opt.aug:
        opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.prob)

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

    train, test = get_data(opt.dataset, opt.kfold)

    test_dataset = NinaDataset(test, dataset=opt.dataset, model=opt.model)
    if opt.aug:
        train_transform = transforms.Compose([
            GaussianNoise(p = opt.prob),
            MagnitudeWarping(p = opt.prob),
            WaveletDecomposition(p = opt.prob),
            Permute(data = opt.dataset)
            ])
        train_dataset = NinaDataset(train, dataset=opt.dataset, model=opt.model, transform = train_transform)
    else:
        train_dataset = NinaDataset(train, dataset=opt.dataset, model=opt.model)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle = False)

    return train_loader, test_loader


def set_model(opt):
    criterion = torch.nn.CrossEntropyLoss()
    model = get_model(opt)

    if opt.encoder is not None:
        pre_trained = torch.load(opt.encoder, weights_only=False)
        model.encoder.load_state_dict({k.replace('encoder.', '', 1): v for k, v in pre_trained['model'].items() if 'encoder' in k})
        if opt.freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, test_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""

    train_losses = AverageMeter()
    val_losses = AverageMeter()
    train_acc = AccuracyMeter()
    val_acc = AccuracyMeter()

    model.train()
    for idx, (inputs, labels, _) in enumerate(train_loader):

        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        features = model(inputs)
        loss = criterion(features, labels)

        # update metric
        train_losses.update(loss.item(), bsz)
        train_acc.update(features, labels)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for idx, (inputs, labels, _) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            features = model(inputs)
            loss = criterion(features, labels)

            val_losses.update(loss.item(), bsz)
            val_acc.update(features, labels)

    return train_losses.avg, val_losses.avg, train_acc.correct / train_acc.total, val_acc.correct / val_acc.total

def set_optimizer(opt, model):
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = opt.learning_rate, 
                                 betas = (opt.beta1, opt.beta2), 
                                 weight_decay = opt.weight_decay)
    return optimizer

def step_decay(epoch):
    drop = 0.1
    epochs_drop = 70.0
    lamb = math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lamb


def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # build scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=step_decay)
    if opt.cosine:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.learning_rate / 100)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    lrs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0


    # training routine
    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()
        train_loss, val_loss, train_acc, val_acc = train(train_loader, test_loader, model, criterion, optimizer, epoch, opt)
        scheduler.step()
        time2 = time.time()
        print('epoch {}, total time {:.2f} train_loss {:.2f} val_loss {:.2f} train_acc {:.2f} val_acc {:.2f}'.format(epoch, time2 - time1, train_loss, val_loss, train_acc*100, val_acc*100))

        # tensorboard logger
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('val_acc', val_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # pkl
        lrs.append(optimizer.param_groups[0]['lr'])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # best_model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)

    # save the best model
    print('best_acc: {}'.format(best_acc))
    save_file = os.path.join(
        opt.save_folder, f'best_model.pth')
    save_model(best_model, optimizer, opt, opt.epochs, save_file)

    # save the figure pkl
    save_pkl = os.path.join(
        opt.pkl_folder, 'figure.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump({
            'lrs': lrs, 
            'train_losses': train_losses, 
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_acc' : best_acc}, f)


if __name__ == '__main__':
    main()