import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from networks.STCNet import STCNetCE

import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, log_loss, matthews_corrcoef


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

    def compute(self):
        return self.correct / self.total * 100


class MetricsMeter(object):
    def __init__(self, dataset):
        if dataset == 'nina1':
            self.num_classes = 27
        elif dataset == 'nina2':
            self.num_classes = 40
        elif dataset == 'nina4':
            self.num_classes = 10
        self.reset()

    def reset(self):
        self.TP = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)
        self.TN = torch.zeros(self.num_classes)
        self.targets = []
        self.outputs = []

    def update(self, outputs, labels):

        outputs = F.softmax(outputs, dim=1) 
        _, predicted = torch.max(outputs, 1)
        self.outputs.extend(outputs.detach().cpu().numpy())
        self.targets.extend(labels.detach().cpu().numpy())

        for i in range(self.num_classes):
            tp = ((predicted == i) & (labels == i)).sum().item()
            fp = ((predicted == i) & (labels != i)).sum().item()
            fn = ((predicted != i) & (labels == i)).sum().item()
            tn = ((predicted != i) & (labels != i)).sum().item()

            self.TP[i] += tp
            self.FP[i] += fp
            self.FN[i] += fn
            self.TN[i] += tn

    def compute_metrics(self):
        targets_array = np.array(self.targets)
        outputs_array = np.array(self.outputs)
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        f1 = 2 * precision * recall / (precision + recall)
        specificity = self.TN / (self.TN + self.FP)
        balanced_accuracy = (recall + specificity) / 2


        return {
            "precision": precision.mean().item() * 100,
            "recall": recall.mean().item() * 100,
            "specificity": specificity.mean().item() * 100,
            "f1_score": f1.mean().item() * 100,
            "balanced_accuracy": balanced_accuracy.mean().item() * 100
        }
    

class PValueMeter(object):
    def __init__(self, dataset):
        if dataset == 'nina1':
            self.num_classes = 27
        elif dataset == 'nina2':
            self.num_classes = 40
        elif dataset == 'nina4':
            self.num_classes = 10
        self.reset()

    def reset(self):
        self.correct = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)

    def update(self, outputs, labels):

        _, predicted = torch.max(outputs, 1)

        for i in range(self.num_classes):
            correct_ = ((predicted == i) & (labels == i)).sum().item()
            total_ = ((labels == i)).sum().item()

            self.correct[i] += correct_
            self.total[i] += total_

    def compute_metrics(self):
        acc = self.correct / self.total
        return 


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

FOLD_CFG = {
    'nina1' : ((22, 5), (22, 5), (22, 5), (21, 6), (21, 6)),
    'nina2' : ((32, 8), (32, 8), (32, 8), (32, 8), (32, 8)),
    'nina4' : ((8, 2), (8, 2), (8, 2), (8, 2), (8, 2))
}

def get_data(dataset, k = -1):
    if k not in range(5):
        train = pd.read_pickle(f'./pkl/train_{dataset}.pkl')
        test = pd.read_pickle(f'./pkl/test_{dataset}.pkl')

        return train, test
    
    data = pd.read_pickle(f'./pkl/{dataset}_fold.pkl')
    train = data[data['fold'] != k]
    test = data[data['fold'] == k]

    train.loc[:, 'subject'] = train['subject'].map(pd.Series(index = train['subject'].unique(), data = range(FOLD_CFG[dataset][k][0])))
    test.loc[:, 'subject'] = test['subject'].map(pd.Series(index = test['subject'].unique(), data = range(FOLD_CFG[dataset][k][1])))

    return train, test

def get_model(opt):
    return STCNetCE(data = opt.dataset)

# def get_data_hr(dataset, k):
#     if k not in range(5):
#         train = pd.read_pickle(f'./pkl/hr_train_{dataset}.pkl')
#         test = pd.read_pickle(f'./pkl/hr_test_{dataset}.pkl')

#         return train, test
    
#     data = pd.read_pickle(f'./pkl/{dataset}_fold.pkl')
#     train = data[data['fold'] != k]
#     test = data[data['fold'] == k]

#     train.loc[:, 'subject'] = train['subject'].map(pd.Series(index = train['subject'].unique(), data = range(FOLD_CFG[dataset][k][0])))
#     test.loc[:, 'subject'] = test['subject'].map(pd.Series(index = test['subject'].unique(), data = range(FOLD_CFG[dataset][k][1])))

#     return train, test