import math
import numpy as np
import torch
import torch.optim as optim
import pandas as pd


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

def get_data(dataset, k):
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