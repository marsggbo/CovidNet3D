# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os

import imblearn
import nni
import numpy as np
import pandas as pd
import torch
from sklearn import metrics as skmetrics

__all__ = [
    'MyLogger',
    'calc_real_model_size',
    'metrics',
    'reward_function',
    'parse_preds'
]

class MyLogger(object):
    def __init__(self, name, cfg=None):
        self.file = cfg.logger.log_file if cfg is not None else 'log.txt'
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        formatter = logging.Formatter('[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

        hdlr = logging.FileHandler(self.file, 'a', encoding='utf-8')
        hdlr.setLevel(logging.INFO)
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

        strhdlr = logging.StreamHandler()
        strhdlr.setLevel(logging.INFO)
        strhdlr.setFormatter(formatter)
        self.logger.addHandler(strhdlr)

        hdlr.close()
        strhdlr.close()

    def getlogger(self):
        return self.logger

def calc_real_model_size(model, mutator):
    '''calculate the size of real model
        real_size = size_choice + size_non_choice
    '''
    def size(module):
        return sum([x.numel() for x in module.parameters()])

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    else:
        model = model
    size_choice = 0 # the size of LayerChoice
    size_non_choice = 0 # the size of normal model part

    # the size of normal model part
    layerchoices = []
    for name, module in model.named_modules():
        if isinstance(module, nni.nas.pytorch.mutables.LayerChoice):
            layerchoices.append(module)
    size_non_choice = size(model) - sum([size(lc) for lc in layerchoices])

    # the real size of all LayerChoice
    for lc in layerchoices:
        size_ops = []
        for index, op in enumerate(lc.choices):
            size_ops.append(sum([p.numel() for p in op.parameters()]))
        lc_key = lc.key
        index = np.argmax(mutator.status()[lc_key])
        size_choice += size_ops[index]

    real_size = size_choice + size_non_choice
    return real_size

def metrics(outputs, targets, topk=(1, 3)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    if maxk > outputs.shape[1]:
        maxk = outputs.shape[1]
    batch_size = outputs.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    res['save_metric'] = res[f"acc{topk[0]}"]
    return res

def reward_function(outputs, targets, topk=(1,)):
    batch_size = targets.size(0)
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == targets).sum().item() / batch_size

def parse_preds(y_true, y_pred, **kwargs):
    cls_report = imblearn.metrics.classification_report_imbalanced(y_true,np.argmax(y_pred,1),digits=4)
    covid_auc = 0
    y_true_onehot = np.eye(len(y_true))[y_true]
    fpr,tpr,_ = skmetrics.roc_curve(y_true_onehot[:,1],y_pred[:,1])
    covid_auc = skmetrics.auc(fpr, tpr)
    valid_report = {
        'y_true': y_true,
        'y_pred': y_pred,
        'cls_report': str(cls_report),
        'covid_auc': covid_auc
    }
    return valid_report

def visualize_report(file, x='flops(MFLOPS)', y='meters'):
    df = pd.read_csv(file)
    plot = df.plot.scatter(x=x, y=y)
    fig = plot.get_figure()
    path = os.path.dirname(file)
    save_file = os.path.join(path, "output.png")
    fig.savefig(path)