# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import nni
from datasets import build_dataset
from kd_model import load_kd_model, loss_fn_kd
from losses import build_loss_fn
from networks import build_model
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter
from utils import (MyLogger, flops_size_counter, generate_optimizer,
                   generate_scheduler, metrics, mixup_loss_fn, mixup_data,
                   parse_cfg_for_scheduler)

from abc import ABC, abstractmethod
global logger
global writter

__all__ = [
    'BaseEvaluator'
]

class BaseEvaluator(ABC):

    def init(self):
        '''init train_epochs, device, loss_fn, dataset, and dataloaders, ...
        '''
        raise NotImplementedError

    def reset(self):
        '''mutable can be only initialized for once, hence it needs to
        reset model, optimizer, scheduler when a new trial starts.
        '''
        raise NotImplementedError

    def compare(self):
        '''evaluate all arcs and find the best one
        '''
        raise NotImplementedError

    def run(self, arc, validate=True):
        '''retrain the best arc from scratch
        
        self.reset()

        # init model and mutator
        mutator = apply_fixed_architecture(self.model, arc)
        for epoch in range(start_epoch, end_epoch):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            self.train_one_epoch(epoch)
            if validate:
                self.validate_one_epoch(epoch)

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)
            
        '''
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def valid_one_epoch(self, epoch):
        pass