# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import random

import torch
import torch.nn as nn

from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .build import TRAINER_REGISTRY
from .default_trainer import Trainer

__all__ = [
    'OnehotTrainer',
]


@TRAINER_REGISTRY.register()
class OnehotTrainer(Trainer):
    def __init__(self, cfg):
        cfg.defrost()
        cfg.mutator.name = 'OnehotMutator'
        cfg.freeze()
        super().__init__(cfg)
        self.cfg = cfg
        self.debug = cfg.debug

        self.arc_lr = cfg.mutator.OnehotMutator.arc_lr
        self.unrolled = cfg.mutator.OnehotMutator.unrolled
        self.ctrl_optim = torch.optim.Adam(self.mutator.parameters(), self.arc_lr, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)

        # preparing dataset
        n_train = len(self.dataset_train)
        split = n_train // 2
        indices = list(range(n_train))
        random.shuffle(indices)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers,
                                                        pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers,
                                                        pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.workers,
                                                       pin_memory=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.mutator.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            if self.debug and step > 0:
                break
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                self._backward(val_X, val_y)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.optimizer.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)  # gradient clipping
            self.optimizer.step()

            metrics = self.metrics(logits, trn_y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Model Epoch [{}/{}] Step [{}/{}] Model size: {} {}".format(
                    epoch + 1, self.num_epochs, step + 1, len(self.train_loader), self.model_size(), meters))

        return meters

    def validate_one_epoch(self, epoch):
        return self.test_one_epoch(epoch)

    def test_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            self.mutator.reset()
            for step, (X, y) in enumerate(self.test_loader):
                if self.debug and step > 0:
                    break
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                metrics = self.metrics(logits, y)
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    self.logger.info("Test: Step [{}/{}]  {}".format(step + 1, len(self.test_loader), meters))
            self.logger.info("Final model metric = {}".format(meters.meters['save_metric'].avg))
        return meters

    def _logits_and_loss(self, X, y):
        self.mutator.reset()
        output = self.model(X)
        if isinstance(output, tuple):
            output, aux_output = output
            aux_loss = self.loss(aux_output, y)
        else:
            aux_loss = 0.
        loss = self.loss(output, y)
        loss = loss + self.cfg.model.aux_weight * aux_loss
        self._write_graph_status()
        return output, loss

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        _, loss = self._logits_and_loss(val_X, val_y)
        loss.backward()

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]["lr"]
        momentum = self.optimizer.param_groups[0]["momentum"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        _, loss = self._logits_and_loss(val_X, val_y)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple(self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.optimizer.state[w].get("momentum_buffer", 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            self.logger.warning("In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.", norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d

            _, loss = self._logits_and_loss(trn_X, trn_y)
            dalphas.append(torch.autograd.grad(loss, self.mutator.parameters()))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
