import json
import os

import imblearn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import nni
from callbacks import CheckpointCallback, CAMCallback
from datasets import build_dataset
from kd_model import load_kd_model, loss_fn_kd
from losses import build_loss_fn
from networks import build_model
from nni.nas.pytorch.fixed import apply_fixed_architecture
from utils import (AverageMeterGroup, MyLogger, flops_size_counter,
                   generate_optimizer, generate_scheduler, metrics, parse_preds, 
                   mixup_data, mixup_loss_fn, parse_cfg_for_scheduler)

from .base_evaluator import BaseEvaluator
from .build import EVALUATOR_REGISTRY

__all__ = [
    'DefaultEvaluator'
]

@EVALUATOR_REGISTRY.register()
class DefaultEvaluator(BaseEvaluator):
    def __init__(self, cfg):
        super(DefaultEvaluator, self).__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.callbacks = self.generate_callbacks()

        self.arcs = self.load_arcs(cfg.args.arc_path)
        self.writter = SummaryWriter(os.path.join(self.cfg.logger.path, 'summary_runs'))
        self.logger = MyLogger(__name__, cfg).getlogger()
        self.size_acc = {} # {'epoch1': [model_size, acc], 'epoch2': [model_size, acc], ...}
        self.init_basic_settings()

    def init_basic_settings(self):
        '''init train_epochs, device, loss_fn, dataset, and dataloaders
        '''
        # train epochs
        try:
            self.train_epochs = self.cfg.args.train_epochs
        except:
            self.train_epochs = 1

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # loss_fn
        self.loss_fn = build_loss_fn(self.cfg)
        self.loss_fn.to(self.device)
        self.logger.info(f"Building loss function ...")

        # dataset
        self.train_dataset, self.test_dataset = build_dataset(self.cfg)

        # dataloader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.cfg.dataset.batch_size, 
            shuffle=True, num_workers=self.cfg.dataset.workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.cfg.dataset.batch_size, 
            shuffle=False, num_workers=self.cfg.dataset.workers, pin_memory=True)
        self.logger.info(f"Building dataset and dataloader ...")

    def load_arcs(self, arc_path):
        '''load arch json files
        Args:
            arc_path:
                (file): [arc_path]
                (dir): [arc_path/epoch_0.json, arc_path/epoch_1.json, ...]
        '''
        if os.path.isfile(arc_path):
            return [arc_path]
        else:
            arcs = os.listdir(arc_path)
            arcs = [os.path.join(arc_path, arc) for arc in arcs 
                    if arc.split('.')[-1]=='json']
            arcs = sorted(arcs, 
                          key=lambda x: int( os.path.splitext( os.path.basename(x) )[0].split('_')[1] ) )
            return arcs

    def reset(self):
        '''mutable can be only initialized for once, hence it needs to
        reset model, optimizer, scheduler when run a new trial.
        '''
        # model
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        self.logger.info(f"Building model {self.cfg.model.name} ...")

        # load teacher model if using knowledge distillation
        if hasattr(self.cfg, 'kd') and self.cfg.kd.enable:
            self.kd_model = load_kd_model(self.cfg).to(self.device)
            self.kd_model.eval()
            self.logger.info(f"Building teacher model {self.cfg.kd.model.name} ...")
        else:
            self.kd_model = None

        # optimizer
        self.optimizer = generate_optimizer(
            model=self.model,
            optim_name=self.cfg.optim.name,
            lr=self.cfg.optim.base_lr,
            momentum=self.cfg.optim.momentum,
            weight_decay=self.cfg.optim.weight_decay)
        self.logger.info(f"Building optimizer {self.cfg.optim.name} ...")

        # scheduler
        self.scheduler_params = parse_cfg_for_scheduler(self.cfg, self.cfg.optim.scheduler.name)
        self.lr_scheduler = generate_scheduler(
            self.optimizer,
            self.cfg.optim.scheduler.name,
            **self.scheduler_params)
        self.logger.info(f"Building optim.scheduler {self.cfg.optim.scheduler.name} ...")

    def compare(self):
        self.logger.info("="*20)
        self.logger.info("Selecting the best architecture ...")
        self.enable_writter = False
        # split train dataset into train and valid dataset
        train_size = int(0.8 * len(self.train_dataset))
        valid_size = len(self.train_dataset) - train_size
        self.train_dataset_part, self.valid_dataset_part = torch.utils.data.random_split(
            self.train_dataset, [train_size, valid_size])

        # dataloader
        self.train_loader_part = torch.utils.data.DataLoader(
            self.train_dataset_part, batch_size=self.cfg.dataset.batch_size, 
            shuffle=True, num_workers=self.cfg.dataset.workers, pin_memory=True)
        self.valid_loader_part = torch.utils.data.DataLoader(
            self.valid_dataset_part, batch_size=self.cfg.dataset.batch_size, 
            shuffle=True, num_workers=self.cfg.dataset.workers, pin_memory=True)

        # choose the best architecture
        for arc in self.arcs:
            self.reset()
            self.mutator = apply_fixed_architecture(self.model, arc)
            size = self.model_size()
            arc_name = os.path.basename(arc)
            self.logger.info(f"{arc} Model size={size*4/1024**2} MB")

            # train
            for epoch in range(self.train_epochs):
                self.train_one_epoch(epoch, self.train_loader_part)
            val_acc = self.valid_one_epoch(-1, self.valid_loader_part)
            self.size_acc[arc_name] = {'size': size,
                                       'val_acc': val_acc,
                                       'arc': arc}
        sorted_size_acc = sorted(self.size_acc.items(),
                                 key=lambda x: x[1]['val_acc']['save_metric'].avg, reverse=True)
        return sorted_size_acc[0][1]

    def run(self, arc, validate=True, test=False):
        '''retrain the best-performing arch from scratch
            arc: the json file path of the best-performing arch 
        '''
        self.logger.info("="*20)
        self.logger.info("Retraining the best architecture ...")
        self.enable_writter = True
        self.reset()

        # init model and mutator
        self.mutator = apply_fixed_architecture(self.model, arc)
        size = self.model_size()
        arc_name = os.path.basename(arc)
        self.logger.info(f"{arc_name} Model size={size*4/1024**2} MB")

        # callbacks
        for callback in self.callbacks:
            callback.build(self.model, self.mutator, self)

        # resume
        self.start_epoch = 0
        self.resume()

        # fintune
        # todo: improve robustness, bug of optimizer resume
        # if self.cfg.model.finetune:
        #     self.logger.info("Freezing params of conv part ...")
        #     for name, param in self.model.named_parameters():
        #         if 'dense' not in name:
        #             param.requires_grad = False

        # dataparallel
        if len(self.cfg.trainer.device_ids) > 1:
            device_ids = self.cfg.trainer.device_ids
            num_gpus_available = torch.cuda.device_count()
            assert num_gpus_available >= len(device_ids), "you can only use {} device(s)".format(num_gpus_available)
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            if self.kd_model:
                self.kd_model = torch.nn.DataParallel(self.kd_model, device_ids=device_ids)

        if test:
            meters = self.test_one_epoch(-1, self.test_loader)
            self.logger.info(f"Final test metrics= {meters}")
            return meters

        # start training
        for epoch in range(self.start_epoch, self.cfg.evaluator.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            self.logger.info("Epoch %d Training", epoch)
            self.train_one_epoch(epoch, self.train_loader)

            if validate:
                self.logger.info("Epoch %d Validating", epoch)
                self.valid_one_epoch(epoch, self.test_loader)

            self.lr_scheduler.step()

            self.cur_meters = getattr(self, 'valid_meters', self.train_meters)
            for callback in self.callbacks:
                if isinstance(callback, CheckpointCallback):
                    callback.update_best_metric(self.cur_meters.meters['save_metric'].avg)
                callback.on_epoch_end(epoch)

        self.logger.info("Final best Prec@1 = {:.4%}".format(self.best_metric))

    def train_one_epoch(self, epoch, dataloader):
        config = self.cfg
        self.train_meters = AverageMeterGroup()

        cur_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info("Epoch %d LR %.6f", epoch, cur_lr)
        if self.enable_writter:
            self.writter.add_scalar("lr", cur_lr, global_step=epoch)

        self.model.train()

        for step, (x, y) in enumerate(dataloader):
            if self.debug and step > 1:
                break
            for callback in self.callbacks:
                callback.on_batch_begin(epoch)
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            bs = x.size(0)
            # mixup data
            if config.mixup.enable:
                x, y_a, y_b, lam = mixup_data(x, y, config.mixup.alpha)
                mixup_y = [y_a, y_b, lam]

            # forward
            logits = self.model(x)

            # loss
            if isinstance(logits, tuple):
                logits, aux_logits = logits
                if config.mixup.enable:
                    aux_loss = mixup_loss_fn(self.loss_fn, aux_logits, *mixup_y)
                else:
                    aux_loss = self.loss_fn(aux_logits, y)
            else:
                aux_loss = 0.
            if config.mixup.enable:
                loss = mixup_loss_fn(self.loss_fn, logits, *mixup_y)
            else:
                loss = self.loss_fn(logits, y)
            if config.model.aux_weight > 0:
                loss += config.model.aux_weight * aux_loss
            if self.kd_model:
                teacher_output = self.kd_model(x)
                loss += (1-config.kd.loss.alpha)*loss + loss_fn_kd(logits, teacher_output, self.cfg.kd.loss)

            # backward
            loss.backward()
            # gradient clipping
            # nn.utils.clip_grad_norm_(model.parameters(), 20)

            if (step+1) % config.trainer.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # post-processing
            accuracy = metrics(logits, y, topk=(1, 3)) # e.g. {'acc1':0.65, 'acc3':0.86}
            self.train_meters.update(accuracy)
            self.train_meters.update({'train_loss':loss.item()})
            if step % config.logger.log_frequency == 0 or step == len(dataloader) - 1:
                self.logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} {}".format(
                        epoch + 1, config.trainer.num_epochs, step, len(dataloader) - 1, self.train_meters))

            for callback in self.callbacks:
                callback.on_batch_end(epoch)
        
        if self.enable_writter:
            self.writter.add_scalar("loss/train", self.train_meters['train_loss'].avg, global_step=epoch)
            self.writter.add_scalar("acc1/train", self.train_meters['acc1'].avg, global_step=epoch)
            self.writter.add_scalar("acc3/train", self.train_meters['acc3'].avg, global_step=epoch)

        self.logger.info("Train: [{:3d}/{}] Final result {}".format(
            epoch + 1, config.trainer.num_epochs, self.train_meters))

        return self.train_meters

    def valid_one_epoch(self, epoch, dataloader):
        config = self.cfg
        self.valid_meters = AverageMeterGroup()
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for step, (X, y) in enumerate(dataloader):
                if self.debug and step > 1:
                    break
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                bs = X.size(0)

                # forward
                logits = self.model(X)

                # loss
                if isinstance(logits, tuple):
                    logits, aux_logits = logits
                    aux_loss = self.loss_fn(aux_logits, y)
                else:
                    aux_loss = 0.
                loss = self.loss_fn(logits, y)
                if config.model.aux_weight > 0:
                    loss = loss + config.model.aux_weight * aux_loss

                # post-processing
                y_true.append(y.cpu().detach())
                y_pred.append(logits.cpu().detach())

                accuracy = metrics(logits, y, topk=(1, 3))
                self.valid_meters.update(accuracy)
                self.valid_meters.update({'valid_loss':loss.item()})
                if step % config.logger.log_frequency == 0 or step == len(dataloader) - 1:
                    self.logger.info(
                        "Valid: [{:3d}/{}] Step {:03d}/{:03d} {}".format(
                            epoch + 1, config.trainer.num_epochs, step, len(dataloader) - 1, self.valid_meters))

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        self.valid_report = parse_preds(np.array(y_true.detach().cpu().numpy()), np.array(y_pred.detach().cpu().numpy()))
        self.valid_report['acc1'] = self.valid_meters['acc1'].avg
        self.valid_report['epoch'] = epoch
        self.logger.info(self.valid_report['cls_report'])
        self.logger.info(self.valid_report['covid_auc'])
        if self.enable_writter  and epoch > 0:
            self.writter.add_scalar("loss/valid", self.valid_meters['valid_loss'].avg, global_step=epoch)
            self.writter.add_scalar("acc1/valid", self.valid_meters['acc1'].avg, global_step=epoch)
            self.writter.add_scalar("acc3/valid", self.valid_meters['acc3'].avg, global_step=epoch)

        self.logger.info("Valid: [{:3d}/{}] Final result {}".format(
            epoch + 1, config.trainer.num_epochs, self.valid_meters))

        return self.valid_meters
        # if self.cfg.callback.checkpoint.mode: # the more the better, e.g. acc
        #     return self.valid_meters['acc1'].avg
        # else: # the less, the better, e.g. epe
        #     return self.valid_meters['valid_loss'].avg


    def test_one_epoch(self, epoch, dataloader):
        config = self.cfg
        self.valid_meters = AverageMeterGroup()
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for step, (X, y) in enumerate(dataloader):
                if self.debug and step > 1:
                    break
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                bs = X.size(0)

                # forward
                logits = self.model(X)

                # loss
                if isinstance(logits, tuple):
                    logits, aux_logits = logits
                    aux_loss = self.loss_fn(aux_logits, y)
                else:
                    aux_loss = 0.
                loss = self.loss_fn(logits, y)
                if config.model.aux_weight > 0:
                    loss = loss + config.model.aux_weight * aux_loss

                # post-processing
                y_true.append(y.cpu().detach())
                y_pred.append(logits.cpu().detach())

                accuracy = metrics(logits, y, topk=(1, 3))
                self.valid_meters.update(accuracy)
                self.valid_meters.update({'valid_loss':loss.item()})
                if step % config.logger.log_frequency == 0 or step == len(dataloader) - 1:
                    self.logger.info(
                        "Test: [{:3d}/{}] Step {:03d}/{:03d} {}".format(
                            epoch + 1, config.trainer.num_epochs, step, len(dataloader) - 1, self.valid_meters))

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        self.valid_report = parse_preds(np.array(y_true.detach().cpu().numpy()), np.array(y_pred.detach().cpu().numpy()))
        self.valid_report['acc1'] = self.valid_meters['acc1'].avg
        self.valid_report['epoch'] = epoch
        self.logger.info(self.valid_report['cls_report'])
        self.logger.info(self.valid_report['covid_auc'])
        torch.save(self.valid_report, os.path.join(config.logger.path, 
                                                   f'best_epoch{epoch}_valid_report.pth'))
        # if self.enable_writter  and epoch > 0:
        #     self.writter.add_scalar("loss/valid", self.valid_meters['valid_loss'].avg, global_step=epoch)
        #     self.writter.add_scalar("acc1/valid", self.valid_meters['acc1'].avg, global_step=epoch)
        #     self.writter.add_scalar("acc3/valid", self.valid_meters['acc3'].avg, global_step=epoch)

        self.logger.info("Test: [{:3d}/{}] Final result {}".format(
            epoch + 1, config.trainer.num_epochs, self.valid_meters))

        return self.valid_meters

    def resume(self, mode=True):
        self.best_metric = -999
        path = self.cfg.model.resume_path
        if path:
            assert os.path.exists(path), "{} does not exist".format(path)
            ckpt = torch.load(path)
            try:
                self.model.load_state_dict(ckpt['model_state_dict'])
            except:
                self.logger.info('Loading from DataParallel model...')
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in ckpt['model_state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.logger.info('Resuming training from epoch {}'.format(self.start_epoch))
            self.best_metric = ckpt['best_metric']
            self.start_epoch = ckpt['epoch'] + 1

        for callback in self.callbacks:
            if isinstance(callback, CheckpointCallback):
                if self.best_metric == -999:
                    self.best_metric = callback.best_metric
                else:
                    callback.best_metric = self.best_metric

    def generate_callbacks(self):
        '''
        Args:
            func: a function to generate other callbacks, must return a list
        Return:
            a list of callbacks.
        '''
        self.ckpt_callback = CheckpointCallback(
            checkpoint_dir=self.cfg.logger.path,
            name='best_retrain.pth',
            mode=self.cfg.callback.checkpoint.mode)
        self.cam_callback = CAMCallback(self.cfg)
        callbacks = [self.ckpt_callback, self.cam_callback]
        return callbacks

    def model_size(self, name='size'):
        assert name in ['size', 'flops']
        size = self.cfg.input.size
        if self.cfg.dataset.is_3d:
            input_size = (1,1,self.cfg.dataset.slice_num,*size)
        else:
            input_size = (1,3,*size)
        return flops_size_counter(self.model, input_size)[name]