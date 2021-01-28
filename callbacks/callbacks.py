import os
import sys
import glob
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from utils import MyLogger, CAM3D

__all__ = [
    'Callback',
    'CheckpointCallback',
    'RelevanceCallback',
    'CAMCallback'
]

class Callback:
    
    def __init__(self):
        self.model = None
        self.mutator = None
        self.trainer = None

    def build(self, model, mutator, trainer):
        self.model = model
        self.mutator = mutator
        self.trainer = trainer

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_begin(self, epoch):
        pass

    def on_batch_end(self, epoch):
        pass

class CAMCallback(Callback):
    def __init__(self, cfg):
        super(CAMCallback, self).__init__()
        self.cfg = cfg

    def on_epoch_end(self, epoch):
        if self.trainer.cur_meters.meters['save_metric'].avg >= 0.8 and self.cfg.cam.enable:
            self.cfg.defrost()
            self.cfg.cam.save_path = os.path.join(self.cfg.logger.path, f'cam_results_epoch{epoch}')
            self.cfg.freeze()
            if not os.path.exists(self.cfg.cam.save_path):
                os.makedirs(self.cfg.cam.save_path)
            cam = CAM3D(self.cfg, self.model)
            cam.run()

class RelevanceCallback(Callback):
    def __init__(self, save_path, filename):
        '''
        record the performance of the search stage and evaluation stage, 
        then get the relevance
        '''
        super(RelevanceCallback, self).__init__()
        self.save_path = save_path
        self.filename = filename
        os.makedirs(self.save_path, exist_ok=True)
        self.filename = os.path.join(save_path, filename)
        self.records = {}

    def on_epoch_end(self, epoch):
        if self.trainer.valid_meters:
            search_meters = self.trainer.train_meters.meters['save_metric'].avg
            valid_meters = self.trainer.valid_meters.meters['save_metric'].avg
            self.records[epoch] = {
                'search_meters': search_meters,
                'valid_metric': valid_meters,
                'distance': valid_meters - search_meters
            }
            with open(self.filename, 'w') as fio:
                fio.write('Epoch,Search,Valid,Distance\n')
                for e in sorted(self.records):
                    search_meters, valid_meters, distance = [self.records[e][k] for k in self.records[e]]
                    fio.write(f"{e},{search_meters:.4f},{valid_meters:.4f},{distance:.4f}\n")

class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir, name, mode=True):
        super(CheckpointCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.mode = self.parse_mode(mode)
        self.name = name
        self.warn_flag = True
        if self.mode: # the more the better, e.g. acc
            self.best_metric = -1. * np.inf
            print("The metric is bigger, the better, like accuracy")
        else: # the less, the better, e.g. epe
            self.best_metric = np.inf
            print("The metric is lower, the better, like loss")

    def parse_mode(self, mode):
        if mode.lower() == 'min':
            return False
        elif mode.lower() == 'max':
            return True
        elif isinstance(mode, bool):
            return mode
        else:
            print(f'''Mode only supports [True / max, False / min], but got {mode, type(mode)}''')
            raise NotImplementedError

    def update_best_metric(self, cur_metric):
        self.is_best = False
        self.cur_metric = cur_metric
        if self.mode:
            if cur_metric > self.trainer.best_metric:
                self.trainer.best_metric = cur_metric
                self.is_best = True
        else:
            if cur_metric < self.trainer.best_metric:
                self.trainer.best_metric = cur_metric
                self.is_best = True
        self.best_metric = self.trainer.best_metric
        
    def on_epoch_end(self, epoch):
        self.save(epoch, is_last=True) # save last ckpt
        if self.is_best: # save best ckpt
            self.save(epoch)

    def save(self, epoch, is_last=False):
        model_state_dict = self.get_state_dict(self.model) if hasattr(self.trainer, 'model') else "No state_dict"
        mutator_state_dict = self.get_state_dict(self.mutator) if hasattr(self.trainer, 'mutator') else "No state_dict"
        optimizer_state_dict = self.get_state_dict(self.trainer.optimizer) if hasattr(self.trainer, 'optimizer') else "No state_dict"
        lr_scheduler_state_dict = self.get_state_dict(self.trainer.lr_scheduler) if hasattr(self.trainer, 'lr_scheduler') else "No state_dict"
        if is_last:
            name = 'last.pth'
            metric = self.cur_metric
            dest_path = os.path.join(self.checkpoint_dir, name)
            if os.path.exists(dest_path):
                os.remove(dest_path)
        else:
            metric = self.trainer.best_metric
            best_ckpts = glob.glob(os.path.join(self.checkpoint_dir, 'best*'))
            if best_ckpts:
                for best_ckpt in best_ckpts:
                    os.remove(best_ckpt)
            name = f'best_{self.trainer.best_metric:.6f}_epoch{epoch}.pth'
            dest_path = os.path.join(self.checkpoint_dir, name)
            valid_report = getattr(self.trainer, 'valid_report', None)
            if valid_report:
                report_path = os.path.join(self.checkpoint_dir, f"best_epoch{epoch}_valid_report.pth")
                torch.save(valid_report, report_path)
        ckpt = {
            'model_state_dict': model_state_dict, # model state_dict
            'mutator_state_dict': mutator_state_dict, # mutator state_dict
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'epoch': epoch,
            'best_metric': metric
        }
        torch.save(ckpt, dest_path)
        model = self.model
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        torch.save(model, dest_path.replace('.pth', '_fullmodel.pt'))
        self.trainer.logger.info(f"Saving model to {dest_path} at epoch {epoch} with metric {metric:.6f}")
        self.warn_flag = False

    def get_state_dict(self, module):
        try:
            if isinstance(module, nn.DataParallel):
                state_dict = module.module.state_dict()
            else:
                state_dict = module.state_dict()
        except:
            if self.warn_flag:
                self.trainer.logger.info(f"{module} has no attribution of 'state_dict'")
            state_dict = 'No state_dict'
        return state_dict
