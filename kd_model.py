import torch
import torchline
import torch.nn as nn
import torch.nn.functional as F

def load_kd_model(cfg):
    kd_cfg = torchline.config.get_cfg()
    kd_cfg.model.name = cfg.kd.model.name
    kd_cfg.model.classes = cfg.model.classes

    ckpt = torch.load(cfg.kd.model.path)
    model_state_dict = {k.replace('model.model', 'model'):v for k, v in ckpt['state_dict'].items()}
    model = torchline.models.build_model(kd_cfg)
    model.load_state_dict(model_state_dict)
    return model

def loss_fn_kd(outputs, teacher_outputs, cfg):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = cfg.alpha
    T = cfg.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)

    return KD_loss