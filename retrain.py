
import argparse
import os
from argparse import ArgumentParser

import numpy as np
import torch

import nni
from networks import build_model
from configs import get_cfg, build_config
from evaluator import build_evaluator
from nni.nas.pytorch.fixed import apply_fixed_architecture
from utils import MyLogger, CAM3D


def setup_cfg(args):
    cfg = get_cfg()
    cfg = build_config(cfg, args.config_name)
    cfg.merge_from_file((args.config_file))
    cfg.merge_from_list(args.opts)
    if cfg.model.resume_path:
        cfg.logger.path = os.path.dirname(cfg.model.resume_path)
    else:
        index = 0
        path = os.path.dirname(args.arc_path)+'_retrain_{}'.format(index)
        while os.path.exists(path):
            index += 1
            path = os.path.dirname(args.arc_path)+'_retrain_{}'.format(index)
        cfg.logger.path = path
    cfg.logger.log_file = os.path.join(cfg.logger.path, 'log_retrain.txt')
    os.makedirs(cfg.logger.path, exist_ok=True)
    cfg.freeze()
    SEED = cfg.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return cfg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--config_file", default='./configs/retrain.yaml', type=str)
    parser.add_argument("--train_epochs", default=1, type=int)
    parser.add_argument('--test_only', action='store_true', help='')
    parser.add_argument('--cam_only', action='store_true', help='')
    parser.add_argument("-cn", "--config_name", default="ctconfig", type=str, help="specify add which type of config")
    parser.add_argument("--arc_path", default="", type=str,
                        help="./outputs/checkpoint_0 or ./outputs/checkpoint_0/epoch_1.json")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config_file = args.config_file
    if os.path.isdir(args.arc_path) and args.arc_path[-1] != '/':
        args.arc_path += '/'
    arc_path = args.arc_path
    
    assert config_file and arc_path, f"please check whether {config_file} and {arc_path} exists"

    # configuration
    cfg = setup_cfg(args)
    with open(os.path.join(cfg.logger.path, 'retrain.yaml'), 'w') as f:
        f.write(str(cfg))
    cfg.update({'args': args})
    logger = MyLogger(__name__, cfg).getlogger()
    logger.info('args:{}'.format(args))

    if args.cam_only:
        model = build_model(cfg)
        apply_fixed_architecture(model, args.arc_path)
        cam = CAM3D(cfg, model)
        cam.run()
    else:
        evaluator = build_evaluator(cfg)
        if os.path.isdir(arc_path):
            best_arch_info = evaluator.compare()
            evaluator.run(best_arch_info['arc'])
        elif os.path.isfile(arc_path):
            evaluator.run(arc_path, validate=True, test=args.test_only)
        else:
            logger.info(f'{arc_path} is invalid.')
