# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from argparse import ArgumentParser

import numpy as np
import torch

from configs import get_cfg, build_config
from trainer import build_trainer
from utils import MyLogger


def main(args):
    #### set up cfg ####
    # default cfg
    cfg = get_cfg()

    # add registered cfg
    cfg = build_config(cfg, args.config_name)
    cfg.setup(args)

    #### seed ####
    SEED = cfg.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #### start searching ####
    trainer = build_trainer(cfg)
    try:
        trainer.train(cfg.trainer.validate_always)
        if not cfg.trainer.validate_always:
            trainer.test()
    except (KeyboardInterrupt, ) as e:
        if isinstance(e, KeyboardInterrupt):
            print(f'Capture KeyboardInterrupt event ...')
        else:
            print(str(e))
    finally:
        trainer.save_cfg()


if __name__ == "__main__":
    parser = ArgumentParser("Search")
    parser.add_argument("--config_file", default="./configs/search.yaml", type=str)
    parser.add_argument("-cn", "--config_name", default="ctconfig", type=str, help="specify add which type of config")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)