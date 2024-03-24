"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import wandb
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description="Training",add_help=False)

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+"
    )
    parser.add_argument("--job_name",default="minigpt_spatial_coco_control",type=str)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # args = parser.parse_args()




    return parser


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.

    print("start!!!")
    job_id = now()
    args = parse_args().parse_args()


    print("0000")
    cfg = Config(args)

    if 'LOCAL_RANK' not in os.environ:
        print("not in the os")
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    print("111")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    print("local rank",local_rank)

    dist.init_process_group(backend='nccl', init_method='env://')
    
    num_nodes = dist.get_world_size()
    print(f"Number of nodes: {num_nodes}")


    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after in
    # it_distributed_mode() to only log on master.
    setup_logger()

    
    wandb.login()
    # print(wandb.run)


    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    if cfg.run_cfg.rank == 0:
        print("project name", args.job_name)

        wandb.init(project="minigpt4-spatial",name=args.job_name)

        wandb.config = {"learning_rate": 0.0001, "epochs": 100, "batch_size": 8}
        wandb.watch(model)

    # print('+++++++++++++++++')
    # print(type(model))
    # print('+++++++++++++++++')
    # print(model)
    # print('+++++++++++++++++')
    # print(model.super().device)
    # print('+++++++++++++++++')
    # print(model.device)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
