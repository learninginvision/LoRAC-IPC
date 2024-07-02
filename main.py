import os.path
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from collections import OrderedDict
import yaml

from datasets import build_continual_dataloader

import utils
import warnings


warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')
    
    if config == 'cifar100_compilora':
        from configs.cifar100_compilora import get_args_parser
        config_parser = subparser.add_parser('cifar100_compilora', help='Split-CIFAR100 CompILora configs')
    elif config == 'cifar100_compilora_mask':
        from configs.cifar100_compilora_mask import get_args_parser
        config_parser = subparser.add_parser('cifar100_compilora_mask', help='Split-CIFAR100 CompILora-mask configs')
    elif config == 'cifar100_compilora_mask_tii':
        from configs.cifar100_compilora_mask_tii import get_args_parser
        config_parser = subparser.add_parser('cifar100_compilora_mask_tii', help='Split-CIFAR100 CompILora-mask-TII configs')
    elif config == 'cifar100_compilora_replay':
        from configs.cifar100_compilora_replay import get_args_parser
        config_parser = subparser.add_parser('cifar100_compilora_replay', help='Split-CIFAR100 CompILora-replay configs')
    elif config == 'five_datasets_complora_single':
        from configs.five_datasets_complora_single import get_args_parser
        config_parser = subparser.add_parser('five_datasets_complora_single', help='five datasets Complora-Single configs')
    elif config == 'five_datasets_compilora':
        from configs.five_datasets_compilora import get_args_parser
        config_parser = subparser.add_parser('five_datasets_compilora', help='five datasets CompILora configs')
    elif config == 'five_datasets_compilora_mask':
        from configs.five_datasets_compilora_mask import get_args_parser
        config_parser = subparser.add_parser('five_datasets_compilora_mask', help='five datasets CompILora-mask configs')
    elif config == 'five_datasets_compilora_mask_tii':
        from configs.five_datasets_compilora_mask_tii import get_args_parser
        config_parser = subparser.add_parser('five_datasets_compilora_mask_tii', help='five datasets CompILora-mask configs')
    elif config == 'imr_compilora_mask_tii':
        from configs.imr_compilora_mask_tii import get_args_parser
        config_parser = subparser.add_parser('imr_compilora_mask_tii', help='Split-ImageNet-R CompILora-mask configs')
    elif config == 'imr_complora':
        from configs.imr_complora import get_args_parser
        config_parser = subparser.add_parser('imr_complora', help='Split-ImageNet-R Complora configs')
    elif config == 'imr_complora_single':
        from configs.imr_complora_single import get_args_parser
        config_parser = subparser.add_parser('imr_complora_single', help='Split-ImageNet-R Complora Single configs')
    elif config == 'cifar100_complora_single':
        from configs.cifar100_complora_single import get_args_parser
        config_parser = subparser.add_parser('cifar100_complora_single', help='Split-CIFAR100 Complora-Single configs')
    elif config == 'cifar100_complora_wram':
        from configs.cifar100_complora_wram import get_args_parser
        config_parser = subparser.add_parser('cifar100_complora_wram', help='Split-CIFAR100 Complora-WRAM configs')
    elif config == 'cifar100_complora_tii':
        from configs.cifar100_complora_tii import get_args_parser
        config_parser = subparser.add_parser('cifar100_complora_tii', help='Split-CIFAR100 Complora-TII configs')
    elif config == 'imr_complora_tii':
        from configs.imr_complora_tii import get_args_parser
        config_parser = subparser.add_parser('imr_complora_tii', help='Split-ImageNet-R Complora-TII configs')
    elif config == 'five_datasets_complora_tii':
        from configs.five_datasets_complora_tii import get_args_parser
        config_parser = subparser.add_parser('five_datasets_complora_tii', help='five datasets Complora-TII configs')
    elif config == 'cifar100_complora':
        from configs.cifar100_complora import get_args_parser
        config_parser = subparser.add_parser('cifar100_complora', help='Split-CIFAR100 Complora configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    args = parser.parse_args()
    args.config = config
    return args


def save_args_to_yaml(args, output_file):
    args_dict = OrderedDict(vars(args))
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)


def main(args):
    utils.init_distributed_mode(args)

    if args.output_dir and not args.eval:
        args.parent_output_dir = args.output_dir
        args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().__format__('%m-%d-%H-%M-%S') + \
                                        f'_lr_{args.lr}_bs_{args.batch_size}_' + \
                                        f"epochs_{args.epochs}_rank_{args.lora_rank if hasattr(args, 'lora_rank') else 'none'}_" + \
                                        f"ortho_{args.ortho if hasattr(args, 'ortho') else 'none'}_" + \
                                        f"omega_lr_scale{args.omega_lr_scale if hasattr(args, 'omega_lr_scale') else None}_" + \
                                        f"TH{args.threshold if hasattr(args, 'threshold') else None}_" + \
                                        f"seed{args.seed if hasattr(args, 'seed') else None}"
                                        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        config_path = os.path.join(args.output_dir, 'config.yaml')
        save_args_to_yaml(args, config_path)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    if 'compilora_mask_tii' in args.config and not args.train_inference_task_only:
        import trainers.compilora_mask_tii_trainer as compilora_mask_tii_trainer
        compilora_mask_tii_trainer.train(args)
    elif 'compilora_mask' in args.config and not args.train_inference_task_only:
        import trainers.compilora_mask_trainer as compilora_mask_trainer
        compilora_mask_trainer.train(args)
    elif 'compilora_drop' in args.config and not args.train_inference_task_only:
        import trainers.compilora_drop_trainer as compilora_drop_trainer
        compilora_drop_trainer.train(args)
    elif 'compilora_replay' in args.config and not args.train_inference_task_only:
        import trainers.compilora_replay_trainer as compilora_replay_trainer
        compilora_replay_trainer.train(args)
    elif 'compilora' in args.config and not args.train_inference_task_only:
        import trainers.compilora_trainer as compilora_trainer
        compilora_trainer.train(args)
    elif 'mllora' in args.config and not args.train_inference_task_only:
        import trainers.mllora_trainer as mllora_trainer
        mllora_trainer.train(args)
    elif 'compadalora' in args.config and not args.train_inference_task_only:
        import trainers.compadalora_trainer as compadalora_trainer
        compadalora_trainer.train(args)
    elif 'compadalora_single' in args.config and not args.train_inference_task_only:
        import trainers.compadalora_single_trainer as compadalora_single_trainer
        compadalora_single_trainer.train(args)
    elif 'complora_tii' in args.config and not args.train_inference_task_only:
        import trainers.complora_tii_trainer as complora_tii_trainer
        complora_tii_trainer.train(args)
    elif 'complora_single' in args.config and not args.train_inference_task_only:
        import trainers.complora_single_trainer as complora_single_trainer
        complora_single_trainer.train(args)
    elif 'complora_wram' in args.config and not args.train_inference_task_only:
        import trainers.complora_wram_trainer as complora_wram_trainer
        complora_wram_trainer.train(args)
    elif 'complora' in args.config and not args.train_inference_task_only:
        import trainers.complora_trainer as complora_trainer
        complora_trainer.train(args)
    elif 'hidelora' in args.config and not args.train_inference_task_only:
        import trainers.hidelora_trainer as hidelora_trainer
        hidelora_trainer.train(args)
    elif 'continual_lora' in args.config:
        import trainers.continual_lora_trainer as continual_lora_trainer
        continual_lora_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    print(args)
    main(args)
