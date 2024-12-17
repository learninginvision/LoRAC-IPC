import sys
sys.path.append('..')

from numpy import mean
import torch
from datasets import build_continual_dataloader
import argparse
import os
import yaml
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from timm.models import create_model
# import vits.comp_ilora_vision_transformer as comp_ilora_vision_transformer
# import vits.comp_ilora_mask_vision_transformer as comp_ilora_vision_transformer
import vits.comp_ilora_mask_allw_vision_transformer as comp_ilora_vision_transformer
from copy import deepcopy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# save args to config file(.yaml)
def save_args_to_yaml(args, output_file):
    args_dict = OrderedDict(vars(args))
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)

# load args from config file(.yaml)
def load_args_from_yaml(input_file):
    with open(input_file, 'r') as yaml_file:
        args_dict = yaml.load(yaml_file, Loader=yaml.Loader)
    # print(args_dict)
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args

exp_path = '//data_8T/ling/comp-lorav7/cifar100_output_tii/sup-21k/all/05-14-00-44-25_lr_0.02_bs_64_epochs_20_rank_32_ortho_1.0_omega_lr_scale0.1_TH0.05'
config_path = os.path.join(exp_path, 'config.yaml')

args = load_args_from_yaml(config_path)
print(args)
print(args.output_dir)

data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)

str2bool = {'T': True, 'F': False}
args.lora_qkv = [str2bool[x.upper()] for x in args.lora_qkv]
device = torch.device('cuda:0')

print(f"Creating model: {args.model}")
model = create_model(args.model,
                        pretrained=args.pretrained,
                        num_classes=args.nb_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=None,
                        lora=True, 
                        lora_type=args.lora_type,
                        rank=args.lora_rank, 
                        lora_pool_size=args.size,
                        lora_qkv=args.lora_qkv,
                        lora_out=args.lora_out,
                        lora_fc1=args.lora_fc1,
                        lora_fc2=args.lora_fc2,
                        lora_depth=args.lora_depth,
                        beta1=args.ilora_beta1,
                        beta2=args.ilora_beta2,
                        )
model.to(device)


checkpoint_path = os.path.join(exp_path, f'checkpoint/task{args.num_tasks}_checkpoint.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])

lora_mask = model.lora_layer.get_lora_mask()

total_train_param = 0

for key, value in lora_mask.items():
    
    print(f'{key}: {value}')
    
    if 'q' in key or 'k' in key or 'v' in key:
        lora_mask = value[:args.num_tasks]
        train_lora_number = np.count_nonzero(lora_mask)
        train_param = train_lora_number * model.embed_dim * args.lora_rank * 2
        total_train_param += train_param
    elif 'out' in key:
        o_lora_mask = value[:args.num_tasks]
        train_lora_number = np.count_nonzero(o_lora_mask)
        train_param = train_lora_number * model.embed_dim * args.lora_rank * 2
        total_train_param += train_param
    elif 'fc1' in key:
        fc1_lora_mask = value[:args.num_tasks]
        train_lora_number = np.count_nonzero(fc1_lora_mask)
        train_param = train_lora_number * (model.embed_dim * args.lora_rank  + args.lora_rank * model.embed_dim * 4)
        total_train_param += train_param
    else:
        fc2_lora_mask = value[:args.num_tasks]
        train_lora_number = np.count_nonzero(fc2_lora_mask)
        train_param = train_lora_number * (args.lora_rank * model.embed_dim * 4 + model.embed_dim * args.lora_rank)
        total_train_param += train_param
    
    print(f'train parameter: {train_param}')

print(f'total train parameter: {total_train_param}')

per_task_train_param = total_train_param / args.num_tasks

print(f'per task train parameter: {per_task_train_param / 1e6:.2f}M')
        
    
    # print(f'{key}: {value}')
    # print(f'{key} sparsity: {1 - value.sum() / value.numel()}')




