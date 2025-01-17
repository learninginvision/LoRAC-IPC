import torch
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import time, datetime, os, sys, random, numpy as np
from datasets import build_continual_dataloader
from engines.comp_ilora_mask_engine import train_and_evaluate, evaluate_till_now,_compute_mean, train_task_adaptive_prediction
import vits.comp_ilora_mask_allw_vision_transformer as comp_ilora_vision_transformer


def train(args):
    device = torch.device(args.device)
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)
    
    str2bool = {'T': True, 'F': False}
    args.lora_qkv = [str2bool[x.upper()] for x in args.lora_qkv]

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

    # all backbobe parameters are frozen for original vit model
    if args.freeze:
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
            # if 'lora_O' in n:
            #     p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            
            _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                          class_mask=class_mask[task_id], args=args)
            
            if task_id > 0 and not args.not_train_ca:
               train_task_adaptive_prediction(model, args, device, class_mask, task_id)


            _ = evaluate_till_now(model, data_loader, device, task_id, class_mask, target_task_map, acc_matrix, args, )

        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0
    
    
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print('training {}'.format(n))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    base_params = [p for name, p in model_without_ddp.named_parameters() if 'lora_O' not in name and p.requires_grad == True]
    base_omega_params = [p for name, p in model_without_ddp.named_parameters() if 'lora_O' in name and p.requires_grad == True]

    base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
    base_omega_params = {'params': base_omega_params, 'lr': args.lr * args.omega_lr_scale, 'weight_decay': args.weight_decay}

    network_params = [base_params, base_omega_params]
    optimizer = create_optimizer(args, network_params)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, criterion, data_loader, data_loader_per_cls,
                       optimizer, lr_scheduler, device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")