import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import torch.nn.functional as F
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
from copy import deepcopy

global cls_mean
global cls_cov

global tii_mean
global tii_cov
global tii_shrink_cov

cls_mean = dict()
cls_cov = dict()

tii_mean = dict()
tii_cov = dict()
tii_shrink_cov = dict()


def train_one_epoch(model: torch.nn.Module, model_without_ddp: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('Loss_OR', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
            
        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        # base criterion (CrossEntropyLoss)
        loss = criterion(logits, target)  

        # TODO add lora orth loss
        loss_or = model_without_ddp.get_loss_ortho(task_id=task_id, device=device, args=args)
        
        loss = loss + loss_or

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        if args.distributed:
            model_without_ddp.lora_layer.balance_ipt(task_id=task_id, device=device, world_size=args.world_size)
        
        model_without_ddp.lora_layer.update_ipt(task_id=task_id)
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Loss_OR=loss_or.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def normalize_cov(cov_mat):
#     norm_cov_mat = {}
#     for key, cov in cov_mat.items():
#         sd = torch.sqrt(torch.diag(cov))  # standard deviations of the variables
#         cov = cov/(torch.matmul(sd.unsqueeze(1),sd.unsqueeze(0)))
#         norm_cov_mat[key] = cov

#     return norm_cov_mat



def normalize_cov(cov_mat):
    norm_cov_mat = {}
    for key, cov_list in cov_mat.items():
        normalize_cov_list = []
        for cov in cov_list:
            sd = torch.sqrt(torch.diag(cov))  # standard deviations of the variables
            normalize_cov = cov/(torch.matmul(sd.unsqueeze(1),sd.unsqueeze(0)))
            normalize_cov_list.append(normalize_cov)
            
        norm_cov_mat[key] = normalize_cov_list

    return norm_cov_mat

# def _mahalanobis(dist, cov=None, preprocess=False):
#     if cov is None:
#         cov = torch.eye(768)
#     if not preprocess:
#         inv_covmat = torch.pinverse(cov)  # pseudo-inverse of an invertible matrix is same as its inverse
#     else:
#         inv_covmat = cov
#     left_term = torch.matmul(dist, inv_covmat)
#     mahal = torch.matmul(left_term, dist.T) 
#     return torch.diagonal(mahal, 0)

def _mahalanobis(dist, inv_cov_list=None, preprocess=False):
    assert inv_cov_list is not None, 'covariance matrix is not provided'

    min_mahal = torch.ones(dist.shape[0], dtype=torch.float32).to(dist.device) * float('inf')
    
    for i in range(len(inv_cov_list)):
        if preprocess:
            inv_cov = inv_cov_list[i]
        else:
            inv_cov = torch.pinverse(inv_cov_list[i])

        left_term = torch.matmul(dist[:,i,:], inv_cov)
        mahal = torch.einsum('ij,ij->i', left_term, dist[:,i,:])
        min_mahal = torch.min(min_mahal, mahal)
        
    return min_mahal

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, All_task_matrix=None, Co_task_matrix=None,
             device=None, dataset_id=-1, model_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Dataset {}]'.format(dataset_id + 1)

    # switch to evaluation mode
    model.eval()

    up2current_mask = []
    for id in range(model_id + 1):
        up2current_mask.extend(class_mask[id])

    norm_cov_mat = normalize_cov(tii_shrink_cov)

    inv_cov_mat = {}
    for key, cov_list in norm_cov_mat.items():
        inv_cov_list = []
        for cov in cov_list:
            inv_cov_list.append(torch.pinverse(cov))
        inv_cov_mat[key] = inv_cov_list

    # norm_inv_cov_mat = {key: torch.pinverse(cov) for key, cov in norm_cov_mat.items()}

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            features = F.normalize(model(input, task_id=0)['features'])

            predictions = []
            maha_dist = []

            for cls_id in up2current_mask:
                mean = tii_mean[cls_id]
                inv_cov = inv_cov_mat[cls_id]
                # dist = features - mean
                dist = features.unsqueeze(1).repeat(1, mean.shape[0], 1) - mean.unsqueeze(0)
                maha_dist.append(_mahalanobis(dist, inv_cov, preprocess=True))

            maha_dist = torch.stack(maha_dist, dim=1)
            predictions = torch.argmin(maha_dist, dim=1)
            lora_id = torch.tensor([target_task_map[v.item()] for v in predictions], device=device)
            
            if args.batch_wise:
                # vote the most lora_id
                idx, idx_count = torch.unique(lora_id, return_counts=True, sorted=True)
                major_idx = torch.argmax(idx_count)
                major_lora_idx = idx[major_idx]
                output = model(input, task_id=int(major_lora_idx.item()))
                logits = output['logits']
            
            else:
                logits = []
                for i in range(input.shape[0]):
                    output = model(input[i].unsqueeze(0), task_id=int(lora_id[i].item()))
                    logits.append(output['logits'])

                logits = torch.cat(logits, dim=0)

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[dataset_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            task_inference_acc = utils.task_inference_accuracy(lora_id.unsqueeze(-1), target, target_task_map)

            _, pred = logits.topk(1, 1, True, True)
            for i in range(input.shape[0]):
                All_task_matrix[lora_id[i].item(), dataset_id] += 1
                if pred[i].item() == target[i].item():
                    Co_task_matrix[lora_id[i].item(), dataset_id] += 1

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    result_str = \
        'Model {model_id} on Dataset {dataset_id}: Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}' \
        .format(model_id=model_id+1, dataset_id=dataset_id+1,
                task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss'])

    if args.output_dir and utils.is_main_process():
        if args.eval and args.batch_wise:
            file_name = 'eval_summary_batchwise.txt'
        elif args.eval:
            file_name = 'eval_summary.txt'
        else:
            file_name = 'summary.txt'

        with open(os.path.join(args.output_dir, file_name), 'a+') as f:
            f.write(result_str + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    All_task_matrix = np.zeros((task_id+1, task_id+1))
    Co_task_matrix = np.zeros((task_id+1, task_id+1))

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'],
                              device=device, dataset_id=i, model_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              All_task_matrix=All_task_matrix, Co_task_matrix=Co_task_matrix, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])

    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix[:, :task_id], axis=1) - acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
    else:
        forgetting = 0.0
        backward = 0.0

    result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    if args.output_dir and utils.is_main_process():
        if args.eval and args.batch_wise:
            file_name = 'eval_summary_batchwise.txt'
        elif args.eval:
            file_name = 'eval_summary.txt'
        else:
            file_name = 'summary.txt'
        with open(os.path.join(args.output_dir, file_name), 'a+') as f:
            f.write(result_str + '\n')

        with open(os.path.join(args.output_dir, 'All_task_matrix.txt'), 'a+') as f:
            f.write(f'Task {task_id+1} All-samples matrix:\n')
            for i in range(task_id + 1):
                f.write('\t\t'.join([str(int(v)) for v in All_task_matrix[:, i]]) + '\n')

            f.write(f'Task {task_id+1} Correct-samples matrix:\n')
            for i in range(task_id + 1):
                f.write('\t\t'.join([str(int(v)) for v in Co_task_matrix[:, i]]) + '\n')

    metric_info = {
        'acc@task': avg_stat[3],
        'acc@1': avg_stat[0],
        'forgetting': forgetting,
        'backward': backward,
    }

    return test_stats, metric_info


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    
    for task_id in range(args.num_tasks):
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            # optimizer = create_optimizer(args, model)
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
        for param_group in optimizer.param_groups:
            print(f"Learning rate: {param_group['lr']}")

        # if model already trained
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, model_without_ddp=model_without_ddp,
                                          criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                          target_task_map=target_task_map, args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)
        model_without_ddp.after_task(task_id=task_id, device=device, args=args)

        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)
        
        _compute_tii_mean(model=model, data_loader=data_loader_per_cls, device=device, class_mask=class_mask[task_id], args=args)

        if args.output_dir and utils.is_main_process():
            # TODO record omegas matrix
            omegas_matrixs = model_without_ddp.lora_layer.get_omegas()
            omega_file_path = args.output_dir + '/omegas_matrix/'
            if not os.path.exists(omega_file_path):
                os.makedirs(omega_file_path)
            for key, value in omegas_matrixs.items():
                with open(os.path.join(omega_file_path, f'{key}_omegas_matrix.txt'), 'a+') as f:
                    tabel_head = 'Omegas\t\t' + '\t'.join([f'Index {i+1}' for i in range(args.num_tasks)])
                    f.write(tabel_head + '\n')
                    for i in range(model_without_ddp.lora_layer.depth):
                        f.write(f'Depth {i+1}\t\t' + '\t'.join([f'{omegas:.5f}' for omegas in value.t()[i, :]]) + '\n')
        
            if args.eval and args.batch_wise:
                file_name = 'eval_summary_batchwise.txt'
            elif args.eval:
                file_name = 'eval_summary.txt'
            else:
                file_name = 'summary.txt'
            with open(os.path.join(args.output_dir, file_name), 'a+') as f:
                f.write('-----------------------------Task {} ----------------------------\n'.format(task_id + 1))

        # current_model_copy = deepcopy(model_without_ddp) # NOTE: for merge lora and model

        if task_id > 0 and not args.not_train_ca:
            pre_ca_test_stats, _ = evaluate_till_now(model=model, data_loader=data_loader,
                                                     device=device,
                                                     task_id=task_id, class_mask=class_mask,
                                                     target_task_map=target_task_map,
                                                     acc_matrix=pre_ca_acc_matrix, args=args)

            train_task_adaptive_prediction(model_without_ddp, args, device, class_mask, task_id)
            
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, 'summary.txt'), 'a+') as f:
                    f.write('Result with task-adaptive prediction:\n'.format(task_id + 1))

        test_stats, metric_info = evaluate_till_now(model=model_without_ddp, data_loader=data_loader,
                                                    device=device,
                                                    task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                                    acc_matrix=acc_matrix, args=args)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'train_stats.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'pre_ca_acc_matrix.txt'), 'a+') as f:
            tabel_head = 'ACC\t\t\t' + '\t'.join([f'Data {i}' for i in range(args.num_tasks)])
            f.write(tabel_head + '\n')
            for i in range(args.num_tasks):
                f.write(f'Model {i}\t\t' + '\t'.join([f'{acc:.2f}' for acc in pre_ca_acc_matrix[:, i]]) + '\n')
        
        with open(os.path.join(args.output_dir, 'acc_matrix.txt'), 'a+') as f:
            tabel_head = 'ACC\t\t\t' + '\t'.join([f'Data {i}' for i in range(args.num_tasks)])
            f.write(tabel_head + '\n')
            for i in range(args.num_tasks):
                f.write(f'Model {i}\t\t' + '\t'.join([f'{acc:.2f}' for acc in acc_matrix[:, i]]) + '\n')

        result_all_path = os.path.join(args.parent_output_dir or args.output_dir, 'result_all.csv')

        if not os.path.exists(result_all_path):
            with open(result_all_path, 'a') as f:
                # f.write
                for key in args.__dict__:
                    f.write(key + ',')
                for key in metric_info.keys():
                    f.write(key + ',')
                f.write('\n')
        
        with open(result_all_path, 'a') as f:
            for key in args.__dict__:
                f.write(str(args.__dict__[key]).replace(',', '_') + ',')
            for key in metric_info.keys():
                f.write(str(metric_info[key]) + ',')
            f.write('\n')

@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=False)['features']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        # features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        if args.distributed:
            features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]
            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)
        else:
            features_per_cls_list = [features_per_cls]

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float32).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float32).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


@torch.no_grad()
def _compute_tii_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, class_mask=None, args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=0, train=False)['features'] # NOTE: fixed task_id=0
            features_per_cls.append(F.normalize(features))
        features_per_cls = torch.cat(features_per_cls, dim=0)

        if args.distributed:
            features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]
            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)
        else:
            features_per_cls_list = [features_per_cls]

        # features_per_cls = torch.cat(features_per_cls_list, dim=0)
        features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()

        from sklearn.cluster import KMeans
        if args.dataset == 'Split-Imagenet-R':
            n_clusters = 2
            # print("n_clusters: 2")
        else:
            n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        kmeans.fit(features_per_cls)
        cluster_lables = kmeans.labels_
        cluster_means = []
        cluster_covs = []
        
        for i in range(n_clusters):
            cluster_data = features_per_cls[cluster_lables == i]
            # cluster_data = features_per_cls
            cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float32).to(device)
            cluster_cov = torch.tensor(np.cov(cluster_data, rowvar=False), dtype=torch.float32).to(device) + (torch.eye(cluster_mean.shape[-1]) * 1e-4).to(device)
            cluster_means.append(cluster_mean)
            cluster_covs.append(cluster_cov)

        # tii_mean[cls_id] = features_per_cls.mean(dim=0)
        tii_mean[cls_id] = torch.stack(cluster_means, dim=0)
        # tii_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(tii_mean[cls_id].shape[-1]) * 1e-4).to(device)
        tii_cov[cls_id] = cluster_covs

        tii_shrink_cov[cls_id] = shrink_cov(tii_cov[cls_id])

def shrink_cov(cov_list):
    shrink_cov_list = []
    for cov in cov_list:
        diag_mean = torch.mean(torch.diag(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag*mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0]).to(cov.device)
        alpha1 = 1
        alpha2 = 0
        shrink_cov = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
        shrink_cov_list.append(shrink_cov)
    return shrink_cov_list


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'lora' not in n]
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float32).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()