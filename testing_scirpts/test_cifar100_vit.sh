#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29501' \
	--use_env main.py \
	cifar100_hideprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 1 \
	--epochs 128 \
	--data-path ./datasets \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed 40 \
	--prompt_momentum 0.01 \
	--reg 0.1 \
	--length 5 \
	--sched step \
	--larger_prompt_lr \
	--trained_original_model ./output/cifar_vit_complora_ling \
	--output_dir ./output/cifar100_vit_pe_seed40/10-24-08-44-23_lr_0.03_bs_48_epochs_20_rank_none_reg_0.1 \
	--eval