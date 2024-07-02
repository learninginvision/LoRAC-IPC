# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29510' \
#         --use_env main.py \
#         cifar100_continual_lora \
#         --model vit_base_patch16_224 \
#         --original_model vit_base_patch16_224 \
#         --batch-size 96 \
#         --data-path ./datasets \
#         --output_dir ./output/cifar_vit_continual_lora_seed42 \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.01 \
#         --lora_type continual \
#         --num_tasks 10 \
#         --lora_rank 8
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29516' \
#         --use_env main.py \
#         cifar100_emalora \
#         --model vit_base_patch16_224 \
#         --original_model vit_base_patch16_224 \
#         --batch-size 24 \
#         --data-path ./datasets \
#         --output_dir ./output/cifar_vit_emalora_seed42 \ 
#         --seed 42 \
#         --epochs 50 \
#         --crct_epochs 30 \
#         --lora_momentum 0.01 \
#         --lr 0.01 \
#         --lora_type ema \
#         --num_tasks 10 \
#         --trained_original_model ./output/cifar_vit_continual_lora_seed42 \
#         --lora_rank 8


CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port='29522' \
        --use_env main.py \
        cifar100_emalora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_emalora_seed40/10-28-16-39-28_lr_0.03_bs_64_epochs_20_rank_2_reg_0.0_lora_reg100.0 \
        --seed 40 \
        --epochs 20 \
        --n_centroids 10 \
        --crct_epochs 40 \
        --lora_out \
        --lora_fc1 \
        --lora_fc2 \
        --lr 0.03 \
        --unscale_lr false \
        --sched step \
        --decay-epochs 10 \
        --lora_lr_scale 0.1 \
        --min_lora_lr \
        --reg 0.0 \
        --lora_momentum 0.1 \
        --lora_reg 100.0 \
        --lora_type ema \
        --lora_depth 5 \
        --num_tasks 10 \
        --trained_original_model ./output/cifar_vit_complora_tap_zhao \
        --lora_rank 2 \
        --eval