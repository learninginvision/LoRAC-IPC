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
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29519' \
        --use_env main.py \
        cifar100_hidelora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 96 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_hidelora_seed42/10-22-11-40-17_lr_0.03_bs_96_epochs_50_rank_8_lo_none \
        --seed 42 \
        --epochs 50 \
        --lr 0.03 \
        --lora_type hide \
        --num_tasks 10 \
        --trained_original_model ./output/cifar_vit_complora_ling \
        --lora_rank 8 \
        --eval