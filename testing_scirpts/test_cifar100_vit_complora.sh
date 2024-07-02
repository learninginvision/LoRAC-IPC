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
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29514' \
        --use_env main.py \
        cifar100_complora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 48 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_epoch50_complora_seed42/10-25-02-03-55_lr_0.0025_bs_48_epochs_50_rank_8_reg_0.01_lora_regNone \
        --seed 42 \
        --epochs 30 \
        --crct_epochs 30 \
        --lr 0.0025 \
        --ortho 0.1 \
        --lora_type comp \
        --num_tasks 10 \
        --trained_original_model ./output/cifar_vit_complora_ling \
        --lora_rank 8 \
        --eval