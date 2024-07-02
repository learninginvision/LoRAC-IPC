CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        cifar100_compilora \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir ./output/sup-21k/lora_mask_qkvo \
        --seed 42 \
        --epochs 20 \
        --lr 0.005 \
        --opt adam \
        --lora_fc1 \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 10 \
        --lora_rank 12 \
        --ortho 1.0 \
        --lora_reg 10 \
        --omega_lr_scale 1.0 