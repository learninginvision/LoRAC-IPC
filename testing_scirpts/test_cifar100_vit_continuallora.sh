CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29520' \
        --use_env main.py \
        cifar100_continual_lora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 96 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_continual_lora_seed42 \
        --seed 42 \
        --epochs 1 \
        --lr 0.01 \
        --lora_type continual \
        --num_tasks 10 \
        --lora_rank 8 \
        --eval