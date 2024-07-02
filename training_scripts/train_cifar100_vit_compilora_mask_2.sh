for threshold in 0.1 0.05
do
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29535' \
        --use_env main.py \
        cifar100_compilora_mask \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir ./output/sup-21k/lora_mask/all \
        --seed 42 \
        --epochs 20 \
        --lr 0.01 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_fc1 \
        --lora_fc2 \
        --lora_type compi \
        --lora_depth 10 \
        --num_tasks 10 \
        --lora_rank 20 \
        --ortho 10.0 \
        --omega_lr_scale 1.0 \
        --threshold $threshold
done