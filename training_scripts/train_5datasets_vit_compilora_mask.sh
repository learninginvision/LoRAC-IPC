for threshold in 0.1 0.2 
do
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        five_datasets_compilora_mask \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir ./output/5-datasets/sup-21k/lora_mask/qkvo \
        --seed 42 \
        --epochs 5 \
        --lr 0.01 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 5 \
        --lora_rank 8 \
        --ortho 1.0 \
        --omega_lr_scale 1.0 \
        --threshold $threshold
done