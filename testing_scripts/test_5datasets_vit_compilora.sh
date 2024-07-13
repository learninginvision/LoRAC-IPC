CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        five_datasets_compilora_mask \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir "./output/exp_name/" \
        --seed 42 \
        --epochs 10 \
        --lr 0.006 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_fc1 \
        --lora_fc2 \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 10 \
        --lora_rank 16 \
        --ortho 0.5 \
        --omega_lr_scale 1.0 \
        --threshold 0.05\
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --eval