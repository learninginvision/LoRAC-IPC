for ortho in 0.5 1 5
do
CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29545' \
        --use_env main.py \
        imr_compilora_mask_tii \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir "./imr_output/sup-21k/lora_mask_tii/all" \
        --seed 42 \
        --epochs 50 \
        --lr 0.01 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_fc1 \
        --lora_fc2 \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 10 \
        --lora_rank 32 \
        --ortho $ortho \
        --omega_lr_scale 0.1 \
        --threshold 0.05 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 
done