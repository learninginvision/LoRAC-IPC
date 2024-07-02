CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29525' \
        --use_env main.py \
        imr_compilora_mask_tii \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir "./imr_output/sup-21k/lora_mask_tii/all/05-16-20-17-56_lr_0.02_bs_64_epochs_50_rank_64_ortho_0.01_omega_lr_scale0.5_TH0.05" \
        --seed 42 \
        --epochs 50 \
        --lr 0.02 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_fc1 \
        --lora_fc2 \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 10 \
        --lora_rank 64 \
        --ortho 0.01 \
        --omega_lr_scale 0.5 \
        --threshold 0.05 \
        --first_session_lr 0.0 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --eval