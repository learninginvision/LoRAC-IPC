CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29525' \
        --use_env main.py \
        five_datasets_compilora_mask_tii \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./output/sup-21k*/lora_mask/all/05-14-19-08-40_lr_0.006_bs_64_epochs_10_rank_16_ortho_0.5_omega_lr_scale1.0_TH0.0" \
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
        --threshold 0.0 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --eval