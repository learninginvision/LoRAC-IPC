CUDA_VISIBLE_DEVICES=1 \
python main.py \
        imr_complora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path ./datasets \
        --output_dir ./output/imr_vit_epoch_complora_seed42/11-10-00-33-30_lr_0.01_bs_24_epochs_40_rank_32_reg_none_lora_reg1.0 \
        --seed 42 \
        --epochs 40 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --lr 0.01 \
        --omega_lr_scale 0.1 \
        --opt adam \
        --ortho 0.1 \
        --lora_type comp \
        --lora_depth 12 \
        --num_tasks 10 \
        --lora_rank 32 \
        --lora_fc1 \
        --lora_sim_type l2 \
        --trained_original_model ./output/imr_vit_epoch_complora_tii_seed42 \
        --lora_reg 1.0 \
        --eval \
        --batch_wise