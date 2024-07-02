CUDA_VISIBLE_DEVICES=2 \
python main.py \
        cifar100_complora \
        --model vit_base_patch16_224_in21k \
        --original_model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_complora_seed42_in21k/12-20-21-58-42_lr_0.003_bs_64_epochs_10_rank_32_reg_0.0_lora_reg0.1 \
        --seed 42 \
        --epochs 10 \
        --n_centroids 10 \
        --crct_epochs 40 \
        --ca_lr 0.005 \
        --lr 0.003 \
        --opt adam \
        --reg 0.0 \
        --lora_reg 0.1 \
        --omega_lr_scale 0.1 \
        --sched constant \
        --decay-epochs 5 \
        --lora_type comp \
        --lora_depth 8 \
        --num_tasks 10 \
        --trained_original_model ./output_old/cifar_vit_epoch_complora_tii_in21k_seed42/10-30-19-34-21_lr_0.003_bs_64_epochs_10_rank_8_reg_0.01_lora_regNone \
        --lora_rank 32 \
        --eval
