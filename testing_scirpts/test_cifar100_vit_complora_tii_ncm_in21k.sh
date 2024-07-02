CUDA_VISIBLE_DEVICES=2 \
python main.py \
        cifar100_complora_tii_ncm \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_epoch_complora_tii_ncm_in21k_seed42/11-07-12-17-35_lr_0.003_bs_64_epochs_10_rank_16_reg_0.01_lora_reg0.1\
        --seed 42 \
        --epochs 10 \
        --n_centroids 50 \
        --lr 0.003 \
        --opt adam \
        --ortho 0.1 \
        --reg_cr 1.0 \
        --omega_lr_scale 0.1 \
        --lora_type comp \
        --lora_depth 5 \
        --num_tasks 10 \
        --lora_rank 16 \
        --num_candidates 2 \
        --sim_type cosine \
        --replay_reg 1 \
        --sample_number 10 \
        --eval