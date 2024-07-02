CUDA_VISIBLE_DEVICES=3 \
python main.py \
        cifar100_complora_ncm \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_epoch_complora_ncm_in21k_seed42/11-05-15-37-06_lr_0.003_bs_128_epochs_20_rank_32_reg_0.01_lora_reg0.1 \
        --seed 42 \
        --epochs 10 \
        --n_centroids 5 \
        --crct_epochs 40 \
        --ca_lr 0.005 \
        --lr 0.003 \
        --opt adam \
        --ortho 0.1 \
        --lora_type comp \
        --lora_depth 5 \
        --num_tasks 10 \
        --lora_rank 32 \
        --num_candidates 1 \
        --sim_type l2 \
        --eval