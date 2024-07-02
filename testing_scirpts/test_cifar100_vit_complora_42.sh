CUDA_VISIBLE_DEVICES=5 \
python main.py \
        cifar100_complora \
        --model vit_base_patch16_224 \
        --original_model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_epoch_complora_seed42 \
        --seed 42 \
        --epochs 10 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --lr 0.03 \
        --opt adam \
        --ortho 0.1 \
        --lora_reg 1.0 \
        --omega_lr_scale 0.1 \
        --lora_type comp \
        --lora_fc1 \
        --lora_depth 12 \
        --num_tasks 10 \
        --trained_original_model ./output/cifar_vit_epoch_complora_tii_seed42 \
        --lora_rank 12 \
        --lora_sim_type l2 \
        --eval \
        --batch_wise