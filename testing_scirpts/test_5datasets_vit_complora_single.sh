CUDA_VISIBLE_DEVICES=0 \
python main.py \
        five_datasets_complora_single \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/5datasets_vit_epoch_complora_single_in21k_seed42/03-11-23-47-35_lr_0.003_bs_64_epochs_10_rank_12_reg_0.01_lora_reg0.1 \
        --seed 42 \
        --lora_type comp \
        --lora_depth 10 \
        --num_tasks 5 \
        --lora_rank 12 \
        --lora_reg 0.1 \
        --batch_wise \
        --eval