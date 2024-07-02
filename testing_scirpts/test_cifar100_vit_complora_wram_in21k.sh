CUDA_VISIBLE_DEVICES=1 \
python main.py \
        cifar100_complora_wram \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_epoch_complora_wram_seed42 \
        --seed 42 \
        --epochs 20 \
        --n_centroids 100 \
        --crct_epochs 20 \
        --ca_lr 0.005 \
        --omega_lr_scale 0.1 \
        --lr 0.003 \
        --opt adam \
        --ortho 0.1 \
        --lora_type comp \
        --lora_depth 10 \
        --num_tasks 10 \
        --lora_rank 24 \
        --lora_fc1

# CUDA_VISIBLE_DEVICES=1,2 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29551' \
#         --use_env main.py \
#         cifar100_complora_wram \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --data-path ./datasets \
#         --output_dir ./output/cifar_vit_epoch_complora_wram_seed42 \
#         --seed 42 \
#         --epochs 1 \
#         --n_centroids 200 \
#         --crct_epochs 20 \
#         --ca_lr 0.005 \
#         --omega_lr_scale 0.1 \
#         --lr 0.003 \
#         --opt adam \
#         --ortho 0.1 \
#         --lora_type comp \
#         --lora_depth 10 \
#         --num_tasks 10 \
#         --lora_rank 12 \
#         --lora_fc1