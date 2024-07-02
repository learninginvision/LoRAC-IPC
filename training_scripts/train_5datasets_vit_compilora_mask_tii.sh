CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        five_datasets_compilora_mask_tii \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir "./five_datasets_output/sup-21k/lora_mask_tii/qkvo" \
        --seed 42 \
        --epochs 5 \
        --lr 0.01 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 5 \
        --lora_rank 4 \
        --ortho 10.0 \
        --omega_lr_scale 0.1 \
        --threshold 0.05 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005  

# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29515' \
#         --use_env main.py \
#         five_datasets_compilora_mask_tii \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir "./five_datasets_output/sup-21k/lora_mask_tii_bw/qkvo" \
#         --seed 42 \
#         --epochs 5 \
#         --lr 0.01 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 5 \
#         --lora_rank 4 \
#         --ortho 1.0 \
#         --omega_lr_scale 0.1 \
#         --threshold 0.05 \
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 \
#         --batch_wise