for seed in 44 40
do
CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        imr_compilora_mask_tii \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./imr_output/sup-21k*/lora_mask_tii/all" \
        --seed $seed \
        --epochs 50 \
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
        --lora_rank 32 \
        --ortho 1.0 \
        --omega_lr_scale 1.0 \
        --threshold 0.05 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 
done

# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29545' \
#         --use_env main.py \
#         imr_compilora_mask_tii \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --output_dir "./imr_output/sup-21k*/lora_mask_tii/all" \
#         --seed 42 \
#         --epochs 50 \
#         --lr 0.006 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_fc1 \
#         --lora_fc2 \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 16 \
#         --ortho $ortho \
#         --omega_lr_scale 1.0 \
#         --threshold 0.05 \
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 


# for ortho in 0.1 0.5 1
# do
# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29545' \
#         --use_env main.py \
#         imr_compilora_mask_tii \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --output_dir "./imr_output/sup-21k*/lora_mask_tii/all" \
#         --seed 42 \
#         --epochs 50 \
#         --lr 0.006 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_fc1 \
#         --lora_fc2 \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 32 \
#         --ortho $ortho \
#         --omega_lr_scale 1.0 \
#         --threshold 0.05 \
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 
# done