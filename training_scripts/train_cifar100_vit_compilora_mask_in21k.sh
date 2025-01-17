# for seed in 44 40
# do
CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29545' \
        --use_env main.py \
        cifar100_compilora_mask \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./output/cifar100/no_mask/qkvo/" \
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
        --ca_lr 0.005 

# done     

# for threshold in 0.0 0.05 0.1
# do
# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29545' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k*/lora_mask/qkvo" \
#         --seed 42 \
#         --epochs 10 \
#         --lr 0.006 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 8 \
#         --ortho 0.5 \
#         --omega_lr_scale 1.0 \
#         --threshold $threshold 
# done

# for threshold in 
# do
# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29545' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k*/lora_mask/all" \
#         --seed 42 \
#         --epochs 10 \
#         --lr 0.006 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 8 \
#         --ortho 0.1 \
#         --omega_lr_scale 1.0 \
#         --threshold 0.05 
# done