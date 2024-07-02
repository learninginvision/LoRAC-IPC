# for ortho in 0.1 0.5 1.0 5 10
# do
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29535' \
        --use_env main.py \
        cifar100_compilora_mask \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --output_dir "./output/sup-21k/lora_mask/qkvo" \
        --seed 42 \
        --epochs 1 \
        --lr 0.02 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_type compi \
        --lora_depth 12 \
        --num_tasks 10 \
        --lora_rank 36 \
        --ortho 1.0 \
        --omega_lr_scale 0.1 \
        --threshold 0.1
# done

# for ortho in 0.5 10
# do
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29535' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k/lora_new_mask/qkvo" \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.02 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 32 \
#         --ortho $ortho \
#         --omega_lr_scale 0.1 \
#         --threshold 0.05 \
#         --new_mask
# done

# for ortho in 15 25 50
# do
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29535' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k/new_lora_mask/all" \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.02 \
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
#         --omega_lr_scale 0.1 \
#         --threshold 0.1
# done

 
# for ortho in 0.01 0.05
# do
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29535' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k/lora_mask/qkvo" \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.02 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 36 \
#         --ortho $ortho \
#         --omega_lr_scale 0.1 \
#         --threshold 0.05
# done

# for threshold in 0.0 0.1 0.05
# do
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29535' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k/lora_mask/qkvo" \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.01 \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 8 \
#         --ortho 0.5 \
#         --omega_lr_scale 0.1 \
#         --threshold $threshold
# done

# for threshold in 0.0 0.1 0.05
# do
# CUDA_VISIBLE_DEVICES=0,1 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29535' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k/lora_mask/qkvo" \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.01 \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 8 \
#         --ortho 5.0 \
#         --omega_lr_scale 0.1 \
#         --threshold $threshold
# done

# for threshold in 0.1 0.05
# do
# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29535' \
#         --use_env main.py \
#         cifar100_compilora_mask \
#         --model vit_base_patch16_224 \
#         --batch-size 64 \
#         --output_dir ./output/sup-21k/lora_mask/qkvo \
#         --seed 42 \
#         --epochs 20 \
#         --lr 0.01 \
#         --sched cosine \
#         --opt adam \
#         --lora_qkv TTT \
#         --lora_out \
#         --lora_type compi \
#         --lora_depth 12 \
#         --num_tasks 10 \
#         --lora_rank 16 \
#         --ortho 0.1 \
#         --omega_lr_scale 1.0 \
#         --threshold $threshold
# done