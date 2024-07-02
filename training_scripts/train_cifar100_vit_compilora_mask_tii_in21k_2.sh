CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29525' \
        --use_env main.py \
        cifar100_compilora_mask_tii \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./output/sup-21k*/lora_mask/all/05-12-14-18-42_lr_0.006_bs_64_epochs_10_rank_16_ortho_0.75_omega_lr_scale1.0_TH0.05" \
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
        --ortho 0.75 \
        --omega_lr_scale 1.0 \
        --threshold 0.05\
        --n_centroids 10 \
        --crct_epochs 20 \
        --ca_lr 0.005 \
        --eval

# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29525' \
#         --use_env main.py \
#         cifar100_compilora_mask_tii \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k*/lora_mask/all/05-12-13-03-50_lr_0.006_bs_64_epochs_10_rank_16_ortho_0.25_omega_lr_scale1.0_TH0.05" \
#         --seed 40 \
#         --epochs 10 \
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
#         --ortho 0.25 \
#         --omega_lr_scale 1.0 \
#         --threshold 0.05\
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 \
#         --eval

# for seed in 44 40
# do
# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29525' \
#         --use_env main.py \
#         cifar100_compilora_mask_tii \
#         --model vit_base_patch16_224_mocov3 \
#         --batch-size 64 \
#         --output_dir "./repeat_output/cifar100/moco" \
#         --seed $seed \
#         --epochs 20 \
#         --lr 0.003 \
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
#         --ortho 0.1 \
#         --omega_lr_scale 1.0 \
#         --threshold 0.05 \
#         --first_session_lr 0.01 \
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 
# done

# for seed in 44 40
# do
# CUDA_VISIBLE_DEVICES=0,1 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29525' \
#         --use_env main.py \
#         imr_compilora_mask_tii \
#         --model vit_base_patch16_224_mocov3 \
#         --batch-size 64 \
#         --output_dir "./repeat_output/imagenet-r/moco" \
#         --seed $seed \
#         --epochs 20 \
#         --lr 0.005 \
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
#         --ortho 0.1 \
#         --omega_lr_scale 1.0 \
#         --first_session_lr 0.01 \
#         --threshold 0.05 \
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 
# done