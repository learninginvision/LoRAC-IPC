CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        cifar100_compilora_mask_tii \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./output/sup-21k*/lora_mask/all/05-21-18-47-42_lr_0.006_bs_64_epochs_10_rank_16_ortho_0.5_omega_lr_scale1.0_TH0.05_seed44" \
        --seed 44 \
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
        --threshold 0.05\
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --eval

CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        cifar100_compilora_mask_tii \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./output/sup-21k*/lora_mask/all/05-12-14-18-42_lr_0.006_bs_64_epochs_10_rank_16_ortho_0.75_omega_lr_scale1.0_TH0.05" \
        --seed 44 \
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
        --crct_epochs 50 \
        --ca_lr 0.005 \
        --eval

for seed in 44 40
do
CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        cifar100_compilora_mask_tii \
        --model vit_base_patch16_224_mocov3 \
        --batch-size 64 \
        --output_dir "./repeat_output/cifar100/moco" \
        --seed $seed \
        --epochs 20 \
        --lr 0.01 \
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
        --ortho 1.0 \
        --omega_lr_scale 1.0 \
        --threshold 0.0 \
        --first_session_lr 0.00 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 
done

for seed in 44 40
do
CUDA_VISIBLE_DEVICES=2,3 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        imr_compilora_mask_tii \
        --model vit_base_patch16_224_mocov3 \
        --batch-size 64 \
        --output_dir "./repeat_output/imagenet-r/moco" \
        --seed $seed \
        --epochs 20 \
        --lr 0.01 \
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
        --ortho 1.0 \
        --omega_lr_scale 1.0 \
        --first_session_lr 0.01 \
        --threshold 0.0 \
        --n_centroids 10 \
        --crct_epochs 50 \
        --ca_lr 0.005 
done

# CUDA_VISIBLE_DEVICES=2,3 
# python -m torch.distributed.launch \
#         --nproc_per_node=2 \
#         --master_port='29515' \
#         --use_env main.py \
#         cifar100_compilora_mask_tii \
#         --model vit_base_patch16_224_in21k \
#         --batch-size 64 \
#         --output_dir "./output/sup-21k*/lora_mask/all/05-21-19-25-31_lr_0.006_bs_64_epochs_10_rank_16_ortho_0.5_omega_lr_scale1.0_TH0.05_seed40" \
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
#         --ortho 0.5 \
#         --omega_lr_scale 1.0 \
#         --threshold 0.05\
#         --n_centroids 10 \
#         --crct_epochs 50 \
#         --ca_lr 0.005 \
#         --eval