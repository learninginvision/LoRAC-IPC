CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29544' \
        --use_env main.py \
        cifar100_complora_tii \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --data-path ./datasets \
        --output_dir ./output/cifar_vit_epoch_complora_tii_seed42 \
        --seed 42 \
        --epochs 10 \
        --n_centroids 10 \
        --crct_epochs 30 \
        --ca_lr 0.005 \
        --lr 0.003 \
        --opt adam \
        --ortho 0.1 \
        --lora_type comp \
        --lora_out \
        --lora_fc1 \
        --lora_fc2 \
        --lora_depth 8 \
        --num_tasks 10 \
        --lora_rank 8