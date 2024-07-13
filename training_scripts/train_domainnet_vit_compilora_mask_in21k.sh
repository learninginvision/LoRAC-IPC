for seed in 42 44 40
do
CUDA_VISIBLE_DEVICES=0,1 
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29515' \
        --use_env main.py \
        domainnet_compilora_mask \
        --model vit_base_patch16_224_in21k \
        --batch-size 64 \
        --output_dir "./output/domainnet_cil/qkvo" \
        --seed $seed \
        --epochs 50 \
        --lr 0.006 \
        --sched cosine \
        --opt adam \
        --lora_qkv TTT \
        --lora_out \
        --lora_type compi \
        --lora_depth 12 \
        --lora_rank 64 \
        --ortho 0.5 \
        --omega_lr_scale 1.0 \
        --threshold 0.05
done