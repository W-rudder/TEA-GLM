#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    train_glm.py \
        --freeze_llama \
        --dataset arxiv \
        --pretrain_gnn $1 \
        --att_d_model 2048 \
        --gnn_output 4096 \
        --neck $2 \
	    --grad_steps $3 \
        --batch_size $4 \
        --num_token $7 \
        --clip_grad_norm 1.0 \
        --backbone $9 \
        --epoch 1 \
	    --weight_decay 0. \
        --max_text_length 700 \
        --gen_max_length 64 \
	    --lr 0.002 \
        --prefix $5 \
        --seed $6 \
        --embed_type $8 \
        --conv_type ${10} \
        --llm_type ${11}