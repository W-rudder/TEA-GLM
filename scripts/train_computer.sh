#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


llm='/path/to/llm'
seed=0
num_token=5
prefix='prefix of your first model to save'
pretrain_gnn='gnn_name.pth'


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    train_glm.py \
        --freeze_llama \
        --dataset computer \
        --pretrain_gnn $pretrain_gnn \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 1 \
        --batch_size 2 \
        --num_token $num_token \
        --clip_grad_norm 1.0 \
        --backbone $llm \
        --epoch 1 \
	    --weight_decay 0. \
        --max_text_length 1000 \
        --gen_max_length 64 \
	    --lr 0.002 \
        --prefix $prefix \
        --seed $seed
