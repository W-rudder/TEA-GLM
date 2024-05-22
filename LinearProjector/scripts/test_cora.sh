#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline


accelerate launch \
    --config_file accelerate_config/my_config_0.yaml \
    train_glm.py \
        --freeze_llama \
        --inference \
        --zero_shot \
        --best_epoch 0 \
        --dataset $1 \
        --test_dataset cora \
        --neck $2 \
        --att_d_model 2048 \
        --gnn_output 4096 \
	    --grad_steps 1 \
        --batch_size 4 \
        --num_token $4 \
        --clip_grad_norm 1.0 \
        --backbone $6 \
        --epoch 1 \
	    --weight_decay 0.1 \
        --max_text_length 850 \
        --gen_max_length 64 \
	    --lr 0.001 \
        --prefix $3 \
        --embed_type $5 \
        --conv_type $7 \
        --llm_type $8