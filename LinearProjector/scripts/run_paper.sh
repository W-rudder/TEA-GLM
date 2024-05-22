#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

neck=512
grad_steps=1
batch_size=2
dataset='arxiv'


llm='path/llm'
conv='sage'
seed=0
num_token=5
embed='bert'
llm_type='llama2'
prefix=''
pretrain_gnn=''

bash ./scripts/train_arxiv.sh $pretrain_gnn $neck $grad_steps $batch_size $prefix $seed $num_token $embed $llm $conv $llm_type

bash ./scripts/test_arxiv.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type

bash ./scripts/test_pubmed.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type

bash ./scripts/test_cora.sh $dataset $neck $prefix $num_token $embed $llm $conv $llm_type