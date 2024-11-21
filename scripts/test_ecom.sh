#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
wandb offline

datasets=('children:1000' 'history:1300' 'computer:1000' 'photo:1000' 'sports:500')


# 其他的参数
dataset='computer'
num_token=5
prefix='test'
llm='/home/wangduo/zhr/model/vicuna-7b-v1.5'


for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"

    echo "Testing with max_text_length $max_text_length on dataset $test_dataset"
    bash ./scripts/test_single.sh $dataset $test_dataset $num_token $max_text_length $prefix $llm
done