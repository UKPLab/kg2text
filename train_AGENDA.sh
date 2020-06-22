#!/bin/bash

data_prefix='graph2text/data/agenda/agenda'
model_dir='graph2text/data/agenda_model'

GPUID=$1
graph_encoder=$2

export CUDA_VISIBLE_DEVICES=${GPUID}
export OMP_NUM_THREADS=10
python -u graph2text/train.py \
                        -data $data_prefix \
                        -save_model $model_dir$RANDOM \
                        -world_size 1 \
                        -gpu_ranks 0 \
                        -save_checkpoint_steps 5000 \
                        -valid_steps 5000 \
                        -report_every 50 \
                        -train_steps 400000 \
                        -warmup_steps 16000 \
                        --share_decoder_embeddings \
                        -share_embeddings \
                        --position_encoding \
                        --optim adam \
                        -adam_beta1 0.9 \
                        -adam_beta2 0.98 \
                        -decay_method noam \
                        -learning_rate 0.5 \
                        -max_grad_norm 0.0 \
                        -batch_size 4096 \
                        -batch_type tokens \
                        -normalization tokens \
                        -dropout 0.3 \
                        -attention_dropout 0.3 \
                        -label_smoothing 0.1 \
                        -max_generator_batches 100 \
                        -param_init 0.0 \
                        -param_init_glorot \
                        -encoder_type $graph_encoder \
                        -decoder_type transformer \
                        -dec_layers 6 \
                        -enc_layers 6 \
                        -word_vec_size 512 \
                        -enc_rnn_size 448 \
                        -dec_rnn_size 512 \
                        -number_edge_types 13 \
                        -heads 8
