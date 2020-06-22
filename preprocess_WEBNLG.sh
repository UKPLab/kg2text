#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "./preprocess_WEBNLG.sh <dataset_folder>"
  exit 2
fi

processed_data_folder='graph2text/data/webnlg2'
mkdir -p ${processed_data_folder}

python preprocess/generate_input_webnlg.py ${1} ${processed_data_folder}

python graph2text/preprocess.py -train_src ${processed_data_folder}/train-nodes.txt \
                       -train_graph ${processed_data_folder}/train-graph.txt \
                       -train_tgt ${processed_data_folder}/train-surfaces-bpe.txt \
                       -valid_src ${processed_data_folder}/dev-nodes.txt  \
                       -valid_graph ${processed_data_folder}/dev-graph.txt  \
                       -valid_tgt ${processed_data_folder}/dev-surfaces-bpe.txt \
                       -save_data ${processed_data_folder}/webnlg \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -dynamic_dict \
                       -share_vocab



