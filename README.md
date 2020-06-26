# Modeling Global and Local Node Contexts for Text Generation from Knowledge Graphs
This repository contains the code for the TACL paper: "[Modeling Global and Local Node Contexts for Text Generation from Knowledge Graphs](https://arxiv.org/pdf/2001.11003.pdf)".

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

This project is implemented using the framework [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and the library [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric). Please, refer to their websites for further details on the installation and dependencies.

## Environments and Dependencies

- python 3.6
- PyTorch 1.1.0
- PyTorch Geometric 1.3.1
- subword-nmt 0.3.6

## Datasets

In our experiments, we use the following datasets:  [AGENDA](https://github.com/rikdz/GraphWriter/tree/master/data) and [WebNLG](https://webnlg-challenge.loria.fr/challenge_2017/).

## Preprocess

First, convert the dataset into the format required for the model.

For the AGENDA dataset, run:
```
./preprocess_AGENDA.sh <dataset_folder>
```
For the WebNLG dataset, run:
```
./preprocess_WEBNLG.sh <dataset_folder>
```


## Training
For traning the model using the AGENDA dataset, execute:
```
./train_AGENDA.sh <graph_encoder> <gpu_id>
```

For the WebNLG dataset, execute:
```
./train_WEBNLG.sh <graph_encoder> <gpu_id>
```

Options for `<graph_encoder>` are `pge`, `cge`, `pge-lw` or `cge-lw`. 

Examples:
```
./train_AGENDA.sh 0 pge
./train_WEBNLG.sh 0 cge-lw
```

## Decoding

For decoding, run:
```
./decode_AGENDA.sh <gpu_id> <model> <nodes_file> <graph_file> <output>
./decode_WEBNLG.sh <gpu_id> <model> <nodes_file> <graph_file> <output>
```

Example:
```
./decode_AGENDA.sh 0 model_agenda_cge_lw.pt test-nodes.txt test-graph.txt output-agenda-testset.txt
```

## Trained models

- CGE-LW trained on AGENDA training set ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/model_agenda_cge_lw.pt)): 
    - BLEU on AGENDA test set: 18.10, 58.8/29.5/16.4/9.0 (BP=0.804, ratio=0.821, hyp_len=114233, ref_len=139162) ([output](https://github.com/UKPLab/kg2text/tree/master/outputs/output_agenda.txt))

- CGE-LW trained on WEBNLG training set ([download](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/model_webnlg_cge_lw.pt)):
    - BLEU on WEBNLG seen test set: 63.68, 89.8/72.8/58.7/47.6 (BP=0.974, ratio=0.975, hyp_len=21984, ref_len=22554)
([output](https://github.com/UKPLab/kg2text/tree/master/outputs/output_webnlg.txt))

## More
For more details regading hyperparameters, please refer to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).


Contact person: Leonardo Ribeiro, ribeiro@aiphes.tu-darmstadt.de

## Citation
```
@misc{ribeiro2020modeling,
    title={Modeling Global and Local Node Contexts for Text Generation from Knowledge Graphs},
    author={Leonardo F. R. Ribeiro and Yue Zhang and Claire Gardent and Iryna Gurevych},
    year={2020},
    eprint={2001.11003},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


