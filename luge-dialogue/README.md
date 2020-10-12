# LUGE-Dialogue



This project provides pretraining model and finetuning model for [luge-dialogue](https://www.datafountain.cn/competitions/470) competition, trained by Knover.

## About the dataset

* pretraining dataset: a large-scale Chinese conversation dataset, which contains 60M utterances of 20M dialogues without dialogues in the test of [luge-dialogue](https://www.datafountain.cn/competitions/470) competition.
* finetuning dataset: dialogues in all of the training set of [luge-dialogue](https://www.datafountain.cn/competitions/470) competition.

## About the model

* [Pretraining model](https://dialogue.bj.bcebos.com/luge/12L.pretrain.tar) : is trained on a large-scale Chinese conversation dataset.
* [Finetuning model](https://dialogue.bj.bcebos.com/luge/12L.finetune.tar) : is finetuned on dataset of [luge-dialogue](https://www.datafountain.cn/competitions/470) competition.

In these model, three special tokens are used to represent subtasks in the competition, which are chitchat dialogue、knowledge dialogue and recommend dialogue.

![模型输入](figures/inputs.png)

## How to finetune/inference with pretraining/finetuning model

- step 1: run tools/convert_data_to_numerical.py to convert dataset into numerical format used by [Knover](https://github.com/PaddlePaddle/Knover). make sure that the input and output files have been modified to real files before running the script.
- step 2: run train or inference script of [Knover](https://github.com/PaddlePaddle/Knover) using the config and model provided by this project.

## Baseline of the finetuning model

|   F1   |  BLEU1 / BLEU2  | DISTINCT1/DISTINCT2 |
| :----: | :-------------: | :-----------------: |
| 10.84% | 7.170% / 2.368% |   0.515% / 5.393%   |



