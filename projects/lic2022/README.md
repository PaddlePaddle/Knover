# Lic2022 DuSinc

This project provides pretraining model and finetuning model for Lic2022 Dusinc competition, trained by Knover.

## About the dataset

* Pretraining dataset: a large-scale Chinese conversation dataset, which contains 60M utterances of 20M dialogues.
* Finetuning dataset: please download the dataset from the competition website.

## About the model

* [Pretraining model](https://dialogue.bj.bcebos.com/Knover/projects/lic2022/12L.pretrain.pdparams) : is trained on a large-scale Chinese conversation dataset.
* [Query Finetuning model](https://dialogue.bj.bcebos.com/Knover/projects/lic2022/query_finetune.pdparams) : is finetuned on dataset of Lic2022 Dusinc competition to generate search query.
* [Dialogue Finetuning model](https://dialogue.bj.bcebos.com/Knover/projects/lic2022/dial_finetune.pdparams) : is finetuned on dataset of Lic2022 Dusinc competition to generate dialogue response.

## How to finetune/inference with pretraining/finetuning model

 - step 1: Data process: run `tools/data_preprocessing` to convert dataset into training format used by [Knover](https://github.com/PaddlePaddle/Knover/tree/dygraph). Make sure that the input and output files have been modified to real files before running the script.
 - step 2: Train and infer: run training or inference script of [Knover](https://github.com/PaddlePaddle/Knover/tree/dygraph) using the config and model provided by this project. 
For example: 
	 - Train: sh ./scripts/local/job.sh  ./projects/lic2022/conf/query_train.conf  
	 - Infer: sh ./scripts/local/job.sh ./projects/lic2022/conf/query_infer.conf
 - step 3: Competition submission: concat the output files of the two tasks like the examples provided in the DuSinc dataset.

## Baseline of the finetuning model

|Q_ACC | Q_F1 | Q_BLEU1/2 | D_F1 | D_BLEU1/2 | DISTINCT1/2 | 
|---|---|---|---|---|---|
|0.553 | 0.218 | 0.159/0.155 | 0.189 | 0.137/0.084 | 0.158/0.557 | 

## Human Evaluation

* server.py : A deployed service sample is provided. Ensure that the service API can be accessed through the public network
* check_server.py : You can use this script to verify that the service is available
