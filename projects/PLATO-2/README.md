# PLATO-2: 
**PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning**

[Paper link](https://arxiv.org/abs/2006.16779)

## Abstract
To build a high-quality open-domain chatbot, we introduce the effective training process of PLATO-2 via curriculum learning. There are two stages involved in the learning process. In the first stage, a coarse-grained generation model is trained to learn response generation under the simplified framework of one-to-one mapping. In the second stage, a fine-grained generation model and an evaluation model are further trained to learn diverse response generation and response coherence estimation, respectively. PLATO-2 was trained on both Chinese and English data, whose effectiveness and superiority are verified through comprehensive evaluations, achieving new state-of-the-art results.

## Requirements:
* [Knover](../..)

## Pre-trained dialogue generation model
A novel pre-training model for dialogue generation is introduced in this work, incorporated with latent discrete variables for one-to-many relationship modeling.

* PLATO-2, 24L (310M params), EN: [Model](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-2/24L.tar)
* PLATO-2, 32L (1.6B params), EN: [Model](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-2/32L.tar)

```bash
MODEL_SIZE=24L # 24L / 32L
cd /path/to/Knover
wget https://baidu-nlp.bj.bcebos.com/PLATO-2/${MODEL_SIZE}.tar
tar xf ${MODEL_SIZE}.tar
```

## Inference

### Data format
You can check the data format of inference in `./data/dailydialog_test_60.tsv`
```
src \t tgt
u_1 [SEP] u_2 [SEP] ... u_n \t r
```

### Inference
Commands for running inference. The 32L PLATO-2 model requires a 32GB V100 while `mem_efficient = false`, and you can run it on a 16GB V100 with `mem_efficient = true`. You can change this config in `infer_args` like [32L inference configuration](pretrain/32L_infer.conf#L27)

##### **PLATO-2, 24L**

```bash
cd /path/to/Knover
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/24L_infer.conf
```

##### **PLATO-2, 32L**
```bash
cd /path/to/Knover
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/32L_infer.conf
```

After inference, you can find the output folder `./output` (by default). It contains the inference result `inference_output.txt`.

## Interaction
Commands for interaction with PLATO-2 models.

##### **PLATO-2, 24L**

```bash
cd /path/to/Knover
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/24L_interact.conf
```

##### **PLATO-2, 32L**

```bash
cd /path/to/Knover
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/32L_interact.conf
```

## Training
Commands for multi-GPU training model. You can find the saved log in `/path/to/Knover/log` and the saved model in `/path/to/Knover/output`.

**Note**: You need to install NCCL and set up the environment variable `LD_LIBRARY` properly.

##### **PLATO-2, 24L, pretrain**

```bash
cd /path/to/Knover
# Run stage-1 training.
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/24L_train_stage-1.conf
# Run stage-2.1 training.
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/24L_train_stage-2.1.conf
# Run stage-2.2 training.
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/24L_train_stage-2.2.conf
```

##### **PLATO-2, 32L, pretrain**

This training requires at least one 32G V100.

```bash
cd /path/to/Knover
# Run stage-1 training.
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/32L_train_stage-1.conf
# Run stage-2.1 training.
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/32L_train_stage-2.1.conf
# Run stage-2.2 training.
bash ./scripts/local/job.sh ./projects/PLATO-2/pretrain/32L_train_stage-2.2.conf
```

##### **PLATO-2, 24L, finetune**

```bash
cd /path/to/Knover
bash ./scripts/local/job.sh ./projects/PLATO-2/finetune/24L_train.conf
```

##### **PLATO-2, 32L, finetune**

This training requires at least one 32G V100.
```bash
cd /path/to/Knover
bash ./scripts/local/job.sh ./projects/PLATO-2/finetune/32L_train.conf
```

## Citation
If you find PLATO-2 useful in your work, please cite the following Arxiv paper:
```bibtex
@article{bao2020plato,
  title={PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning},
  author={Bao, Siqi and He, Huang and Wang, Fan and Wu, Hua and Wang, Haifeng and Wu, Wenquan and Guo, Zhen and Liu, Zhibin and Xu, Xinchao},
  journal={arXiv preprint arXiv:2006.16779},
  year={2020}
}
```

## Disclaimer
This project aims to facilitate further research progress in dialogue generation. Baidu is not responsible for the 3rd party's generation with the pre-trained system.

## Contact information
For help or issues using PLATO-2, please submit a GitHub issue.

For personal communication related to PLATO-2, please contact Siqi Bao (`baosiqi@baidu.com`), or Huang He (`hehuang@baidu.com`).
