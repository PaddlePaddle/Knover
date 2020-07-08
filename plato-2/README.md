This repository will host code and models for PLATO-2 (https://arxiv.org/abs/2006.16779).

All the release will be ready on or before July 12, 2020.

The English models of PLATO-2 and the inference script for static evaluation have been released. We will release the interactive script and training code for generative models soon.

# PLATO-2: 
**PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning**

[Paper link](https://arxiv.org/abs/2006.16779)

## Abstract
To build a high-quality open-domain chatbot, we introduce the effective training process of PLATO-2 via curriculum learning. There are two stages involved in the learning process. In the first stage, a coarse-grained generation model is trained to learn response generation under the simplified framework of one-to-one mapping. In the second stage, a fine-grained generation model and an evaluation model are further trained to learn diverse response generation and response coherence estimation, respectively. PLATO-2 was trained on both Chinese and English data, whose effectiveness and superiority are verified through comprehensive evaluations, achieving new state-of-the-art results.

## Requirements
```
- python >= 3.7.3
- paddlepaddle-gpu >= 1.8.1
- numpy
- sentencepiece
```

## Pre-trained dialogue generation model
A novel pre-training model for dialogue generation is introduced in this work, incorporated with latent discrete variables for one-to-many relationship modeling.

* PLATO-2, 24L (310M params), EN: [Model](https://baidu-nlp.bj.bcebos.com/PLATO-2/24L.tar)
* PLATO-2, 32L (1.6B params), EN: [Model](https://baidu-nlp.bj.bcebos.com/PLATO-2/32L.tar)

```bash
MODEL_SIZE=24L # 24L / 32L
mv /path/to/${MODEL_SIZE}.tar /path/to/Knover
tar xf ${MODEL_SIZE}.tar
```

## Inference

### Data format
You can check the data format of inference in "./data/dailydialog_test_60.tsv"
```bash
src \t tgt
u_1 [SEP] u_2 [SEP] ... u_n \t r
```

### Inference
Commands for running inference. The 32L PLATO-2 model requires a 32GB V100 while `mem_efficient = false`, and you can run it on a 16GB V100 with `mem_efficient = true`.

**PLATO-2, 24L**
```bash
sh ./script/local/save_nsp_model.sh 24L
sh ./script/local/24L_plato_inference.sh
```

**PLATO-2, 32L**
```bash
sh ./script/local/save_nsp_model.sh 32L
sh ./script/local/32L_plato_inference.sh
```

After inference, you can find the output folder `./output` (by default). It contains the inference result `inference_output.txt`.

## Citation
If you find PLATO-2 useful in your work, please cite the following Arxiv paper:
```
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
