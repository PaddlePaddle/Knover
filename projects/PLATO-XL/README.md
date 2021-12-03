# PLATO-XL: 
**PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation**

[Paper link](https://arxiv.org/abs/2109.09519)

## Abstract
To explore the limit of dialogue generation pre-training, we present the models of PLATO-XL with up to 11 billion parameters, trained on both Chinese and English social media conversations. To train such large models, we adopt the architecture of unified transformer with high computation and parameter efficiency. In addition, we carry out multi-party aware pre-training to better distinguish the characteristic information in social media conversations. With such designs, PLATO-XL successfully achieves superior performances as compared to other approaches in both Chinese and English chitchat. We further explore the capacity of PLATO-XL on other conversational tasks, such as knowledge grounded dialogue and task-oriented conversation. The experimental results indicate that PLATO-XL obtains state-of-the-art results across multiple conversational tasks, verifying its potential as a foundation model of conversational AI.

## Requirements:
* [Knover](../..)
* PaddlePaddle >= 2.2.0

## Pre-trained dialogue generation model
* PLATO-XL (11B params), EN: [Model](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-XL/11B.tar)

## Inference

### Data format
You can check the data format of inference in `./data/dailydialog_test_60.tsv`
```
src \t tgt
u_1 [SEP] u_2 [SEP] ... u_n \t r
```

### Inference
Commands for running inference. The 11B PLATO-XL model requires two 32GB V100.

##### **PLATO-XL**
```bash
bash ./infer.sh
```

After inference, you can find the output folder `./output` (by default). It contains the inference result `inference_output.txt`. You can change the [config file](infer.conf) for inference.

## Interaction
Commands for interaction with PLATO-XL models. The 11B PLATO-XL model requires two 32GB V100.

##### **PLATO-XL**
```bash
bash ./interact.sh
```

## Citation
If you find PLATO-XL useful in your work, please cite the following Arxiv paper:
```bibtex
@article{bao2021plato,
  title={PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation},
  author={Bao, Siqi and He, Huang and Wang, Fan and Wu, Hua and Wang, Haifeng and Wu, Wenquan and Wu, Zhihua and Guo, Zhen and Lu, Hua and Huang, Xinxian and Tian, Xin and and Xu, Xinchao and Lin, Yingzhan and Niu, Zhengyu},
  journal={arXiv preprint arXiv:2109.09519},
  year={2021}
}
```

## Disclaimer
This project aims to facilitate further research progress in dialogue generation. Baidu is not responsible for the 3rd party's generation with the pre-trained system.

## Contact information
For help or issues using PLATO-XL, please submit a GitHub issue.

For personal communication related to PLATO-XL, please contact Siqi Bao (`baosiqi@baidu.com`), or Huang He (`hehuang@baidu.com`).
