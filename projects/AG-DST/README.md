# AG-DST
[**Amendable Generation for Dialogue State Tracking**](https://arxiv.org/abs/2110.15659)

![AG-DST Architecture](./images/AG-DST.png)

## Abstract
In task-oriented dialogue systems, recent dialogue state tracking methods tend to perform one-pass generation of the dialogue state based on the previous dialogue state. The mistakes of these models made at the current turn are prone to be carried over to the next turn, causing error propagation.

We propose a novel **A**mendable **G**eneration for **D**ialogue **S**tate **T**racking (**AG-DST**), which contains a two-pass generation process: (1) generating a primitive dialogue state based on the dialogue of the current turn and the previous dialogue state, and (2) amending the primitive dialogue state from the first pass. With the additional amending generation pass, our model is tasked to learn more robust dialogue state tracking by amending the errors that still exist in the primitive dialogue state, which plays the role of reviser in the double-checking process and alleviates unnecessary error propagation.

Experimental results show that AG-DST significantly outperforms previous works in two active DST datasets (MultiWOZ 2.2 and WOZ 2.0), achieving new state-of-the-art performances.

## Table of Contents
- [Requirements](#Requirements)
- [Usage](#Usage)
    - [Preparation](#Preparation)
    - [Training](#Training)
    - [Inference and Evaluation](#Inference-and-Evaluation)
- [Citation](#Citation)
- [Contact Information](#Contact-Information)

## Requirements:
* [Knover](../..)

## Usage

### Preparation
Prepare models and datasets.
```bash
bash ./prepare.sh
```
It downloads three models to `./models`:
- [AG-DST-24L](https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/AG-DST-24L.tar): the pre-trained 24L PLATO-2 in stage 1 (24-layer, 1024-hidden, 16-heads, 310M parameters).
- [AG-DST-24L-multiwoz](https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/AG-DST-24L-multiwoz.tar): the pre-trained 24L PLATO-2 in stage 1 is fine-tuned with the [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2) dataset.
- [AG-DST-24L-woz](https://dialogue.bj.bcebos.com/Knover/projects/AG-DST/AG-DST-24L-woz.tar): the pre-trained 24L PLATO-2 in stage 1 is fine-tuned with the [WOZ 2.0](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz) dataset.

It also processes MultiWOZ 2.2 and WOZ 2.0 under the `./data`, which includes downloading, preprocessing, negative sampling and serialization.

### Training
Use pre-trained model to fine-tune the DST dataset. You can set the dataset and GPUs in `./train.sh`.
```bash
bash ./train.sh
```
After training, you can find training log and checkpoints in `./output`.

### Inference and Evaluation
Use fine-tuned model to infer and evaluate the test set. You can set `init_params` in `./conf/infer.conf` to use your fine-tuned checkpoint.
```bash
bash ./infer.sh
```
After inference and evaluation, you can find results of inference and evaluation score in `./output`.

## Citation
Please cite the [arXiv paper](https://arxiv.org/abs/2110.15659) if you use AG-DST in your work:
```bibtex
@article{tian2021amendable,
  title={Amendable Generation for Dialogue State Tracking},
  author={Tian, Xin and Huang, Liankai and Lin, Yingzhan and Bao, Siqi and He, Huang and Yang, Yunyi and Wu, Hua and Wang, Fan and Sun, Shuqi},
  journal={arXiv preprint arXiv:2110.15659},
  year={2021}
}
```

## Contact Information
For help or issues using AG-DST, please submit a GitHub [issue](https://github.com/PaddlePaddle/Knover/issues).
