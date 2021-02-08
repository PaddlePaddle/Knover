# Knover
Knover is a toolkit for knowledge grounded dialogue generation based on PaddlePaddle. Knover allows researchers and developers to carry out efficient training/inference of large-scale dialogue generation models. 

### What's New:

* July 2020: We are opening [PLATO-2](projects/PLATO-2/README.md), a large-scale generative model with latent space for open-domain dialogue systems.

## Requirements and Installation

* python version >= 3.7
* paddlepaddle-gpu version 1.8.0 ~ 1.8.5 (recommended version: 1.8.2.post107).
    * You can install PaddlePaddle following [the instructions](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html).
    * The specific version of PaddlePaddle is also based on your [CUDA version](https://developer.nvidia.com/cuda-downloads) (recommended version: 10.1) and [CuDNN version](https://developer.nvidia.com/rdp/cudnn-download) (recommended version: 7.6). See more information on [PaddlePaddle document about GPU support](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html#paddlepaddle-s-support-for-gpu)
* sentencepiece
* termcolor
* If you want to run distributed training, you'll also need [NCCL](https://developer.nvidia.com/nccl/nccl-download)
* Install Knover locally:

```bash
git clone https://github.com/PaddlePaddle/Knover.git
cd Knover
pip3 install -e .
```

* Or you can setup `PYTHONPATH` only:

```bash
export PYTHONPATH=/abs/path/to/Knover:$PYTHONPATH
```

## Basic usage

- See [usage document](docs/usage.md).

## Disclaimer
This project aims to facilitate further research progress in dialogue generation. Baidu is not responsible for the 3rd party's generation with the pre-trained system.

## Contact information
For help or issues using Knover, please submit a GitHub issue.
