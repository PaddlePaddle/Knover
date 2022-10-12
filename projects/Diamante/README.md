## Diamante

[**Diamante: Towards Boosting the Open-Domain Chatbot with Human Feedback**](https://arxiv.org/abs/2208.14165)

Diamante is a novel and efficient framework consisting of a data collection strategy and a learning method to boost the performance of pre-trained dialogue models. Two kinds of human feedback are collected and leveraged in Diamante, including explicit demonstration and implicit preference. The Diamante dataset is publicly available at the [LUGE platform](https://www.luge.ai/#/luge/dataDetail?id=52). 



## Abstract

Many open-domain dialogue models pre-trained with social media comments can generate coherent replies but have difficulties producing engaging responses. This phenomenon might mainly result from the deficiency of annotated human-human conversations and the misalignment with human preference. In this paper, we propose a novel and efficient framework Diamante to boost the open-domain chatbot, where two kinds of human feedback (including explicit demonstration and implicit preference) are collected and leveraged. By asking annotators to select or amend the model-generated candidate responses, Diamante efficiently collects the human demonstrated responses and constructs a Chinese chit-chat dataset. To enhance the alignment with human preference, Diamante leverages the implicit preference in the data collection process and introduces the generation-evaluation joint training. Comprehensive experiments indicate that the Diamante dataset and joint training paradigm can significantly boost the performance of pre-trained dialogue models. The overall engagingness of the previous state-of-the-art model has been improved remarkably by 50% in Chinese open-domain conversations.



## Requirements

* [Knover](../..)


## Dataset

The statistics of the Diamante dataset are listed as follows:

| Diamante                  |      Train      |      Valid      |      Test       |      Total      |
| :------------------------ | :-------------: | :-------------: | :-------------: | :-------------: |
| Number of Dialogues       |      5,838      |       500       |       500       |      6,838      |
| Number of Utterances      |     83,765      |      7,166      |      7,184      |     98,115      |
| Average Utterance Length  |      14.26      |      14.20      |      14.29      |      14.25      |
| Select / Revise / Rewrite | 18% / 41% / 41% | 19% / 40% / 41% | 19% / 40% / 41% | 18% / 41% / 41% |



## Usage

### Data Preparation

Download the [Diamante dataset](https://www.luge.ai/#/luge/dataDetail?id=52) and decompress it.

```bash
cd /path/to/Knover
python ./projects/Diamante/build_data.py
```

### Training

Apply Diamante's dataset and joint training paradigm to one pre-trained model (you can prepare the model and change the directory in the `train.conf` file). The default configuration will save the best and latest checkpoints to the `./output/` folder.
```bash
bash ./scripts/local/job.sh ./projects/Diamante/train.conf
```

### Inference

```bash
# Decompress the saved model
tar -xvf ./projects/Diamante/output/best.tar

# Interact with the model
bash ./scripts/local/job.sh ./projects/Diamante/infer.conf
```

After inference, you can find the output folder `./output` (by default). It contains the inference result `inference_output.txt`. You can change the `infer.conf` file for inference.

### Interacting with the model

```bash
# Decompress the saved model
tar -xvf ./projects/Diamante/output/best.tar

# Interact with the model
bash ./scripts/local/job.sh ./projects/Diamante/interact.conf
```


## Citation

If you find Diamante useful in your work, please cite the following paper:

```bibtex
@article{lu2022towards,
  title={Towards Boosting the Open-Domain Chatbot with Human Feedback},
  author={Lu, Hua and Bao, Siqi and He, Huang and Wang, Fan and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2208.14165},
  year={2022}
}
```



## Contact Information

For help or issues using Diamante, please submit a GitHub [issue](https://github.com/PaddlePaddle/Knover/issues).

For personal communication related to Diamante, please contact Hua Lu (`luhua05@baidu.com`), or Siqi Bao (`baosiqi@baidu.com`).
