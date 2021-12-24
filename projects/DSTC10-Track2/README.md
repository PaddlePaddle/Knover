# DSTC10-Track2
[**TOD-DA: Towards Boosting the Robustness of Task-oriented Dialogue Modeling on Spoken Conversations**](https://arxiv.org/abs/2112.12441)

## Abstract
Task-oriented dialogue systems have been plagued by the difficulties of obtaining large-scale and high-quality annotated conversations. Furthermore, most of the publicly available datasets only include written conversations, which are insufficient to reflect actual human behaviors in practical spoken dialogue systems. In this paper, we propose Task-oriented Dialogue Data Augmentation (TOD-DA), a novel model-agnostic data augmentation paradigm to boost the robustness of task-oriented dialogue modeling on spoken conversations. The TOD-DA consists of two modules: 1) Dialogue Enrichment to expand training data on task-oriented conversations for easing data sparsity and 2) Spoken Conversation Simulator to imitate oral style expressions and speech recognition errors in diverse granularities for bridging the gap between written and spoken conversations. With such designs, our approach ranked first in both tasks of DSTC10 Track2, a benchmark for task-oriented dialogue modeling on spoken conversations, demonstrating the superiority and effectiveness of our proposed TOD-DA.

## Tasks
[DSTC10 Track2](https://github.com/alexa/alexa-with-dstc10-track2-dataset) aims to benchmark the robustness of the task-oriented dialogue modeling against the gaps between written and spoken conversations. The TOD-DA dataset and our trained models are included for the two tasks:
- [Task 1: Multi-domain Dialogue State Tracking](task1/)
- [Task 2: Knowledge-grounded Dialogue Modeling](task2/)

## Citation
If you find the TOD-DA dataset or models useful in your work, please cite the following [arXiv paper](https://arxiv.org/abs/2112.12441):
```bibtex
@article{tian2021tod-da,
  title={TOD-DA: Towards Boosting the Robustness of Task-oriented Dialogue Modeling on Spoken Conversations},
  author={Tian, Xin and Huang, Xinxian and He, Dongfeng and Lin, Yingzhan and Bao, Siqi and He, Huang and Huang, Liankai and Ju, Qiang and Zhang, Xiyuan and Xie, Jian and Sun, Shuqi and Wang, Fan and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2112.12441},
  year={2021}
}
```

## Contact Information
For help or questions, please submit a GitHub [issue](https://github.com/PaddlePaddle/Knover/issues).
