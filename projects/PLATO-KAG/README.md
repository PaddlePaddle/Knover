# PLATO-KAG
[**PLATO-KAG: Unsupervised Knowledge-Grounded Conversation via Joint Modeling**](https://aclanthology.org/2021.nlp4convai-1.14/)

## Abstract
Large-scale conversation models are turning to leveraging external knowledge to improve the factual accuracy in response generation. Considering the infeasibility to annotate the external knowledge for large-scale dialogue corpora, it is desirable to learn the knowledge selection and response generation in an unsupervised manner. 

We propose **PLATO-KAG** (Knowledge-Augmented Generation), an unsupervised learning approach for end-to-end knowledge-grounded conversation modeling. For each dialogue context, the top-k relevant knowledge elements are selected and then employed in knowledge-grounded response generation. The two components of knowledge selection and response generation are optimized jointly and effectively under a balanced objective. 

Experimental results on two publicly available datasets validate the superiority of PLATO-KAG.

## Requirements
* [Knover](../..)

## Usage
We provide models and scripts for our experiments on two datasets: 

- [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) (wow) 
- [Holl-E](https://github.com/nikitacs16/Holl-E) (holle)

### Preparation
Prepare models and datasets.
```bash
cd $DATASET
bash ./prepare.sh
```

It downloads the pre-trained models and corresponding fine-tuned model to `./${DATASET}/models`:

- [24L_NSP](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/24L_NSP.tar): the pre-trained 24L dialogue evaluation model optimized with NSP and MLM loss, for initializing the knowledge selection module.
- [24L_SU](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/24L_SU.tar): the pre-trained 24L dialogue generation model optimized with NLL loss, for initializing the knowledge-grounded response generation module.
- [24L_PLATO_KAG_WOW](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/wow/24L_PLATO_KAG.tar): the PLATO-KAG model fine-tuned with [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) (wow).
- [24L_PLATO_KAG_HOLLE](https://dialogue.bj.bcebos.com/Knover/projects/PLATO-KAG/holle/24L_PLATO_KAG.tar): the PLATO-KAG model fine-tuned with [Holl-E](https://github.com/nikitacs16/Holl-E) (holle).

It also downloads the corresponding dataset to `./${DATASET}/data`.

### Training
Use pre-trained models to fine-tune PLATO-KAG on the corresponding dataset.
```bash
cd $DATASET
bash ./train.sh
```
After training, you can find the training log and checkpoints in `./${DATASET}/output`.

### Inference and Evaluation
Use fine-tuned model to infer and evaluate the test set.
```bash
cd $DATASET
bash ./infer.sh
```
After inference and evaluation, you can find the results of inference and evaluation score in `./${DATASET}/output`.


## Citation
Please cite the [paper](https://aclanthology.org/2021.nlp4convai-1.14/) if you use PLATO-KAG in your work:
```bibtex
@inproceedings{huang-etal-2021-plato,
    title = "{PLATO}-{KAG}: Unsupervised Knowledge-Grounded Conversation via Joint Modeling",
    author = "Huang, Xinxian and He, Huang and Bao, Siqi and Wang, Fan and Wu, Hua and Wang, Haifeng",
    booktitle = "Proceedings of the 3rd Workshop on Natural Language Processing for Conversational AI",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nlp4convai-1.14",
    doi = "10.18653/v1/2021.nlp4convai-1.14",
    pages = "143--154"
}
```

## Contact Information
For help or issues using PLATO-KAG, please submit a GitHub [issue](https://github.com/PaddlePaddle/Knover/issues).