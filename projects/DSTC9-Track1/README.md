# DSTC9-Track1: 
**Learning to Select External Knowledge with Multi-Scale Negative Sampling**

[Paper link](https://arxiv.org/abs/2102.02096)

## Abstract
The Track-1 of DSTC9 aims to effectively answer user requests or questions during task-oriented dialogues, which are out of the scope of APIs/DB. By leveraging external knowledge resources, relevant information can be retrieved and encoded into the response generation for these out-of-API-coverage queries. In this work, we have explored several advanced techniques to enhance the utilization of external knowledge and boost the quality of response generation, including schema guided knowledge decision, negatives enhanced knowledge selection, and knowledge grounded response generation. To evaluate the performance of our proposed method, comprehensive experiments have been carried out on the publicly available dataset. Our approach was ranked as the best in human evaluation of DSTC9 Track-1.

## Requirements:
* [Knover](../..)


## Download Dataset and Models

[Dataset](https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/data.tar) (this dataset is provided by [DSTC9-Track1](https://github.com/alexa/alexa-with-dstc9-track1-dataset)) contains:

* Dialogue logs and labels: `${DATASET_TYPE}_logs.json` and `${DATASET_TYPE}_labels.json`
* External knowledge base: `${DATASET_TYPE}_knowledge.json`
* Schema description: `schema_desc.json` (based on schema.json in [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz))

We also provide our models which are used in our submissions.

* Task1 fine-tuned model:
    * [SOP-32L-Context](https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SOP-32L-Context.tar): the pre-trained 32L evaluation model optimized with SOP and MLM loss and the knowledge-seeking turn detection is estimated based on the dialogue context.
    * [SOP-32L-Schema](https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SOP-32L-Schema.tar): the pre-trained 32L evaluation model optimized with SOP and MLM loss and the knowledge-seeking turn detection is estimated with the dialogue context, external knowledges and schema descriptions.
* Task2 fine-tuned model:
    * [SOP-32L-Selection](https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SOP-32L-Selection.tar): the pre-trained 32L evaluation model optimized with SOP and MLM loss and the knowledge selection is estimated with the dialogue context and the external knowledge.
* Task3 fine-tuned model:
    * [SU-32L](https://dialogue.bj.bcebos.com/Knover/projects/DSTC9-Track1/SU-32L.tar): the pre-trained  32L generation model optimized with NLL loss and the knowledge grounded generation task is based on the dialogue context and carried out with the retrieved knowledge.


## Run Inference
Here are the scripts used in our submission 0 and submissions 1. These scripts contain download commands and inference commands.

### Submission 0
Use SOP-32L-Context in the knowledge-seeking turn detection task, SOP-32L-Selection in the knowledge selection task and SU-32L in the knowledge grounded generation task.
```bash
bash ./submission_0_infer.sh
```

### Submission 1
Use SOP-32L-Schema in the knowledge-seeking turn detection task, SOP-32L-Selection in the knowledge selection task and SU-32L in the knowledge grounded generation task.
```
bash ./submission_1_infer.sh
```


## Citation
If you find the code or pre-trained models useful in your work, please cite the following Arxiv paper:
```bibtex
@article{he2021learning,
  title={Learning to Select External Knowledge with Multi-Scale Negative Sampling},
  author={He, Huang and Lu, Hua and Bao, Siqi and Wang, Fan and Wu, Hua and Niu, Zhengyu and Wang, Haifeng},
  journal={arXiv preprint arXiv:2102.02096},
  year={2021}
}
```

## Disclaimer
This project aims to facilitate further research progress in dialogue generation. Baidu is not responsible for the 3rd party's generation with the pre-trained system.
