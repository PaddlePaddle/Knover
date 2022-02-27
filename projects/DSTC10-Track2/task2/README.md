# Task 2: Knowledge-grounded Dialogue Modeling

## Dataset
The augmented dataset by TOD-DA is constructed through ontology based augmentation with written conversations from [DSTC9 Track1](https://github.com/alexa/alexa-with-dstc9-track1-dataset). 
The spoken conversation simulation is further applied to obtain noisy and spoken-like conversations. The augmented dataset consists of 3 parts:

- [Knowledge-seeking Turn Detection](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/knowledge-seeking-turn-detection.json): The training data to help determine whether to trigger external knowledge access or not.
It contains 363,627 samples.

- [Knowledge Selection](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/knowledge-selection.json): The training data to help select the appropriate external knowledge snippet. 
It contains 392,527 samples. 
Each sample has one relevant positive and multiple enhanced negative knowledge snippets.
- [Knowledge-grounded Response Generation](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/knowledge-grounded-response-generation.json): The training data to help generate system response grounded on the provided knowledge snippet. It contains 20,084 samples.

### Data Formats
#### Knowledge-seeking Turn Detection
The *knowledge-seeking-turn-detection.json* file provides the dialogue contexts and the corresponding detection labels. It includes a list of JSON objects with the following fields:

- `context`: A list of the dialogue turns, from the conversation beginning to the target user turn.
- `target`: The boolean value to indicate whether the target user turn should trigger external knowledge access or not.

#### Knowledge Selection
The *knowledge-selection.json* file provides the dialogue contexts and the corresponding knowledge snippet candidates. It includes a list of JSON objects with the following fields:

- `context`: A list of the dialogue turns, from the conversation beginning to the target user turn.
- `candidates`: A list of knowledge candidates with identifiers from [knowledge.json](https://github.com/alexa/alexa-with-dstc10-track2-dataset/blob/main/task2/data/knowledge.json).
Each of the candidates is a JSON object with the following fields:
    - `domain`: The candidate's domain identifier.
    - `entity_id`: The candidate's entity identifier.
    - `doc_id`: The candidate's document identifier.
    - `label`: The boolean value to indicate whether this knowledge snippet is positive or not.

#### Knowledge-grounded Response Generation
The *knowledge-grounded-response-generation.json* file provides the system responses grounded on the provided knowledge snippets and the corresponding dialogue contexts. It includes a list of JSON objects with the following fields:

- `context`: A list of the dialogue turns, from the conversation beginning to the target user turn.
- `knowledge`: It is a JSON object with the following fields:
    - `domain`: The knowledge snippet's domain identifier.
    - `entity_id`: The knowledge snippet's entity identifier.
    - `doc_id`: The knowledge snippet's document identifier.
- `response`: The target system response.

## Inference
### Requirements:
* [Knover](../..)

### Download Dataset and Models

The [dataset](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/data.tar) (based on dataset provided by [DSTC10-Track2](https://github.com/alexa/alexa-with-dstc10-track2-dataset)) for inference contains:

* Dialogue logs and labels: `${DATASET_TYPE}_logs.json` and `${DATASET_TYPE}_labels.json`
* External knowledge base: `knowledge.json`
* Entities and locations: `entities.json` and `locations.json` (extracted from `knowledge.json` to enable fuzzy matching)

We also provide our models which are used in our submissions.

* Subtask1 fine-tuned model:
    * [SOP-32L-Detection](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/SOP-32L-Detection.tar): the pre-trained 32L evaluation model optimized with SOP and MLM loss and the knowledge-seeking turn detection is estimated based on the dialogue context.
* Subtask2 fine-tuned model:
    * [SOP-32L-Selection](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/SOP-32L-Selection.tar): the pre-trained 32L evaluation model optimized with SOP and MLM loss and the knowledge selection is estimated with the dialogue context and the external knowledge.
* Subtask3 fine-tuned model:
    * [SU-32L](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task2/SU-32L.tar): the pre-trained 32L generation model optimized with NLL loss and the knowledge grounded generation task is based on the dialogue context and carried out with the retrieved knowledge.


### Run Inference
Here is the script used in our submission 0. The script contains download commands and inference commands.
```bash
bash ./submission_0_infer.sh
```