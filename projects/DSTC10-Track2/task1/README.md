# Task 1: Multi-domain Dialogue State Tracking

## Dataset
The augmented dataset by TOD-DA for multi-domain dialogue state tracking consists of 2 parts:
- [Ontology Based Augmentation](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task1/ontology-based-augmentation.json): This part is constructed by ontology based augmentation with MultiWOZ 2.3. The spoken conversation simulation is further applied to obtain noisy and spoken-like conversations. It contains 136,890 dialogues and 1,490,490 utterances.
- [Pattern Based Augmentation](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task1/pattern-based-augmentation.json): This part is constructed by pattern based augmentation with MultiWOZ 2.3 and DSTC10 validation set. The speech disfluency simulation and phoneme-level simulation are further applied to obtain noisy and spoken-like conversations. It contains 240,000 dialogues and 3,709,854 utterances.

The above two augmented parts share the same [schema](https://github.com/alexa/alexa-with-dstc10-track2-dataset/blob/main/task1/data/output_schema.json) and [database](https://github.com/alexa/alexa-with-dstc10-track2-dataset/blob/main/task1/data/db.json) as DSTC10 Track2. Each dialogue is represented as a JSON object with the following fields:
- `aug_strategy`: A list of data augmentation strategies.
- `log`: A list of user or system turns. Each turn consists of the following fields:
    - `text`: The utterance after augmentation.
    - `origin_text` (ontology based augmentation only): The original utterance from the MultiWOZ dataset.
    - `speaker`: The speaker of the current turn.
    - `dialog_state` (user turn only): The dialogue state of the current turn.

## Inference

### Requirements
* [Knover](../..)
* fuzzywuzzy
* python-Levenshtein

### Preparation
Prepare model and dataset.
```bash
bash ./prepare.sh
```
It downloads fine-tuned model to `./models`:
- [DST-32L](https://dialogue.bj.bcebos.com/Knover/projects/DSTC10-Track2/task1/DST-32L.tar): the pre-trained 32L PLATO-2 in stage 1 is fine-tuned with the TOD-DA dataset.

It also processes DSTC10 Track2 Task1 test set under the `./data`, which includes downloading, preprocessing and serialization.

### Run Inference
Use fine-tuned model to infer the test set.
```bash
bash ./infer.sh
```
After inference, the inference output is placed in `./output`.
You can use [official scripts](https://github.com/alexa/alexa-with-dstc10-track2-dataset/tree/main/task1/scripts) to evaluate the output.
