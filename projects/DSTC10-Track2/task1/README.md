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
The trained model will be released before the end of February 2022.
