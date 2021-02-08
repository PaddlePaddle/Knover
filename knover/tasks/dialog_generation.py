#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dialogue generation task."""

from collections import defaultdict
import math

from knover.core.task import Task
from knover.data.dialog_reader import DialogReader
from knover.data.plato_reader import PlatoReader
from knover.tasks import register_task
from knover.utils.args import str2bool
from knover.utils.inference_utils import create_predictor


@register_task("DialogGeneration")
class DialogGeneration(Task):
    """Define dialogue response generation task."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Task")
        group.add_argument("--do_generation", type=str2bool, default=True,
                           help="Whether to run generation on inference phase. "
                           "Dialogue generation support two type of inference: generation and scoring.")
        group.add_argument("--is_cn", type=str2bool, default=False,
                           help="Whether to run in Chinese data.")

        group.add_argument("--nsp_inference_model_path", type=str, default=None,
                           help="The path of NSP inference model which is used to provide the NSP ranking scores.")

        group.add_argument("--ranking_score", type=str, default="decode_score",
                           help="Which score will be used to rerank.")

        args, _ = parser.parse_known_args()
        if args.model == "Plato":
            PlatoReader.add_cmdline_args(parser)
        else:
            DialogReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(DialogGeneration, self).__init__(args)
        self.do_generation = args.do_generation
        self.is_cn = args.is_cn

        # reserve example for inference
        args.reserve_example = True
        if args.model == "Plato":
            self.reader = PlatoReader(args)
        else:
            self.reader = DialogReader(args)

        if args.nsp_inference_model_path:
            self.nsp_predictor = create_predictor(args.nsp_inference_model_path, args.get("is_distributed", False))
        else:
            self.nsp_predictor = None

        self.ranking_score = args.ranking_score
        self.max_dec_len = args.max_dec_len

        return

    def _post_process_generation_output(self, predictions):
        """Post-process generation output.

        Calculate repetion, reranking.

        Args:
            predictions: the generation outputs of the model.

        Returns:
            Return the top-1 prediction.
        """
        if self.nsp_predictor is not None:
            get_nsp_score_batch(self.nsp_predictor, predictions)

        group = defaultdict(list)
        for pred in predictions:
            group[pred["data_id"]].append(pred)

        predictions = []
        for data_id in group:
            example = self.reader.features[data_id]
            preds = group[data_id]
            for pred in preds:
                # TODO: fix tokenized input
                words = [self.reader.tokenizer.preprocess(s).split(" ") for s in example.src.split("[SEP]")]
                pred_token_ids, pred_words = post_process_response(pred["response_token_ids"], self.reader)
                num_token = len(pred_token_ids)

                cross_turn_repetition = check_cross_turn_repetition(
                    words, pred_words, self.reader.eos_id, self.is_cn)
                in_turn_repetition = check_in_turn_repetition(pred_words, self.is_cn) \
                    or check_in_turn_repetition(pred_token_ids)

                pred["response"] = " ".join(pred_words)
                pred["score"] = pred[self.ranking_score]
                if self.max_dec_len is not None and num_token >= self.max_dec_len: # not ending
                    pred["score"] -= 1e3
                elif cross_turn_repetition:
                    pred["score"] -= 1e3
                elif in_turn_repetition:
                    pred["score"] -= 1e3

            preds = sorted(preds, key=lambda pred: -pred["score"])
            if self.debug_mode:
                print("Example:", example.data_id)
                print("Context:")
                for s in example.src.split(" [SEP] "):
                    print("\t" + s)
                if "knowledge" in example._fields:
                    print("Knowledge:")
                    print("\t" + example.knowledge)
                print("Predictions:")
                for pred in preds:
                    print(f"\t{pred['response']}\t{pred['score']:.5f}")
            pred = preds[0]
            keep_attr = ["data_id", "score", "response"]
            pred = {k: pred[k] for k in keep_attr}
            predictions.append(pred)
        return predictions

    def _post_process_scoring_output(self, predictions):
        """Post-process scoring output.

        The score is calculated by perplexity (PPL).
        """
        raise NotImplementedError

    def _post_process_infer_output(self, predictions):
        """Post-process inference output.

        Dialogue generation task supports two type of inference.
        1. Generating a response for the given context.
        2. Calculate the response generation score for the give context.
        """
        if self.do_generation:
            return self._post_process_generation_output(predictions)
        else:
            return self._post_process_scoring_output(predictions)

    def merge_metrics_and_statistics(self, outputs, part_outputs):
        """Merge two evaulation output.

        Args:
            outputs: Original outputs which contains metrics and statistics.
            part_outputs: New outputs which contains metrics and statistics.

        Returns:
            Return merged output which contains metrics and statistics.
        """
        if outputs is None:
            return part_outputs

        if part_outputs is None:
            return outputs

        batch_size = outputs.pop("batch_size")
        tokens_num = outputs.pop("tokens_num")
        part_batch_size = part_outputs.pop("batch_size")
        part_tokens_num = part_outputs.pop("tokens_num")

        new_outputs = {
            "batch_size": batch_size + part_batch_size,
            "tokens_num": tokens_num + part_tokens_num
        }
        for k in outputs:
            if k.startswith("token_"):
                new_outputs[k] = (
                    outputs[k] * tokens_num + part_outputs[k] * part_tokens_num
                ) / new_outputs["tokens_num"]
            else:
                new_outputs[k] = (
                    outputs[k] * batch_size + part_outputs[k] * part_batch_size
                ) / new_outputs["batch_size"]
        return new_outputs

    def get_metrics(self, outputs):
        """Get metrics."""
        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        metrics = {}
        batch_size = outputs.pop("batch_size", None)
        tokens_num = outputs.pop("tokens_num", None)
        for k in outputs:
            if k.startswith("token_"):
                metrics[k[6:]] = outputs[k]
            else:
                metrics[k] = outputs[k]
            if k == "token_lm_loss":
                metrics["ppl"] = math.exp(outputs[k])
        return metrics


def post_process_context(token_ids, reader, merge=True):
    """Post-process the context id sequence.

    Truncate the <bos> token.
    Convert token ids to words (merge = True) or tokens (merge = False).

    Args:
        token_ids: Token id sequence.

    Returns:
        context: A list of utterances. Each utterance is either a word list or a token list.
    """
    context = []
    utt = []
    for tok_id in token_ids[1:]:
        if tok_id == reader.eos_id:
            utt = reader.tokenizer.convert_ids_to_tokens(utt)
            if merge:
                utt = reader.tokenizer.merge_subword(utt)
            context.append(utt)
            utt = []
        else:
            utt.append(tok_id)
    return context


def post_process_response(token_ids, reader, merge=True):
    """Post-process the decoded sequence.

    Truncate from the first <eos> and remove the <bos> and <eos> tokens.
    Convert token_ids to words(merge=True) or tokens(merge=False)

    Args:
        token_ids: Token id sequence.
        merge: If true, merge subwords (tokens) into words, otherwise return tokens.

    Returns:
        token_ids: Truncated token_ids.
        response: A word list or a token list.
    """
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == reader.eos_id:
            eos_pos = i
            break
    token_ids = token_ids[1:eos_pos]
    response = reader.tokenizer.convert_ids_to_tokens(token_ids)
    if merge:
        response = reader.tokenizer.merge_subword(response)
    return token_ids, response


def check_cross_turn_repetition(context, pred, eos_idx, is_cn=False):
    """Check the cross-turn repetition.

    Calcuate tri-gram repetition.

    Args:
        context: Words or tokens or token_ids.
        pred: Words or tokens or token_ids.
        is_cn: Chinese version repetition detection. If true, calcuate repetition on characters.

    Returns:
        Whether the cross-turn repetition is detected.
    """
    if isinstance(pred[0], str):
        context = [[tok.lower() for tok in utt] for utt in context]
        pred = [tok.lower() for tok in pred]
        if is_cn:
            context = ["".join(utt) for utt in context]
            pred = "".join(pred)

    pred_tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        pred_tri_grams.add(tri_gram)
    for utt in context:
        for i in range(len(utt) - 2):
            tri_gram = tuple(utt[i:i + 3])
            if tri_gram in pred_tri_grams:
                return True
    return False


def check_in_turn_repetition(pred, is_cn=False):
    """Check the in-turn repetition.

    Calcuate tri-gram repetition.

    Args:
        pred: Words or tokens or token_ids.
        is_cn: Chinese version repetition detection. If true, calcuate repetition on characters.

    Returns:
        Whether the in-turn repetion is detected.
    """
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)

    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return True
        tri_grams.add(tri_gram)
    return False


def get_nsp_score_batch(nsp_predictor, predictions):
    """Get the NSP scores of a batch.

    Args:
        nsp_predictor: The NSP model predictor.
        predictions: A batch of prediction, contains `context_token_ids` and `response_token_ids`.

    Returns:
        Add `nsp_score` to each prediction.
    """
    import argparse
    from collections import namedtuple

    from knover.data.nsp_reader import NSPReader
    from knover.tasks.next_sentence_prediction import NextSentencePrediction
    from knover.utils.args import parse_args

    parser = argparse.ArgumentParser()
    NextSentencePrediction.add_cmdline_args(parser)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--mem_efficient", type=str2bool, default=False)

    args = parse_args(parser, allow_unknown=True)
    args.load(args.config_path)
    if not args.mem_efficient:
        if args.num_samples:
            args.batch_size *= args.num_samples
        if args.latent_type_size:
            args.batch_size *= args.latent_type_size
    args.tokenized_input = True
    args.use_mlm = False
    reader = NSPReader(args)

    def __reader__():
        headers = ["src", "tgt", "data_id"]

        Example = namedtuple("Example", headers)

        for i, pred in enumerate(predictions):
            context = post_process_context(pred["context_token_ids"], reader, merge=False)
            ctx_tokenized_input = " [SEP] ".join(" ".join(utt) for utt in context)
            _, response = post_process_response(pred["response_token_ids"], reader, merge=False)
            response_tokenized_input = " ".join(response)
            example = Example(
                src=ctx_tokenized_input,
                tgt=response_tokenized_input,
                data_id=i
            )
            record = reader._convert_example_to_record(example, is_infer=True)
            yield record
        return

    generator = reader.data_generator(
        reader=__reader__,
        is_infer=True,
        phase="test",
    )

    steps = 0
    for data in generator():
        outputs = nsp_predictor(data)
        for probs, data_id in zip(outputs[0], outputs[-1]):
            data_id = data_id[0]
            pred = predictions[data_id]
            pred["nsp_score"] = float(probs[1])

    return
