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

from readers.dialog_reader import DialogReader
from readers.plato_reader import PlatoReader
from tasks import register_task
from tasks.task_base import Task
from utils.args import str2bool
from utils.inference import create_predictor


def post_process_context(token_ids, reader, merge=True):
    """Post-process the context sequence."""
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
    """
    Post-process the decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
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


def get_cross_turn_repetition(context, pred_tokens, eos_idx, is_cn=False):
    """Get cross-turn repetition."""
    if len(pred_tokens) == 0:
        return 1.0
    if is_cn:
        context = ["".join(utt) for utt in context]
        pred_tokens = "".join(pred_tokens)

    pred_tri_grams = set()
    for i in range(len(pred_tokens) - 2):
        tri_gram = tuple(pred_tokens[i:i + 3])
        pred_tri_grams.add(tri_gram)
    for utt in context:
        for i in range(len(utt) - 2):
            tri_gram = tuple(utt[i:i + 3])
            if tri_gram in pred_tri_grams:
                return 1.0
    return 0.0


def get_in_turn_repetition(pred, is_cn=False):
    """Get in-turn repetition."""
    if len(pred) == 0:
        return 1.0
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)
    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return 1.0
        tri_grams.add(tri_gram)
    return 0.0


def get_nsp_score_batch(nsp_predictor, predictions):
    """
    Get NSP scores of a batch.
    """
    import argparse
    from collections import namedtuple

    from readers.nsp_reader import NSPReader
    from utils.args import parse_args
    from tasks.next_sentence_prediction import NextSentencePrediction

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
    reader = NSPReader(args)

    def __reader__():
        headers = ["src", "tgt", "data_id"]

        Example = namedtuple("Example", headers)

        for i, info in enumerate(predictions):
            context = post_process_context(info["context_token_ids"], reader, merge=False)
            context_tokenized_input = " [SEP] ".join(" ".join(utt) for utt in context)
            _, response = post_process_response(info["response_token_ids"], reader, merge=False)
            response_tokenized_input = " ".join(response)
            example = Example(src=context_tokenized_input, tgt=response_tokenized_input, data_id=i)
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
            info = predictions[data_id]
            info["nsp_score"] = float(probs[1])

    return


@register_task("DialogGeneration")
class DialogGeneration(Task):
    """
    Define dialogue response generation.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = parser.add_argument_group("Task")
        group.add_argument("--do_generation", type=str2bool, default=False)
        group.add_argument("--is_cn", type=str2bool, default=False)

        group.add_argument("--nsp_inference_model_path", type=str, default=None)
        group.add_argument("--nsp_attention_style", type=str, default="bidirectional")

        group.add_argument("--ranking_score", type=str, default="decode_score")

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
        if args.model == "Plato":
            self.reader = PlatoReader(args)
        else:
            self.reader = DialogReader(args)

        if args.nsp_inference_model_path:
            self.nsp_predictor = create_predictor(args.nsp_inference_model_path, args.is_distributed)
            self.nsp_attention_style = args.nsp_attention_style
        else:
            self.nsp_predictor = None

        self.ranking_score = args.ranking_score
        self.max_dec_len = args.max_dec_len
        return

    def _post_process_generation_output(self, predictions):
        """
        Post process generation output.

        Calculate repetion, reranking.
        """
        for info in predictions:
            tokens = post_process_context(info["context_token_ids"], self.reader)
            pred_token_ids, pred_tokens = post_process_response(info["response_token_ids"], self.reader)
            info["context"] = " [SEP] ".join(" ".join(u) for u in tokens)
            info["response"] = " ".join(pred_tokens)
            info["num_token"] = len(pred_token_ids)
            info["cross_turn_repetition"] = get_cross_turn_repetition(
                tokens, pred_tokens, self.reader.eos_id, self.is_cn)
            info["in_turn_repetition"] = max(
                get_in_turn_repetition(pred_tokens, self.is_cn),
                get_in_turn_repetition(pred_token_ids))
        if self.nsp_predictor is not None:
            get_nsp_score_batch(self.nsp_predictor, predictions)

        group = defaultdict(list)
        for info in predictions:
            group[info["data_id"]].append(info)

        predictions = []
        for data_id in group:
            infos = group[data_id]
            for info in infos:
                info["score"] = info[self.ranking_score]
                if self.max_dec_len is not None and info["num_token"] >= self.max_dec_len: # not ending
                    info["score"] -= 1e3
                elif info["cross_turn_repetition"] > 0:
                    info["score"] -= 1e3
                elif info["in_turn_repetition"] > 0:
                    info["score"] -= 1e3
            infos = sorted(infos, key=lambda info: -info["score"])
            pred = infos[0]
            keep_attr = ["data_id", "score", "response"]
            pred = {k: pred[k] for k in keep_attr}
            predictions.append(pred)
        return predictions

    def _post_process_scoring_output(self, predictions):
        raise NotImplementedError

    def _post_process_infer_output(self, predictions):
        if self.do_generation:
            return self._post_process_generation_output(predictions)
        else:
            return self._post_process_scoring_output(predictions)

    def merge_mertrics_and_statistics(self, outputs, part_outputs):
        """
        Merge two evaulation output.
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
        """
        Get metrics.
        """
        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        outputs.pop("batch_size", None)
        outputs.pop("tokens_num", None)
        metrics = {}
        for k in outputs:
            if k.startswith("token_"):
                metrics[k[6:]] = outputs[k]
            else:
                metrics[k] = outputs[k]
            if k == "token_lm_loss":
                metrics["ppl"] = math.exp(outputs[k])
        return metrics
