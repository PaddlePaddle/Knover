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
"""Model base."""

from abc import abstractmethod, ABC
import os
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.fluid.layers as layers

import knover.optim
import knover.optim.lr_scheduler as lr_scheduler
from knover.utils import to_lodtensor, get_tensor
from knover.utils.args import str2bool


class Model(ABC):
    """Basic model wrapper of PaddlePaddle.

    Attributes:
        place: CPUPlace or CUDAPlace.
        exe: A executor which is used in static graph mode.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Model")

        # initialize model
        group.add_argument("--init_checkpoint", type=str, default="")
        group.add_argument("--init_pretraining_params", type=str, default="")

        # optimizer related
        group.add_argument("--optimizer", type=str, default="AdamW",
                           choices=["AdamW"])
        group.add_argument("-lr", "--learning_rate", type=float, default=1e-5,
                           help="The peak learning rate for optimizer.")
        group.add_argument("--warmup_steps", type=int, default=0,
                           help="The warmup steps.")
        group.add_argument("--lr_scheduler", type=str, default="noam",
                           choices=["linear", "noam", "constant", "cosine"],
                           help="The learning rate scheduler for training.")
        group.add_argument("--max_training_steps", type=int, default=2000,
                           help="The maximum training step used in linear or cosine decay lr_scheduler.")
        group.add_argument("--min_learning_rate", type=float, default=0,
                           help="The minimum learning rate used in linear or cosine decay lr_scheduler.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for optimizer.")
        group.add_argument("--max_grad_norm", type=float, default=.1,
                           help="The maximum norm of gradient.")

        # training related
        group.add_argument("--use_recompute", type=str2bool, default=False,
                           help="Whether to use recompute for saving memory.")
        group.add_argument("--use_amp", type=str2bool, default=False,
                           help="Whether to use automatic mixed precision(AMP) training")
        group.add_argument("--amp_loss_scaling", type=float, default=32768.,
                           help="The initial loss scaling of AMP.")

        return group

    def __init__(self, args, place):
        self.place = place
        self.exe = fluid.Executor(place)

        self.init_checkpoint = args.init_checkpoint
        self.init_pretraining_params = args.init_pretraining_params

        # optimizer related
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.lr_scheduler = args.lr_scheduler
        self.max_training_steps = args.max_training_steps
        self.min_learning_rate = args.min_learning_rate
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm

        # training related
        self.is_distributed = args.get("is_distributed", False)
        self.use_recompute = args.use_recompute
        self.use_amp = args.use_amp
        self.amp_loss_scaling = args.amp_loss_scaling
        if not self.is_distributed:
            if self.use_recompute:
                print("[WARM] Cannot support recomputation in non-distributed mode.")
            if self.use_amp:
                print("[WARM] Cannot support AMP in non-distributed mode.")

        # model mode
        self.run_infer = args.get("run_infer", False)
        self.batch_size = args.get("batch_size", 1)

        self._build_programs()
        return

    def _init_distributed_strategy(self):
        """Initialize distributed strategy."""
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        exec_strategy.num_iteration_per_drop_scope = 1

        dist_strategy = DistributedStrategy()
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.nccl_comm_num = 1
        dist_strategy.fuse_all_reduce_ops = True
        if self.use_recompute:
            dist_strategy.forward_recompute = True
            dist_strategy.enable_sequential_execution = True
        if self.use_amp:
            dist_strategy.use_amp = True
            dist_strategy.amp_loss_scaling = self.amp_loss_scaling
        self.dist_strategy = dist_strategy
        return

    def _set_checkpoints(self, checkpoints):
        """Set checkpoints for recompute.

        Args:
            checkpoints: A list of Variables which need to set as checkpoints.
        """
        self.dist_strategy.recompute_checkpoints = checkpoints
        return

    def _build_programs(self):
        """Build programs.

        Build training program, evaluation program and inference program. Only use in static graph mode.
        """
        self.startup_program = fluid.Program()
        if self.run_infer:
            if self.is_distributed:
                self._init_distributed_strategy()
            # build inference program
            self.infer_program = fluid.Program()
            with fluid.program_guard(self.infer_program, self.startup_program):
                with fluid.unique_name.guard():
                    self.infer_feed_dict = inputs = self._get_feed_dict(is_infer=True)
                    outputs = self.forward(inputs, is_infer=True)
                    
                    predictions, sharding_info = self.infer(inputs, outputs)
                    self.infer_fetch_dict = predictions

                    
            
            self.without_beam_program = fluid.Program()
            with fluid.program_guard(self.without_beam_program, fluid.Program()):
                with fluid.unique_name.guard():
                    inputs = self._get_feed_dict(is_infer=True)
                    max_len = layers.fill_constant(shape=[1], dtype="int64", value=sharding_info["max_dec_len"], force_cpu=True)
                    min_len = layers.fill_constant(shape=[1], dtype="int64", value=sharding_info["min_dec_len"], force_cpu=True)
                    step_idx = layers.fill_constant(shape=[1], dtype="int64", value=0, force_cpu=True)
                    ids = layers.array_write(layers.reshape(inputs["tgt_ids"], (-1, 1)), step_idx)
                    pos_biases = layers.array_write(layers.reshape(inputs["tgt_pos"], (-1, 1)), step_idx)
                    scores = layers.array_write(inputs["init_score"], step_idx)
                    tgt_generation_mask = layers.array_write(inputs["tgt_generation_mask"], step_idx)
                    parent_idx = inputs["parent_idx"]
                    eos_penalty = np.zeros(sharding_info["vocab_size"], dtype="float32")
                    eos_penalty[sharding_info["eos_id"]] = -1e9
                    eos_penalty = layers.assign(eos_penalty)

                    token_penalty = np.zeros(sharding_info["vocab_size"], dtype="float32")
                    token_penalty[sharding_info["unk_id"]] = -1e9
                    if sharding_info["mask_id"] >= 0:
                        token_penalty[sharding_info["mask_id"]] = -1e9
                    token_penalty = layers.assign(token_penalty)
                    # start while loop
                    cond = layers.less_than(x=step_idx, y=max_len)
                    while_op = layers.While(cond)
                    with while_op.block():
                        pre_ids = layers.array_read(array=ids, i=step_idx)
                        pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
                        pre_scores = layers.array_read(array=scores, i=step_idx)
                        pos_bias = layers.array_read(array=pos_biases, i=step_idx)
                        pos_bias = layers.gather(input=pos_bias, index=parent_idx)
                        tmp_tgt_generation_mask = layers.array_read(tgt_generation_mask, i=step_idx)
                        dtype = tmp_tgt_generation_mask.dtype
                        append_mask = layers.fill_constant_batch_size_like(
                                input=pre_ids,
                                value=1.0,
                                shape=[-1, 1, 1],
                                dtype=dtype)
                        tmp_tgt_generation_mask = layers.concat([tmp_tgt_generation_mask, append_mask], axis=2)
                        pre_mask = tmp_tgt_generation_mask = layers.gather(input=tmp_tgt_generation_mask, index=parent_idx)
                        pre_sent = layers.fill_constant_batch_size_like(
                                input=pre_mask,
                                value=1,
                                shape=[-1, 1, 1],
                                dtype=pre_ids.dtype)

                        pre_pos = layers.elementwise_add(
                            layers.elementwise_mul(
                                x=layers.fill_constant_batch_size_like(
                                    input=pre_mask,
                                    value=1,
                                    shape=[-1, 1, 1],
                                    dtype=pre_ids.dtype), y=step_idx, axis=0),
                            pos_bias, axis=0)
                        if sharding_info["use_role"]:
                            pre_role = layers.fill_constant_batch_size_like(
                                input=pre_mask,
                                value=0,
                                shape=[-1, 1, 1],
                                dtype=pre_ids.dtype)
                        else:
                            pre_role = None
                        #import pdb
                        #pdb.set_trace()
                        dec_out, _ = sharding_info["model"]._generation_network(
                            token_ids=pre_ids,
                            type_ids=pre_sent,
                            pos_ids=pre_pos,
                            role_ids=pre_role,
                            generation_mask=tmp_tgt_generation_mask,
                            gather_idx=parent_idx)
                        logits = sharding_info["model"]._calc_logits(dec_out)

                        if sharding_info["ignore_unk"]:
                            logits = layers.elementwise_add(logits, token_penalty, axis=1)
                        min_len_cond = layers.less_than(x=step_idx, y=min_len)
                        def min_len_penalty():
                            """Plus minimum length penalty."""
                            return layers.elementwise_add(logits, eos_penalty, axis=1)
                        def no_penalty():
                            """No penalty."""
                            return logits
                        logits = layers.case([(min_len_cond, min_len_penalty)], default=no_penalty)


                        # get probs
                        probs = layers.softmax(logits / sharding_info["temperature"])

                        if sharding_info["decoding_strategy"] == "beam_search":
                            topk_scores, topk_indices = layers.topk(
                                input=probs, k=sharding_info["beam_size"])
                        else:
                            if sharding_info["decoding_strategy"].startswith("sampling"):
                                sampling_ids = layers.sampling_id(probs, dtype="int")
                            elif sharding_info["decoding_strategy"].startswith("topk_sampling"):
                                topk_probs, _ = layers.topk(input=probs, k=sharding_info["topk"])
                                ge_cond = layers.cast(
                                    layers.greater_equal(
                                        probs,
                                        layers.unsqueeze(topk_probs[:, -1], [1])),
                                    "float32")
                                old_probs = probs
                                probs = probs * ge_cond / layers.reduce_sum(topk_probs, dim=-1, keep_dim=True)
                                sampling_ids = layers.sampling_id(probs, dtype="int")
                                probs = old_probs
                            elif sharding_info["decoding_strategy"].startswith("topp_sampling"):
                                sorted_probs, sorted_idx = layers.argsort(probs, descending=True)
                                cum_sorted_probs = layers.cumsum(sorted_probs, axis=1, exclusive=True)
                                lt_cond = layers.cast(
                                    layers.less_than(
                                        cum_sorted_probs,
                                        layers.fill_constant_batch_size_like(
                                            cum_sorted_probs,
                                            cum_sorted_probs.shape,
                                            cum_sorted_probs.dtype,
                                            sharding_info["topp"])
                                    ),
                                    "float32"
                                )
                                old_probs = probs
                                candidate_probs = sorted_probs * lt_cond
                                probs = candidate_probs / layers.reduce_sum(candidate_probs, dim=-1, keep_dim=True)
                                sampling_ids = layers.sampling_id(probs, dtype="int")
                                sampling_ids = layers.index_sample(sorted_idx, layers.unsqueeze(sampling_ids, [1]))
                                sampling_ids = layers.squeeze(sampling_ids, [1])
                                probs = old_probs
                            else:
                                raise ValueError(sharding_info["decoding_strategy"])
                            sampling_scores = layers.one_hot(
                                layers.unsqueeze(sampling_ids, [1]), probs.shape[1]
                            )
                            sampling_scores = sampling_scores * probs - (1 - sampling_scores) * 1e3
                            topk_scores, topk_indices = layers.topk(
                                input=sampling_scores, k=1)
                        pre_len = layers.cast(step_idx, "float32")
                        layers.increment(x=step_idx, value=1.0, in_place=True)
                        cur_len = layers.cast(step_idx, "float32")
                        # update scores
                        if sharding_info["length_average"]:
                            accu_scores = layers.elementwise_add(
                                x=layers.log(topk_scores), y=pre_scores * pre_len, axis=0) / cur_len
                        elif sharding_info["length_penalty"] > 0:
                            pre_lp = layers.pow((5 + pre_len) / 6, sharding_info["length_penalty"])
                            cur_lp = layers.pow((5 + cur_len) / 6, sharding_info["length_penalty"])
                            accu_scores = layers.elementwise_add(
                                x=layers.log(topk_scores), y=pre_scores * pre_lp, axis=0) / cur_lp
                        else:
                            accu_scores = layers.elementwise_add(
                                x=layers.log(topk_scores), y=pre_scores, axis=0)
                        topk_indices = layers.lod_reset(topk_indices, pre_ids)
                        accu_scores = layers.lod_reset(accu_scores, pre_ids)
                        selected_ids, selected_scores, gather_idx = layers.beam_search(
                            pre_ids=pre_ids,
                            pre_scores=pre_scores,
                            ids=topk_indices,
                            scores=accu_scores,
                            beam_size=sharding_info["beam_size"],
                            end_id=sharding_info["eos_id"],
                            return_parent_idx=True)
                        layers.array_write(selected_ids, i=step_idx, array=ids)
                        layers.array_write(selected_scores, i=step_idx, array=scores)
                        layers.array_write(pre_mask, i=step_idx, array=tgt_generation_mask)
                        layers.array_write(pos_bias, i=step_idx, array=pos_biases)

                        layers.assign(gather_idx, parent_idx)

                        length_cond = layers.less_than(x=step_idx, y=max_len)
                        finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
                        layers.logical_and(x=length_cond, y=finish_cond, out=cond)
                    
            print("====================ahahahahah============================")




        
            # sharding for without_beam_program
            
            self.without_beam_program = self.without_beam_program.clone(for_test=True)
            with open("sharding_program_no_grad.txt", "w") as f:
                f.write(str(self.without_beam_program))
                print("sharding_program ================")



            self.infer_program = self.infer_program.clone(for_test=True)

            self.program = self.infer_program
            
        else:
            # initialize distributed setting
            if self.is_distributed:
                self._init_distributed_strategy()

            # build training program
            self.train_program = fluid.Program()
            with fluid.program_guard(self.train_program, self.startup_program):
                with fluid.unique_name.guard():
                    self.feed_dict = inputs = self._get_feed_dict()
                    outputs = self.forward(inputs)

                    if self.is_distributed and self.use_recompute:
                        self._set_checkpoints(outputs["checkpoints"])

                    metrics = self.get_metrics(inputs, outputs)

                    # build evaluation program
                    self.eval_program = self.train_program.clone(for_test=True)
                    self.eval_fetch_dict = dict(**metrics)

                    scheduled_lr = self.optimize(metrics)
                    metrics["scheduled_lr"] = scheduled_lr
                    if self.is_distributed and self.use_amp:
                        vars_ = fluid.default_main_program().global_block().vars
                        loss_scaling = vars_["loss_scaling_0"]
                        metrics["loss_scaling"] = loss_scaling
                    self.train_fetch_dict = metrics

            self.program = self.train_program
            if self.is_distributed:
                self.train_program = fleet.main_program

        # initialize model
        self.exe.run(self.startup_program)
        if self.init_pretraining_params != "":
            self.load(self.init_pretraining_params)
        elif self.init_checkpoint != "":
            self.load(self.init_checkpoint, is_checkpoint=True)
        return

    def load(self, model_path, is_checkpoint=False):
        """Load persistables or parameters.

        Args:
            model_path: The path of initial model.
            is_checkpoint: If true, load parameters and other variables (such as moments in Adam optimizer), otherwise load only parameters.
        """
        # TODO: support dygraph.
        print(f"Loading model from {model_path}.")
        assert os.path.exists(model_path), f"[{model_path}] cann't be found."
        def __predicate__(var):
            if is_checkpoint and not fluid.io.is_persistable(var):
                return False
            if not is_checkpoint and not fluid.io.is_parameter(var):
                return False
            # only load existing variable.
            return os.path.exists(os.path.join(model_path, var.name))
        fluid.io.load_vars(
            self.exe,
            model_path,
            main_program=self.program,
            predicate=__predicate__)
        if is_checkpoint:
            print(f"Load model from checkpoint: {model_path}")
            start_step = get_tensor("@LR_DECAY_COUNTER@")
            if start_step is not None:
                self.args.start_step = start_step[0]
        else:
            print(f"Load pretraining parameters from {model_path}")
        return

    def save(self, model_path, is_checkpoint=False):
        """Save persistables or parameters into the given path.

        Args:
            model_path: The path where we save the model.
            is_checkpoint: If true, save parameters and other variables (such as moments in Adam optimizer), otherwise save only parameters.
        """
        # TODO: support dygraph.
        if is_checkpoint:
            fluid.io.save_persistables(self.exe, model_path, self.program)
        else:
            fluid.io.save_params(self.exe, model_path, self.program)
        return

    def _get_feed(self, inputs):
        """Convert inputs into model's input data format.

        Convert hierarchical list into LoD Tensor, and keep numpy.ndarray.

        Args:
            inputs: A dict mapping keys to corresponding data. The data is either a hierarchical list or a numpy.ndarray.

        Returns:
            inputs: A dict mapping keys to corresponding model's data. The data is either a LoDTensor or a numpy.ndarray.
        """
        if isinstance(inputs, list):
            # return list direclty which is used in `get_data_loader`.
            return inputs
        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = to_lodtensor(inputs[k], self.place)
        return inputs

    def get_data_loader(self, generator=None, is_infer=False):
        """Get the DataLoader of the model.

        Args:
            generator: If generator is not `None`, the DataLoader sets it as the batch generator.
            is_infer: If true, get inference's DataLoader, otherwise get training / evaluation 's DataLoader.

        Returns:
            loader: A DataLoader which is used to generate batch data.
        """
        # TODO: support dygraph.
        if is_infer:
            feed_name_list, feed_list = zip(*self.infer_feed_dict.items())
        else:
            feed_name_list, feed_list = zip(*self.feed_dict.items())
        loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list,
            capacity=64,
            use_double_buffer=True,
            iterable=True)
        if generator is not None:
            def __wrapper__():
                for batch in generator():
                    batch = self._get_feed(batch)
                    batch = [batch[name] for name in feed_name_list if name in batch]
                    yield batch
            loader.set_batch_generator(__wrapper__, self.place)
        return loader

    @abstractmethod
    def _get_feed_dict(self, is_infer=False):
        """Get model's input feed dict.

        Args:
            is_infer: If true, get inference input feed dict, otherwise get training / evaluation input feed dict.

        Returns:
            feed_dict: A feed dict mapping keys to feed input variable.
        """
        pass

    @abstractmethod
    def forward(self, inputs, is_infer=False):
        """Run model main forward.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            outputs: A dict mapping keys to corresponding output data.
        """
        pass

    @abstractmethod
    def get_metrics(self, inputs, outputs):
        """Get metrics.

        Args:
            inputs: A dict mapping keys to corresponding input data.
            outputs: A dict mapping keys to corresponding output data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        pass

    @abstractmethod
    def get_statistics(self, inputs, outputs):
        """Get statistics.

        Args:
            inputs: A dict mapping keys to corresponding input data.
            outputs: A dict mapping keys to corresponding output data.

        Returns:
            statistics: A dict mapping keys to corresponding statistics.
        """
        pass

    @abstractmethod
    def infer(self, inputs, outputs):
        """Run inference.

        Args:
            inputs: A dict mapping keys to corresponding input data.
            outputs: A dict mapping keys to corresponding output data.

        Returns:
            predictions: A dict mapping keys to corresponding predictions.
        """
        pass

    def optimize(self, metrics):
        """Optimize the model by loss.

        Args:
            metrics: A dict mapping metric names to corresponding metrics, which must include loss.
        """
        # TODO: support dygraph
        # lr scheduler
        if self.lr_scheduler == "noam" and self.warmup_steps <= 0:
            print("[WARMING] Using constant learning rate because of `warmup_steps` is not positive while using NoamScheduler.")
        if self.lr_scheduler == "noam" and self.warmup_steps > 0:
            scheduled_lr = layers.learning_rate_scheduler.noam_decay(
                1 / (self.warmup_steps * (self.learning_rate ** 2)),
                self.warmup_steps)
        elif self.lr_scheduler == "linear":
            scheduled_lr = lr_scheduler.linear_warmup_and_linear_decay(
                self.learning_rate,
                self.min_learning_rate,
                self.warmup_steps,
                self.max_training_steps)
        elif self.lr_scheduler == "cosine":
            scheduled_lr = lr_scheduler.linear_warmup_and_cosine_decay(
                self.learning_rate,
                self.min_learning_rate,
                self.warmup_steps,
                self.max_training_steps)
        else: # constant
            scheduled_lr = layers.create_global_var(
                name=fluid.unique_name.generate("learning_rate"),
                shape=[1],
                value=self.learning_rate,
                dtype="float32",
                persistable=True)
        # grad norm
        if self.max_grad_norm > 0:
            grad_clip = fluid.clip.GradientClipByGlobalNorm(self.max_grad_norm)
        else:
            grad_clip = None

        # optimizer
        optimizer_cls = getattr(knover.optim, self.optimizer)
        optimizer = optimizer_cls(
            learning_rate=scheduled_lr,
            grad_clip=grad_clip,
            weight_decay=self.weight_decay)

        # distributed optimizer
        if self.is_distributed:
            optimizer = fleet.distributed_optimizer(optimizer, strategy=self.dist_strategy)

        optimizer.minimize(metrics["loss"])
        return scheduled_lr

    def _execute(self, program, inputs, fetch_dict, **kwargs):
        """Execute program in static graph mode.

        Args:
            program: The executable program.
            inputs: A dict mapping variable names to corresponding input variables.
            fetch_dict: A dict mapping variable names to corresponding output variables.
            kwargs: Other arguments for executing program.

        Returns:
            outputs: A dict mapping keys to output variables.
        """
        feed = self._get_feed(inputs)
        fetch_list = [var.name for var in fetch_dict.values()]
        fetch_vars = self.exe.run(program, feed, fetch_list, **kwargs)
        return dict(zip(fetch_dict.keys(), fetch_vars))

    def train_step(self, inputs):
        """Run one training step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        # TODO: support dygraph.
        return self._execute(
            self.train_program,
            inputs,
            self.train_fetch_dict,
            use_program_cache=True)

    def eval_step(self, inputs):
        """Run one evaluation step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        # TODO: support dygraph.
        outputs = self._execute(
            self.eval_program,
            inputs,
            self.eval_fetch_dict)
        if isinstance(inputs, list):
            inputs = inputs[0]
        statistics = self.get_statistics(inputs, outputs)
        outputs.update(statistics)
        return outputs

    def infer_step(self, inputs):
        """Run one inference step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            predictions: A dict mapping keys to corresponding predictions.
        """
        # TODO: support dygraph.
        return self._execute(
            self.infer_program,
            inputs,
            self.infer_fetch_dict)

    def save_inference_model(self, inference_model_path):
        """Save the inference model.

        Only save the inference program.

        Args:
            inference_model_path: The path of saved inference model.
        """
        feed_list = [var.name for var in self.infer_feed_dict.values()]
        fetch_list = list(self.infer_fetch_dict.values())

        fluid.io.save_inference_model(
            inference_model_path,
            feed_list,
            fetch_list,
            self.exe,
            self.infer_program,
            program_only=True)