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

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.fluid.layers as layers

from models.optimizer import AdamW
from utils import init_pretraining_params, init_checkpoint, to_lodtensor
from utils.args import str2bool


class Model(ABC):
    """
    Basic model wrapper for paddle.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = parser.add_argument_group("Model")
        # Init checkpoint
        group.add_argument("--init_checkpoint", type=str, default="")
        group.add_argument("--init_pretraining_params", type=str, default="")

        # Optimizer
        group.add_argument("-lr", "--learning_rate", type=float, default=1e-5,
                           help="The learning rate for optimizer.")
        group.add_argument("--warmup_steps", type=int, default=0,
                           help="The warmup steps.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for optimizer.")
        group.add_argument("--max_grad_norm", type=float, default=.1,
                           help="The maximum norm of gradient.")

        group.add_argument("--use_recompute", type=str2bool, default=False)
        group.add_argument("--use_amp", type=str2bool, default=False)
        group.add_argument("--amp_loss_scaling", type=float, default=12800)
        return group

    def __init__(self, args, place):
        self.place = place
        self.exe = fluid.Executor(place)

        self.init_checkpoint = args.init_checkpoint
        self.init_pretraining_params = args.init_pretraining_params

        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm

        self.is_distributed = args.is_distributed
        self.use_recompute = args.use_recompute
        self.use_amp = args.use_amp
        self.amp_loss_scaling = args.amp_loss_scaling
        self.run_infer = args.get("run_infer", False)
        self.batch_size = args.get("batch_size", 1)
        self._build_programs()
        return

    def _build_programs(self):
        """
        Build programs.

        Build train_program, eval_program and inference_program. Only use in static graph mode.
        """
        if self.run_infer:
            self.startup_program = fluid.Program()
            # build infer program
            self.infer_program = fluid.Program()
            with fluid.program_guard(self.infer_program, self.startup_program):
                with fluid.unique_name.guard():
                    self.infer_feed_dict = inputs = self._get_feed_dict(is_infer=True)
                    outputs = self.forward(inputs, is_infer=True)
                    predictions = self.infer(inputs, outputs)
                    self.infer_fetch_dict = predictions
            self.infer_program = self.infer_program.clone(for_test=True)

            self.program = self.infer_program
        else:
            if self.is_distributed:
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
                    dist_strategy.enable_sequential_execution=True
                if self.use_amp:
                    dist_strategy.use_amp = True
                    dist_strategy.amp_loss_scaling = self.amp_loss_scaling
                self.dist_strategy = dist_strategy

            self.startup_program = fluid.Program()
            # build train program
            self.train_program = fluid.Program()
            with fluid.program_guard(self.train_program, self.startup_program):
                with fluid.unique_name.guard():
                    self.feed_dict = inputs = self._get_feed_dict()
                    outputs = self.forward(inputs)
                    if self.is_distributed and self.use_recompute:
                        self.dist_strategy.recompute_checkpoints = outputs["checkpoints"]
                    metrics, statistics = self.get_metrics_and_statistics(inputs, outputs)

                    # build eval program
                    self.eval_program = self.train_program.clone(for_test=True)
                    self.eval_fetch_dict = {**metrics, **statistics}

                    scheduled_lr = self.optimize(metrics)
                    metrics["scheduled_lr"] = scheduled_lr
                    self.train_fetch_dict = metrics

            self.program = self.train_program
            if self.is_distributed:
                self.train_program = fleet.main_program

        self.exe.run(self.startup_program)
        if self.init_pretraining_params != "":
            init_pretraining_params(self.exe, self.init_pretraining_params, self.program)
        elif self.init_checkpoint != "":
            init_checkpoint(self.exe, self.init_checkpoint, self.program)
        return

    def load(self, model_dir, is_checkpoint=False):
        """
        Load persistables or parameters.
        """
        # TODO: support dygraph.
        if is_checkpoint:
            init_checkpoint(self.exe, model_dir, self.program)
        else:
            init_pretraining_params(self.exe, model_dir, self.program)
        return

    def save(self, model_dir, is_checkpoint=False):
        """
        Save persistables or parameters.
        """
        # TODO: support dygraph.
        if is_checkpoint:
            fluid.io.save_persistables(self.exe, model_dir, self.program)
        else:
            fluid.io.save_params(self.exe, model_dir, self.program)
        return

    @abstractmethod
    def _get_feed_dict(self, is_infer=False):
        """
        Return input feed list.
        """
        pass

    def _get_feed(self, inputs, is_infer=False):
        """
        Convert `inputs` into model's feed data format.
        """
        if isinstance(inputs, list):
            # return list direclty which is used in `get_data_loader`.
            return inputs
        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = to_lodtensor(inputs[k], self.place)
        return inputs

    def get_data_loader(self, generator=None, is_infer=False):
        """
        Return DataLoader.

        If generator is not `None`, the data loader set it as the batch generator.
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
    def forward(self, inputs, is_infer=False):
        """
        Run model main forward.
        """
        pass

    @abstractmethod
    def get_metrics_and_statistics(self, inputs, outputs):
        """
        Get metrics and statistics.
        """
        pass

    @abstractmethod
    def infer(self, inputs, outputs):
        """
        Run model inference.
        """
        pass

    def optimize(self, metrics):
        """
        Optimize the model by metrics(mainly `metrics["loss"]`).
        """
        # TODO: support dygraph
        if self.warmup_steps > 0:
            scheduled_lr = layers.learning_rate_scheduler.noam_decay(
                1 / (self.warmup_steps * (self.learning_rate ** 2)),
                self.warmup_steps)
        else:
            scheduled_lr = layers.create_global_var(
                name=fluid.unique_name.generate("learning_rate"),
                shape=[1],
                value=self.learning_rate,
                dtype="float32",
                persistable=True)
        grad_clip = fluid.clip.GradientClipByGlobalNorm(self.max_grad_norm)

        self.optimizer = AdamW(
            learning_rate=scheduled_lr,
            grad_clip=grad_clip,
            weight_decay=self.weight_decay)

        if self.is_distributed:
            self.optimizer = fleet.distributed_optimizer(self.optimizer, strategy=self.dist_strategy)

        self.optimizer.minimize(metrics["loss"])
        return scheduled_lr

    def _execute(self, program, feed, fetch_dict, **kwargs):
        """
        Execute program.
        """
        fetch_list = [var.name for var in fetch_dict.values()]
        fetch_vars = self.exe.run(program, feed, fetch_list, **kwargs)
        return dict(zip(fetch_dict.keys(), fetch_vars))

    def train_step(self, inputs):
        """
        Run one training step.
        """
        # TODO: support dygraph.
        return self._execute(
            self.train_program,
            self._get_feed(inputs),
            self.train_fetch_dict,
            use_program_cache=True)

    def eval_step(self, inputs):
        """
        Run one evaluation step.
        """
        # TODO: support dygraph.
        return self._execute(
            self.eval_program,
            self._get_feed(inputs),
            self.eval_fetch_dict)

    def infer_step(self, inputs):
        """
        Run one inference step.
        """
        # TODO: support dygraph.
        return self._execute(
            self.infer_program,
            self._get_feed(inputs, is_infer=True),
            self.infer_fetch_dict)

    def save_inference_model(self, inference_model_path):
        """
        Save the inference model.
        """
        feed_list = [var.name for var in self.infer_feed_dict.values()]
        fetch_list = list(self.infer_fetch_dict.values())

        fluid.io.save_inference_model(
            inference_model_path,
            feed_list,
            fetch_list,
            self.exe,
            self.infer_program)
