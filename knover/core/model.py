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

import paddle
from paddle.distributed import init_parallel_env
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.optimizer.lr as lr

import knover.optim
from knover.optim.lr_scheduler import CosineDecay
from knover.utils import to_lodtensor, get_tensor, str2bool
from knover.utils.topo import Topology


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
        group.add_argument("--beta1", type=float, default=0.9)
        group.add_argument("--beta2", type=float, default=0.999)
        group.add_argument("--warmup_steps", type=int, default=0,
                           help="The warmup steps.")
        group.add_argument("--lr_scheduler", type=str, default="noam",
                           choices=["linear", "noam", "constant", "cosine"],
                           help="The learning rate scheduler for training.")
        group.add_argument("--max_decay_steps", type=int, default=2000,
                           help="The maximum training step used in linear or cosine decay lr_scheduler.")
        group.add_argument("--min_learning_rate", type=float, default=0,
                           help="The minimum learning rate used in linear or cosine decay lr_scheduler.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for optimizer.")
        group.add_argument("--max_grad_norm", type=float, default=.1,
                           help="The maximum norm of gradient.")

        # fleet related
        group.add_argument("--use_recompute", type=str2bool, default=False,
                           help="Whether to use recompute for saving memory.")
        group.add_argument("--checkpointing_every_n_layers", type=int, default=1,
                           help="Create checkpointing every n layers.")
        group.add_argument("--use_amp", type=str2bool, default=False,
                           help="Whether to use automatic mixed precision(AMP) training")
        group.add_argument("--amp_loss_scaling", type=float, default=32768.,
                           help="The initial loss scaling of AMP.")
        group.add_argument("--use_sharding", type=str2bool, default=False,
                           help="Whether to use sharding strategy.")
        group.add_argument("--dp_degree", type=int, default=1, help="Data parallism degree.")
        group.add_argument("--sharding_degree", type=int, default=1, help="Sharding parallism degree.")
        group.add_argument("--mp_degree", type=int, default=1, help="Tensor model parallism degree.")
        group.add_argument("--pp_degree", type=int, default=1, help="Pipeline model parallism degree.")

        return group

    def __init__(self, args, place):
        self.args = args

        self.place = place

        self.init_checkpoint = args.init_checkpoint
        self.init_pretraining_params = args.init_pretraining_params

        # fleet related
        self.is_distributed = args.get("is_distributed", False)
        self.use_recompute = args.use_recompute
        self.checkpointing_every_n_layers = args.checkpointing_every_n_layers
        self.use_amp = args.use_amp
        self.amp_loss_scaling = args.amp_loss_scaling
        self.use_sharding = args.use_sharding
        self.dp_degree = args.dp_degree
        self.sharding_degree = args.sharding_degree
        self.mp_degree = args.mp_degree
        self.pp_degree = args.pp_degree

        # setup topology
        if self.is_distributed:
            fleet.init(is_collective=True)
            if self.use_sharding:
                self.topo = Topology(
                    device_rank=fleet.worker_index(),
                    world_size=fleet.worker_num(),
                    dp_degree=self.dp_degree,
                    pp_degree=self.pp_degree,
                    sharding_degree=self.sharding_degree,
                    mp_degree=self.mp_degree)
            else:
                self.topo = Topology(
                    device_rank=fleet.worker_index(),
                    world_size=fleet.worker_num(),
                    dp_degree=fleet.worker_num())
        else:
            self.topo = Topology(device_rank=0, world_size=1)
            if self.use_recompute:
                print("[WARN] Cannot support recomputation in non-distributed mode.")
            if self.use_amp:
                print("[WARN] Cannot support AMP in non-distributed mode.")

        self.exe = fluid.Executor(place)
        # model mode
        self.run_infer = args.get("run_infer", False)
        self.batch_size = args.get("batch_size", 1)

        self._lr_scheduler = self._get_lr_scheduler(args)
        self._optimizer = self._get_optimizer(args, self._lr_scheduler)

        self._build_programs()
        return

    def _init_distributed_strategy(self):
        """Initialize distributed strategy."""
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        exec_strategy.num_iteration_per_drop_scope = 1

        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.execution_strategy = exec_strategy
        dist_strategy.nccl_comm_num = 1
        dist_strategy.fuse_all_reduce_ops = True
        if self.use_recompute:
            dist_strategy.recompute = True
        if self.use_amp:
            dist_strategy.amp = True
            dist_strategy.amp_configs = {
                "custom_white_list": ["softmax", "layer_norm", "gelu"],
                "init_loss_scaling": self.amp_loss_scaling
            }
        if self.use_sharding:
            dist_strategy.sharding = True
            dist_strategy.sharding_configs = {
                "segment_broadcast_MB": 32,
                "dp_degree": self.dp_degree,
                "sharding_degree": self.sharding_degree,
                "mp_degree": self.mp_degree,
                "pp_degree": self.pp_degree
            }
        self.dist_strategy = dist_strategy
        self._init_build_strategy()
        print(self.dist_strategy)
        return

    def _init_build_strategy(self):
        pass

    def _set_checkpoints(self, checkpoints):
        """Set checkpoints for recompute.

        Args:
            checkpoints: A list of Variables which need to set as checkpoints.
        """
        self.dist_strategy.recompute_configs = {
            "checkpoints": [x.name for x in checkpoints[
                self.checkpointing_every_n_layers - 1::self.checkpointing_every_n_layers]]
        }
        return

    def _build_programs(self):
        """Build programs.

        Build training program, evaluation program and inference program. Only use in static graph mode.
        """
        self.startup_program = fluid.Program()

        if self.run_infer:
            if self.is_distributed and self.use_sharding:
                init_parallel_env()
            # build inference program
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
                    self.optimize(metrics)

                    # build evaluation program
                    self.eval_program = self.train_program.clone(for_test=True)
                    self.eval_fetch_dict = dict(**metrics)

                    global_vars = fluid.default_main_program().global_block().vars
                    metrics["scheduled_lr"] = global_vars["learning_rate_0"]
                    if self.is_distributed and self.use_amp:
                        loss_scaling = global_vars["loss_scaling_0"]
                        metrics["loss_scaling"] = loss_scaling
                    self.train_fetch_dict = metrics

            self.program = self.train_program

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
            if os.path.exists(os.path.join(model_path, var.name)):
                return True
            else:
                print(f"Warning: {var.name} does not exist.")
                return False
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
            if isinstance(self._lr_scheduler, lr.LRScheduler):
                lr_scheduler_dict_path = os.path.join(model_path, "__lr_scheduler__")
                if os.path.isfile(lr_scheduler_dict_path):
                    lr_scheduler_dict = paddle.load(lr_scheduler_dict_path)
                    self.args.start_step = lr_scheduler_dict["last_epoch"]
                    self._lr_scheduler.set_state_dict(lr_scheduler_dict)
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
            if isinstance(self._lr_scheduler, lr.LRScheduler):
                lr_scheduler_dict_path = os.path.join(model_path, "__lr_scheduler__")
                paddle.save(self._lr_scheduler.state_dict(), lr_scheduler_dict_path)
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

    def _get_lr_scheduler(self, args):
        assert args.warmup_steps >= 0

        if args.lr_scheduler == "noam" and args.warmup_steps == 0:
            print("[WARN] Using constant learning rate because of `warmup_steps` is not positive while using NoamScheduler.")
            args.lr_scheduler = "constant"

        if args.lr_scheduler == "noam" and args.warmup_steps > 0:
            scheduler = lr.NoamDecay(
                1 / (args.warmup_steps * (args.learning_rate ** 2)),
                args.warmup_steps)
        elif args.lr_scheduler == "linear":
            assert args.max_decay_steps >= args.warmup_steps
            scheduler = lr.PolynomialDecay(
                args.learning_rate,
                decay_steps=args.max_decay_steps - args.warmup_steps,
                end_lr=args.min_learning_rate,
                power=1.0)
        elif args.lr_scheduler == "cosine":
            assert args.max_decay_steps >= args.warmup_steps
            scheduler = CosineDecay(
                args.learning_rate,
                decay_steps=args.max_decay_steps - args.warmup_steps,
                end_lr=args.min_learning_rate)
        else: # constant
            scheduler = args.learning_rate

        # linear warmup
        if args.lr_scheduler != "noam" and args.warmup_steps > 0:
            scheduler = lr.LinearWarmup(
                scheduler,
                args.warmup_steps,
                start_lr=0,
                end_lr=args.learning_rate)
        return scheduler

    def _get_optimizer(self, args, lr_scheduler):
        """Get the optimizer of model.

        Args:
            args: arguments.
            lr_scheduler: learning rate scheduler

        Returns:
            optimizer: the optimizer used in model training.
        """
        # grad norm
        if args.max_grad_norm > 0:
            grad_clip = fluid.clip.GradientClipByGlobalNorm(args.max_grad_norm)
        else:
            grad_clip = None

        # optimizer
        optimizer_cls = getattr(knover.optim, args.optimizer)
        optimizer = optimizer_cls(
            learning_rate=lr_scheduler,
            grad_clip=grad_clip,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2)
        return optimizer


    def optimize(self, metrics):
        """Optimize the model by loss.

        Args:
            metrics: A dict mapping metric names to corresponding metrics, which must include loss.
        """
        optimizer = self._optimizer
        # distributed optimizer
        if self.is_distributed:
            optimizer = fleet.distributed_optimizer(optimizer, strategy=self.dist_strategy)

        optimizer.minimize(metrics["loss"])

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
        if isinstance(self._lr_scheduler, lr.LRScheduler):
            self._lr_scheduler.step()
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
