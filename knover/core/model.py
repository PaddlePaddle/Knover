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

from abc import abstractmethod, ABC, ABCMeta
import os

import numpy as np
import paddle
from paddle.distributed import fleet
import paddle.nn as nn
import paddle.optimizer.lr as lr
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import DygraphShardingOptimizer
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

import knover.optim
from knover.optim.lr_scheduler import CosineDecay
from knover.utils import str2bool


class ModelMeta(ABCMeta, type(nn.Layer)):
    """Handle multiply inherit from ABC and paddle.nn.Layer."""
    pass


class Model(nn.Layer, metaclass=ModelMeta):
    """Basic model wrapper in dygraph mode."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        return ModelInterface.add_cmdline_args(parser)

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self._build_model(args)
        return

    def _build_model(self, args):
        """Build model."""
        raise NotImplementedError

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

    def _get_inputs(self, inputs, is_infer=False):
        """Wrapper of input data format in dygraph.

        Convert a list of Tensors into a dictionary which maps names into Tensors.

        Args:
            inputs: A list of input Tensors.
            is_infer: If true, get inference's feed dict, otherwise get training / evaluation 's feed dict.

        Returns:
            inputs: A dict mapping keys to corresponding Tensors.
        """
        inputs = [v if paddle.is_tensor(v) else paddle.Tensor(v) for v in inputs]
        if is_infer:
            return dict(zip(self.infer_feed_names, inputs))
        else:
            return dict(zip(self.feed_names, inputs))

    def _get_outputs(self, outputs):
        """Wrapper of output data format in dygraph."""
        return outputs

    def train_step(self, inputs):
        """Run one training step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        outputs = self.forward(inputs)
        metrics = self.get_metrics(inputs, outputs)
        return metrics

    def eval_step(self, inputs):
        """Run one evaluation step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        outputs = self.forward(inputs)
        metrics = self.get_metrics(inputs, outputs)
        statistics = self.get_statistics(inputs, outputs)
        metrics.update(statistics)
        return metrics

    def infer_step(self, inputs):
        """Run one inference step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            predictions: A dict mapping keys to corresponding predictions.
        """
        outputs = self.forward(inputs, is_infer=True)
        predictions = self.infer(inputs, outputs)
        return predictions

    def __call__(self, *inputs, mode="train"):
        inputs = self._get_inputs(inputs, is_infer=mode == "infer")
        if mode == "train":
            outputs = self.train_step(inputs)
        elif mode == "eval":
            outputs = self.eval_step(inputs)
        elif mode == "infer":
            outputs = self.infer_step(inputs)
        else:
            raise ValueError(f"Unspported mode: {mode}.")
        return self._get_outputs(outputs)


class ModelInterface(object):
    """The model inference."""

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
        group.add_argument("--max_decay_steps", type=int, default=2000,
                           help="The maximum decay step used in lr_scheduler.")
        group.add_argument("--min_learning_rate", type=float, default=0,
                           help="The minimum learning rate used in lr_scheduler.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for optimizer.")
        group.add_argument("--max_grad_norm", type=float, default=.1,
                           help="The maximum norm of gradient.")

        # fleet related
        # amp related
        group.add_argument("--use_amp", type=str2bool, default=False,
                           help="Whether to use automatic mixed precision(AMP) to speedup training and save memory.")
        group.add_argument("--amp_level", type=str, default="O1",
                           choices=["O1", "O2"],
                           help="The level of amp training")
        group.add_argument("--amp_loss_scaling", type=float, default=32768.,
                           help="The initial loss scaling of AMP.")
        # recompute related
        group.add_argument("--use_recompute", type=str2bool, default=False,
                           help="Whether to use recompute to save memory usage.")
        group.add_argument("--use_sharding", type=str2bool, default=False,
                           help="Whether to use sharding strategy.")
        group.add_argument("--dp_degree", type=int, default=1, help="Data parallism degree.")
        group.add_argument("--sharding_degree", type=int, default=1, help="Sharding parallism degree.")
        group.add_argument("--mp_degree", type=int, default=1, help="Tensor model parallism degree.")
        group.add_argument("--pp_degree", type=int, default=1, help="Pipeline model parallism degree.")
        return group

    def __init__(self, args, model_cls, place):
        self._init_distributed_envs(args)

        model = model_cls(args)

        assert isinstance(model, Model), "The model must be an instance of Model"
        self._model = model
        self._is_distributed = args.is_distributed and fleet.worker_num() > 1
        self._use_recompute = args.use_recompute

        self._place = place

        # optimizer related
        self._lr_scheduler = self._get_lr_scheduler(args)
        self._optimizer = self._get_optimizer(args, self._lr_scheduler)

        if args.is_distributed:
            # distributed settings for dygraph.
            # now only support data parallel for dygraph.
            if self._use_amp:
                self._scaler = fleet.distributed_scaler(self._scaler)
            self._dist_model = fleet.distributed_model(self._model)
            self._optimizer = fleet.distributed_optimizer(self._optimizer)

        # initialize paramters
        if args.init_checkpoint != "":
            self.load(args.init_checkpoint, is_checkpoint=True)
        elif args.init_pretraining_params != "":
            self.load(args.init_pretraining_params)

    def _init_distributed_envs(self, args):
        dist_strategy = fleet.DistributedStrategy()
        if args.use_sharding:
            dist_strategy.hybrid_configs = {
                "dp_degree": args.dp_degree,
                "sharding_degree": args.sharding_degree,
                "mp_degree": args.mp_degree,
                "pp_degree": args.pp_degree
            }
            dist_strategy.tensor_parallel_configs = {
                "tensor_init_seed": args.random_seed
            }
        self._dist_strategy = dist_strategy
        fleet.init(is_collective=True, strategy=self._dist_strategy)

        self._hcg = fleet.get_hybrid_communicate_group()

    def get_global_rank(self):
        return self._hcg.get_global_rank()

    def is_last_rank(self):
        return self._hcg.global_rank == self._hcg.nranks - 1

    def get_data_parallel_world_size(self):
        return self._hcg.get_data_parallel_world_size()

    def get_data_parallel_rank(self):
        return self._hcg.get_data_parallel_rank()

    def get_model_world_size(self):
        return self._hcg.nranks // self.get_data_parallel_world_size()

    def get_data_world_size(self):
        return self._hcg.get_data_parallel_world_size() * self._hcg.get_sharding_parallel_world_size()

    def get_data_rank(self):
        self._hcg.get_data_parallel_group
        return self._hcg.get_data_parallel_rank() * self._hcg.get_sharding_parallel_world_size() \
            + self._hcg.get_sharding_parallel_rank()

    def sync(self):
        with paddle.no_grad():
            tensor = paddle.to_tensor(np.array([1]).astype(np.int))
            paddle.distributed.all_reduce(tensor, group=self._hcg.get_check_parallel_group())
        return

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
        # amp settings
        self._use_amp = args.use_amp
        self._amp_level = args.amp_level

        # optimizer
        if not hasattr(knover.optim, args.optimizer):
            raise ValueError(f"Unspported optimizer class: {args.optimizer}.")
        optimizer_cls = getattr(knover.optim, args.optimizer)
        if args.max_grad_norm > 0:
            grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
        else:
            grad_clip = None

        self._use_sharding = args.use_sharding
        if args.use_sharding and args.sharding_degree > 1:
            optimizer = DygraphShardingOptimizer(
                hcg=self._hcg,
                user_defined_strategy=self._dist_strategy,
                params=self._model.parameters(),
                inner_optimizer_class=optimizer_cls,
                learning_rate=lr_scheduler,
                weight_decay=args.weight_decay,
                grad_clip=grad_clip)
        else:
            optimizer = optimizer_cls(
                lr_scheduler,
                parameters=self._model.parameters(),
                weight_decay=args.weight_decay,
                grad_clip=grad_clip)

        if self._use_amp:
            self._scaler = paddle.amp.GradScaler(init_loss_scaling=args.amp_loss_scaling)
            self._model, optimizer = paddle.amp.decorate(
                models=self._model,
                optimizers=optimizer,
                level=self._amp_level)

        return optimizer

    def _get_outputs(self, outputs):
        """Convert Tensors into numpy arrays.

        Args:
            outputs: A dict mapping keys to output Tensors.

        Returns:
            outputs: A dict mapping keys to numpy arrays.
        """
        if isinstance(outputs, dict):
            return {k: v.numpy() if isinstance(v, paddle.Tensor) else v
                    for k, v in outputs.items()}
        return outputs

    def backward(self, loss):
        # backward
        if self._use_amp:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimize(self, metrics):
        """Optimize the model by loss.

        Args:
            metrics: A dict mapping metric names to corresponding metrics, which must include loss.
        """
        if isinstance(self._lr_scheduler, paddle.optimizer.lr.LRScheduler):
            self._lr_scheduler.step()
            metrics["scheduled_lr"] = self._optimizer.get_lr()
        else:
            metrics["scheduled_lr"] = self._lr_scheduler
        if self._use_amp:
            self._scaler.minimize(self._optimizer, metrics["loss"])
            metrics["loss_scaling"] = self._scaler._scale
        else:
            self._optimizer.step()
        self._optimizer.clear_grad()
        return

    def train_step(self, inputs):
        """Run one training step.

        Args:
            inputs: A list of input data. All elements are Tensor.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        with paddle.amp.auto_cast(
                self._use_amp,
                custom_white_list=["softmax", "layer_norm", "gelu"],
                custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div"],
                level=self._amp_level):
            if self._is_distributed:
                self._dist_model.train()
                if not self._use_sharding and self._use_recompute:
                    with self._dist_model.no_sync():
                        metrics = self._dist_model(*inputs, mode="train")
                        self.backward(metrics["loss"])
                else:
                    metrics = self._dist_model(*inputs, mode="train")
                    self.backward(metrics["loss"])
            else:
                self._model.train()
                metrics = self._model(*inputs, mode="train")
                self.backward(metrics["loss"])
        if self._is_distributed and self._use_recompute:
            fused_allreduce_gradients(list(self._dist_model.parameters()), None)
        self.optimize(metrics)
        return self._get_outputs(metrics)

    def eval_step(self, inputs):
        """Run one evaluation step.

        Args:
            inputs: A list of input data. All elements are Tensor.

        Returns:
            metrics: A dict mapping keys to corresponding metrics (numpy arrays).
        """
        with paddle.no_grad():
            with paddle.amp.auto_cast(
                self._use_amp,
                custom_white_list=["softmax", "layer_norm", "gelu"],
                custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy", "elementwise_div"],
                level=self._amp_level):
                if self._is_distributed:
                    self._dist_model.eval()
                    metrics = self._dist_model(*inputs, mode="eval")
                else:
                    self._model.eval()
                    metrics = self._model(*inputs, mode="eval")
        return self._get_outputs(metrics)

    def infer_step(self, inputs):
        """Run one inference step.

        Args:
            inputs: A list of input data. All elements are Tensor.

        Returns:
            predictions: A dict mapping keys to corresponding predictions (numpy arrays).
        """
        with paddle.no_grad():
            if isinstance(inputs, dict):
                self._model.infer_feed_names = list(inputs.keys())
                inputs = list(inputs.values())
            if self._is_distributed:
                self._dist_model.eval()
                predictions = self._dist_model(*inputs, mode="infer")
            else:
                self._model.eval()
                predictions = self._model(*inputs, mode="infer")
        return self._get_outputs(predictions)

    def get_data_loader(self, generator=None, is_infer=False):
        """Get the DataLoader of the model.

        Args:
            generator: If generator is not `None`, the DataLoader sets it as the batch generator.
            is_infer: If true, get inference's DataLoader, otherwise get training / evaluation 's DataLoader.

        Returns:
            loader: A DataLoader which is used to generate batch data.
        """
        loader = paddle.io.DataLoader.from_generator(capacity=64, return_list=True)
        if generator is not None:
            def __wrapper__():
                for batch in generator():
                    if is_infer:
                        self._model.infer_feed_names = list(batch.keys())
                    else:
                        self._model.feed_names = list(batch.keys())
                    yield list(batch.values())
            loader.set_batch_generator(__wrapper__, places=self._place)
        return loader

    def save(self, model_path, is_checkpoint=False):
        """Save persistables or parameters into the given path.

        Args:
            model_path: The path where we save the model.
            is_checkpoint: If true, save parameters and other variables (such as moments in Adam optimizer), otherwise save only parameters.
        """
        params_path = model_path + ".pdparams"
        print(f"Saving parameters into {params_path}.")
        paddle.save(self._model.state_dict(), params_path)
        if is_checkpoint:
            opt_path = model_path + ".pdopt"
            print(f"Saving optimizer state into {opt_path}.")
            paddle.save(self._optimizer.state_dict(), opt_path)
        print("Saving has done!")
        return

    def load(self, model_path, is_checkpoint=False):
        """Load persistables or parameters from the given path.

        Args:
            model_path: The path of initial model.
            is_checkpoint: If true, load parameters and other variables (such as moments in Adam optimizer), otherwise load only parameters.
        """
        params_path = model_path + ".pdparams"
        assert os.path.exists(params_path), f"params_path: [{params_path}] cannot be found."
        print(f"Loading parameters from {params_path}.")
        params_state_dict = paddle.load(params_path)
        self._model.set_state_dict(params_state_dict)
        if is_checkpoint:
            opt_path = model_path + ".pdopt"
            assert os.path.exists(opt_path), f"opt_path: [{opt_path}] cannot be found."
            print(f"Loading optimizer state from {opt_path}.")
            opt_state_dict = paddle.load(opt_path)
            self._optimizer.set_state_dict(opt_state_dict)
            if "LR_Scheduler" in opt_state_dict:
                self._model.args.start_step = opt_state_dict["LR_Scheduler"]["last_epoch"]
            else:
                print("[WARN] Cannot determinate current start_step from the checkpoint.")
        print("Loading has done!")
        return

    def save_inference_model(self, inference_model_path):
        """Save the inference model.

        Only save the inference program.

        Args:
            inference_model_path: The path of saved inference model.
        """
        raise NotImplementedError("Cannot support save_inference_model now.")
