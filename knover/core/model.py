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

import paddle
from paddle.distributed import fleet
import paddle.nn as nn
from paddle.optimizer.lr import NoamDecay, LinearWarmup, PolynomialDecay

import knover.optim
from knover.optim.lr_scheduler import CosineDecay
from knover.utils import str2bool


class ModelMeta(ABCMeta, type(nn.Layer)):
    """Handle multiply inherit from ABC and paddle.nn.Layer."""
    pass


class Model(nn.Layer, metaclass=ModelMeta):
    """Basic model wrapper in dygraph mode.

    Attributes:
        place: CPUPlace or CUDAPlace.
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
        group.add_argument("--use_amp", type=str2bool, default=False,
                           help="Whether to use automatic mixed precision(AMP) training")

        return group

    def __init__(self, args, place):
        super(Model, self).__init__()
        self._build_model(args)

        self.place = place

        # distributed settings
        self.use_amp = args.use_amp

        # optimizer related
        self.lr_scheduler = self._get_lr_scheduler(args)
        self.optimizer = self._get_optimizer(args, self.lr_scheduler)

        # initialize paramters
        if args.init_checkpoint != "":
            self.load(args.init_checkpoint, is_checkpoint=True)
        elif args.init_pretraining_params != "":
            self.load(args.init_pretraining_params)

        return

    def _build_model(self, args):
        """Build model."""
        raise NotImplementedError

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
        self.set_state_dict(params_state_dict)
        if is_checkpoint:
            opt_path = model_path + "pdopt"
            assert os.path.exists(opt_path), f"opt_path: [{opt_path}] cannot be found."
            print(f"Loading optimizer state from {opt_path}.")
            opt_state_dict = paddle.load(model_path + "pdopt")
            self.optimizer.set_state_dict(opt_state_dict)
        print("Loading has done!")
        return

    def save(self, model_path, is_checkpoint=False):
        """Save persistables or parameters into the given path.

        Args:
            model_path: The path where we save the model.
            is_checkpoint: If true, save parameters and other variables (such as moments in Adam optimizer), otherwise save only parameters.
        """
        params_path = model_path + ".pdparams"
        print(f"Saving parameters into {params_path}.")
        paddle.save(self.state_dict(), params_path)
        if is_checkpoint:
            opt_path = model_path + ".pdopt"
            print(f"Saving optimizer state into {opt_path}.")
            paddle.save(self.optimizer.state_dict(), opt_path)
        print("Saving has done!")
        return

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
                        self.infer_feed_names = list(batch.keys())
                    else:
                        self.feed_names = list(batch.keys())
                    yield list(batch.values())
            loader.set_batch_generator(__wrapper__, places=self.place)
        return loader

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
        metrics["loss"].backward()
        if isinstance(self.lr_scheduler, paddle.optimizer.lr.LRScheduler):
            self.lr_scheduler.step()
        metrics["scheduled_lr"] = self.optimizer.get_lr()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return

    def _get_lr_scheduler(self, args):
        if args.lr_scheduler == "noam" and args.warmup_steps <= 0:
            print("[WARMING] Using constant learning rate because of `warmup_steps` is not positive while using NoamScheduler.")
        if args.lr_scheduler == "noam" and args.warmup_steps > 0:
            scheduler = NoamDecay(
                1 / (args.warmup_steps * (args.learning_rate ** 2)),
                args.warmup_steps)
        elif args.lr_scheduler == "linear":
            scheduler = PolynomialDecay(
                args.learning_rate,
                decay_steps=args.max_training_steps - args.warmup_steps,
                end_lr=args.min_learning_rate,
                power=1.0)
            scheduler = LinearWarmup(
                scheduler,
                args.warmup_steps,
                start_lr=0,
                end_lr=args.learning_rate)
        elif args.lr_scheduler == "cosine":
            scheduler = CosineDecay(
                args.learning_rate,
                decay_steps=args.max_training_steps - args.warmup_steps,
                end_lr=args.min_learning_rate)
            scheduler = LinearWarmup(
                scheduler,
                args.warmup_steps,
                start_lr=0,
                end_lr=args.learning_rate)
        else: # constant
            scheduler = args.learning_rate
        return scheduler

    def _get_optimizer(self, args, lr_scheduler):
        """Get the optimizer of model.

        Args:
            args: arguments.
            lr_scheduler: learning rate scheduler

        Returns:
            optimizer: the optimizer used in model training.
        """
        # optimizer
        if not hasattr(knover.optim, args.optimizer):
            raise ValueError(f"Unspported optimizer class: {args.optimizer}.")
        optimizer_cls = getattr(knover.optim, args.optimizer)
        optimizer = optimizer_cls(
            lr_scheduler,
            parameters=self.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))
        return optimizer

    def _get_inputs(self, inputs, is_infer=False):
        """Wrapper of input data format in dygraph.

        Convert a list of Tensors into a dictionary which maps names into Tensors.

        Args:
            inputs: A list of input Tensors.
            is_infer: If true, get inference's feed dict, otherwise get training / evaluation 's feed dict.

        Returns:
            inputs: A dict mapping keys to corresponding Tensors.
        """
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
        self.train()
        with paddle.amp.auto_cast(self.use_amp):
            inputs = self._get_inputs(inputs)
            outputs = self.forward(inputs)
            metrics = self.get_metrics(inputs, outputs)
            return self._get_outputs(metrics)

    @paddle.no_grad()
    def eval_step(self, inputs):
        """Run one evaluation step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        self.eval()
        inputs = self._get_inputs(inputs)
        outputs = self.forward(inputs)
        metrics = self.get_metrics(inputs, outputs)
        statistics = self.get_statistics(inputs, outputs)
        metrics.update(statistics)
        return self._get_outputs(metrics)

    @paddle.no_grad()
    def infer_step(self, inputs):
        """Run one inference step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            predictions: A dict mapping keys to corresponding predictions.
        """
        self.eval()
        inputs = self._get_inputs(inputs, is_infer=True)
        outputs = self.forward(inputs, is_infer=True)
        predictions = self.infer(inputs, outputs)
        return self._get_outputs(predictions)

    def __call__(self, inputs, mode="train"):
        if mode == "train":
            return self.train_step(inputs)
        elif mode == "eval":
            return self.eval_step(inputs)
        elif mode == "infer":
            return self.infer_step(inputs)
        else:
            raise ValueError(f"Unspported mode: {mode}.")

    def save_inference_model(self, inference_model_path):
        """Save the inference model.

        Only save the inference program.

        Args:
            inference_model_path: The path of saved inference model.
        """
        raise NotImplementedError("Cannot support save_inference_model now.")


class ModelInterface(object):
    """The model inference."""

    def __init__(self, args, model):
        assert isinstance(model, Model), "The model must be an instance of Model"
        self.model = model
        self.is_distributed = args.is_distributed
        if args.is_distributed:
            # distributed settings for dygraph.
            # now only support data parallel for dygraph.
            self.model.optimizer = fleet.distributed_optimizer(self.model.optimizer)
            self.dp_model = fleet.distributed_model(self.model)

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

    def train_step(self, inputs):
        """Run one training step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics.
        """
        if self.is_distributed:
            metrics = self.dp_model(inputs, mode="train")
        else:
            metrics = self.model(inputs, mode="train")
        self.model.optimize(metrics)
        return self._get_outputs(metrics)

    def eval_step(self, inputs):
        """Run one evaluation step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            metrics: A dict mapping keys to corresponding metrics (numpy arrays).
        """
        if self.is_distributed:
            metrics = self.dp_model(inputs, mode="eval")
        else:
            metrics = self.model(inputs, mode="eval")
        return self._get_outputs(metrics)

    def infer_step(self, inputs):
        """Run one inference step.

        Args:
            inputs: A dict mapping keys to corresponding input data.

        Returns:
            predictions: A dict mapping keys to corresponding predictions (numpy arrays).
        """
        if self.is_distributed:
            predictions = self.dp_model(inputs, mode="infer")
        else:
            predictions = self.model(inputs, mode="infer")
        return self._get_outputs(predictions)

    def get_data_loader(self, generator=None, is_infer=False):
        """Get the DataLoader of the model.

        Args:
            generator: If generator is not `None`, the DataLoader sets it as the batch generator.
            is_infer: If true, get inference's DataLoader, otherwise get training / evaluation 's DataLoader.

        Returns:
            loader: A DataLoader which is used to generate batch data.
        """
        return self.model.get_data_loader(generator, is_infer)

    def save(self, model_path, is_checkpoint=False):
        """Save persistables or parameters into the given path.

        Args:
            model_path: The path where we save the model.
            is_checkpoint: If true, save parameters and other variables (such as moments in Adam optimizer), otherwise save only parameters.
        """
        self.model.save(model_path, is_checkpoint)

    def load(self, model_path, is_checkpoint=False):
        """Load persistables or parameters from the given path.

        Args:
            model_path: The path of initial model.
            is_checkpoint: If true, load parameters and other variables (such as moments in Adam optimizer), otherwise load only parameters.
        """
        self.model.load(model_path, is_checkpoint)

    def save_inference_model(self, inference_model_path):
        """Save the inference model.

        Only save the inference program.

        Args:
            inference_model_path: The path of saved inference model.
        """
        raise NotImplementedError("Cannot support save_inference_model now.")
