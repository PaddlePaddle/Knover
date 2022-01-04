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
"""Traine main program."""

import argparse
from collections import defaultdict
import json
import os
import random
import subprocess
import time

import numpy as np
import paddle
import paddle.distributed as distributed
import paddle.fluid as fluid

import knover.models as models
import knover.tasks as tasks
from knover.utils import check_cuda, parse_args, str2bool, Timer


def setup_args():
    """Setup training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", type=str2bool, default=False,
                        help="Whether to run distributed training.")
    parser.add_argument("--save_path", type=str, default="output",
                        help="The path where to save models.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="The training dataset: file / filelist. "
                        "See more details in `docs/usage.md`: `file_format`.")
    parser.add_argument("--valid_file", type=str, required=True,
                        help="The validation datasets: files / filelists. "
                        "The files / filelists are separated by `,`. "
                        "See more details in `docs/usage.md`: `file_format`.")

    parser.add_argument("--start_step", type=int, default=0,
                        help="The start step of training. It will be updated if you load from a checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="The number of times that the learning algorithm will work through the entire training dataset.")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Display training / evaluation log information every X steps.")
    parser.add_argument("--validation_steps", type=int, default=1000,
                        help="Run validation every X training steps.")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save the latest model every X training steps. "
                        "If `save_steps = 0`, then it only keep the latest checkpoint.")
    parser.add_argument("--eval_metric", type=str, default="-loss",
                        help="Keep the checkpoint with best evaluation metric.")
    parser.add_argument("--save_checkpoint", type=str2bool, default=True,
                        help="Save completed checkpoint or parameters only. "
                        "The checkpoint contains all states for continuous training.")

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.display()
    return args


def run_cmd(cmd):
    """Helpful function for running shell command in py scripts."""
    exitcode, output = subprocess.getstatusoutput(cmd)
    if exitcode != 0:
        raise ValueError("Raise error while running shell command.")
    return output


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def train(args):
    """The main function of training."""
    set_seeds(args.random_seed)
    gpu_id = int(os.getenv("FLAGS_selected_gpus")) if args.is_distributed else 0
    place = fluid.CUDAPlace(gpu_id)

    # setup task and model
    task = tasks.create_task(args)
    model = models.create_model(args, place)

    global use_vdl
    use_vdl = args.use_k8s and model.is_last_rank()
    # setup datasets
    train_generator = task.get_data_loader(
        model,
        input_file=args.train_file,
        num_epochs=args.num_epochs,
        num_part=model.get_data_world_size(),
        part_id=model.get_data_rank(),
        phase="train"
    )
    valid_tags = []
    valid_generators = []
    for valid_file in args.valid_file.split(","):
        if ":" in valid_file:
            valid_tag, valid_file = valid_file.split(":")
        else:
            valid_tag = "valid"
        valid_tags.append(valid_tag)
        valid_generators.append(task.get_data_loader(
            model,
            input_file=valid_file,
            num_part=model.get_data_world_size(),
            part_id=model.get_data_rank(),
            phase="distributed_valid" if args.is_distributed else "valid"
        ))

    # maintain best metric (init)
    best_metric = -1e10
    if args.eval_metric.startswith("-"):
        scale = -1.0
        eval_metric = args.eval_metric[1:]
    else:
        scale = 1.0
        eval_metric = args.eval_metric

    # start training
    timer = Timer()
    timer.start()
    print("Training is start.")
    for step, data in enumerate(train_generator(), args.start_step + 1):
        outputs = task.train_step(model, data)
        timer.pause()

        if step % args.log_steps == 0:
            time_cost = timer.pass_time
            current_epoch, current_file_index, total_file = task.reader.get_train_progress()
            current_lr = outputs.pop('scheduled_lr')
            print(f"[train][{current_epoch}] progress: {current_file_index}/{total_file} "
                  f"step: {step}, time: {time_cost:.3f}, "
                  f"queue size: {train_generator.queue.size()}, "
                  f"speed: {args.log_steps / time_cost:.3f} steps/s")
            print(f"\tcurrent lr: {current_lr:.7f}")
            metrics = task.get_metrics(outputs)
            print("\t" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            timer.reset()

        if step % args.validation_steps == 0:
            for valid_tag, valid_generator in zip(valid_tags, valid_generators):
                eval_metrics = evaluate(task, model, valid_generator, args, step, tag=valid_tag)
                if valid_tag == "valid":
                    valid_metrics = eval_metrics

            # save latest model
            if args.save_steps <= 0:
                save_model(model, args.save_path, "latest", args)
            # maintain best metric (update)
            if valid_metrics[eval_metric] * scale > best_metric:
                best_metric = valid_metrics[eval_metric] * scale
                print(f"Get better valid metric: {eval_metric} = {valid_metrics[eval_metric]}")
                # save best model (with best evaluation metric)
                save_model(model, args.save_path, "best", args)

        if args.save_steps > 0 and step % args.save_steps == 0:
            save_model(model, args.save_path, f"step_{step}", args)

        timer.start()
    print("Training is completed.")

    return


def evaluate(task,
             model,
             generator,
             args,
             training_step,
             tag=None):
    """Run evaluation.

    Run evaluation on dataset which is generated from a generator. Support evaluation on single GPU and multiple GPUs.

    Single GPU:
    1. Run evaluation on the whole dataset (the generator generates the completed whole dataset).
    2. Disply evaluation result.

    Multiple GPUs:
    1. Each GPU run evaluation on a part of dataset (the generator only generate a part of dataset).
    2. Merge all evaluation results in distributed mode.
    3. Display evaluation result.
    """
    outputs = None
    print("=" * 80)
    print(f"Evaluation: {tag}")
    timer = Timer()
    timer.start()
    for step, data in enumerate(generator(), 1):
        part_outputs = task.eval_step(model, data)
        outputs = task.merge_metrics_and_statistics(outputs, part_outputs)

        if step % args.log_steps == 0:
            metrics = task.get_metrics(outputs)
            print(f"\tstep {step}:" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    if model._is_distributed:
        assert outputs is not None, "Validation set must have at least a batch of data in each GPU."
        # merge in distributed mode.
        outputs = task.merge_distributed_metrics_and_statistics(outputs)

    metrics = task.get_metrics(outputs)
    print(f"[Evaluation][{training_step}] " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    print(f"\ttime cost: {timer.pass_time:.3f}")
    print("=" * 80)
    return metrics


def save_model(model, save_path, tag, args):
    """Save model.

    In normal mode, only the master GPU need to save the model.
    In sharding mode, it need to save each part of model in GPUs.
    """
    if model.get_data_parallel_rank() != 0:
        return
    path = os.path.join(save_path, tag)
    if args.use_sharding:
        # save part of model in sharding mode
        print(f"Saving part of model into {path}.")
        model.save(path + f".part_{model.get_global_rank()}", is_checkpoint=args.save_checkpoint)
        print(f"Part of model has saved into {path}.")
        model.sync()
    else:
        print(f"Saving model into {path}.")
        model.save(path, is_checkpoint=args.save_checkpoint)
        print(f"Model has saved into {path}.")
    return


if __name__ == "__main__":
    args = setup_args()
    check_cuda(True)
    train(args)
