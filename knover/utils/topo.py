#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Define model topology."""

from collections import namedtuple

import numpy as np


GroupInfo = namedtuple("GroupInfo", ["size", "rank", "world"])


class Topology(object):

    def __init__(self,
                 device_rank,
                 world_size,
                 dp_degree=1,
                 pp_degree=1,
                 sharding_degree=1,
                 mp_degree=1):
        assert dp_degree * pp_degree * sharding_degree * mp_degree  == world_size
        arr = np.arange(0, world_size).reshape([dp_degree, pp_degree, sharding_degree, mp_degree])
        dp_idx, pp_idx, sharding_idx, mp_idx = np.where(arr == device_rank)
        dp_idx, pp_idx, sharding_idx, mp_idx = dp_idx[0], pp_idx[0], sharding_idx[0], mp_idx[0]

        self.world = GroupInfo(size=world_size, rank=device_rank, world=list(range(0, world_size)))

        # parallelism groups
        mp_world = arr[dp_idx, pp_idx, sharding_idx, :].tolist()
        self.mp_info = GroupInfo(size=len(mp_world), rank=mp_idx, world=mp_world)
        sharding_world = arr[dp_idx, pp_idx, :, mp_idx].tolist()
        self.sharding_info = GroupInfo(size=len(sharding_world), rank=sharding_idx, world=sharding_world)
        pp_world = arr[dp_idx, :, sharding_idx, mp_idx].tolist()
        self.pp_info = GroupInfo(size=len(pp_world), rank=pp_idx, world=pp_world)
        dp_world = arr[:, pp_idx, sharding_idx, mp_idx].tolist()
        self.dp_info = GroupInfo(size=len(dp_world), rank=dp_idx, world=dp_world)

        # the last rank of a pipeline group
        self.is_last = self.pp_info.rank == pp_degree - 1

        # dataset partition
        data_arr = np.arange(0, dp_degree * sharding_degree).reshape([dp_degree, sharding_degree])
        data_arr = np.expand_dims(data_arr, axis=1).repeat(pp_degree, axis=1)
        data_arr = np.expand_dims(data_arr, axis=3).repeat(mp_degree, axis=3)
        data_world = data_arr.reshape(-1).tolist()
        self.data_info = GroupInfo(
            size=dp_degree * sharding_degree,
            rank=self.dp_info.rank * sharding_degree + self.sharding_info.rank,
            world=data_world)

        self.data_inner_times = world_size // self.data_info.size
        self.num_model_partitions = world_size // dp_degree

    def __repr__(self):
        return f"dp: {self.dp}, pp: {self.pp}, sharding: {self.sharding}, mp: {self.mp}"
