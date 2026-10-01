"""NCCL process groups and distributed objective execution under torchrun."""
import os
from dataclasses import dataclass

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass
class ProcessGroup:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    owns_group: bool = False

    @classmethod
    def create(cls, device):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        local = int(os.environ.get('LOCAL_RANK', '0'))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Distributed rank and world size are invalid.')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('Distributed RoPA training requires CUDA and NCCL.')
        if world_size > 1:
            if device.index is not None and device.index != local:
                raise ValueError('Under torchrun use --device cuda or the assigned LOCAL_RANK.')
            device = torch.device('cuda', local)
        elif device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
        owns_group = world_size > 1 and not dist.is_initialized()
        if owns_group:
            dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
        if dist.is_initialized():
            if dist.get_backend() != 'nccl':
                raise ValueError('RoPA process groups use the NCCL CUDA backend.')
            rank, world_size = dist.get_rank(), dist.get_world_size()
        return cls(device, rank, world_size, local, owns_group)

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier(device_ids=[self.device.index or 0])

    def sum(self, value):
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self.device).clone()
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def weighted_metrics(self, values, count):
        keys = sorted(values)
        packed = [values[key] * count for key in keys] + [count]
        sums = self.sum(packed)
        denominator = float(sums[-1])
        if denominator <= 0:
            raise ValueError('Distributed metric aggregation received no samples.')
        return {key: float(sums[index]) / denominator for index, key in enumerate(keys)}, int(denominator)

    def gather_state(self, state):
        if self.world_size == 1:
            return [state]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, state)
        return gathered

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()


class TemporalTrainingObjective(nn.Module):
    def __init__(self, model, anchor, objective):
        super().__init__()
        self.model = model
        self.anchor = anchor
        self.objective = objective

    def forward(self, video, spacing, step, steps):
        return self.objective(self.model, self.anchor, video, spacing, step, steps)


def distributed_objective(model, anchor, objective, group):
    wrapped = TemporalTrainingObjective(model, anchor, objective)
    if group.world_size == 1:
        return wrapped
    return DistributedDataParallel(
        wrapped,
        device_ids=[group.device.index],
        output_device=group.device.index,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )


def distributed_stream_indices(order, offset, batch_size, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('The stream rank is outside its process group.')
    if len(order) < world_size:
        raise ValueError('The clip dataset must have at least one example per CUDA process.')
    global_size = batch_size * world_size
    selected = order[offset:offset + global_size]
    if len(selected) == 0:
        raise ValueError('The stream must start a new epoch before requesting another batch.')
    remainder = len(selected) % world_size
    if remainder:
        padding = world_size - remainder
        selected = torch.cat((selected, order[:padding]))
    return selected[rank::world_size], min(len(order), offset + global_size)
