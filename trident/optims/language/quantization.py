"""Portable, frozen blockwise NF4 weights for the native decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# NormalFloat4 reconstruction levels, in ascending order.
_NF4 = (-1.0, -.6961928, -.52507305, -.3949175, -.28444138, -.18477343,
        -.09105004, 0., .07958030, .16093020, .24611230, .33791524,
        .44070983, .562617, .72295684, 1.)


class NF4Linear(nn.Module):
    block_size = 64

    def __init__(self, linear):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("native NF4 expects a bias-free projection")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        blocks = (linear.weight.numel() + self.block_size - 1) // self.block_size
        device = linear.weight.device
        self.register_buffer("packed_weight", torch.empty(blocks * 32, device=device, dtype=torch.uint8))
        self.register_buffer("absmax", torch.empty(blocks, device=device, dtype=torch.float32))

    @torch.no_grad()
    def pack(self, source, device):
        flat = source.reshape(-1)
        if flat.numel() != self.in_features * self.out_features:
            raise ValueError("incorrect NF4 source size")
        self.to_empty(device=device)
        levels = torch.tensor(_NF4, device=device)
        boundaries = (levels[:-1] + levels[1:]) / 2
        for start in range(0, flat.numel(), 65536):
            values = flat[start:start + 65536].to(device=device, dtype=torch.float32)
            values = F.pad(values, (0, -values.numel() % self.block_size)).view(-1, self.block_size)
            scales = values.abs().amax(1)
            codes = torch.bucketize(values / scales.masked_fill(scales == 0, 1)[:, None],
                                    boundaries).flatten().byte()
            packed = (codes[::2] << 4) | codes[1::2]
            self.packed_weight[start // 2:start // 2 + packed.numel()].copy_(packed)
            offset = start // self.block_size
            self.absmax[offset:offset + len(scales)].copy_(scales)

    def unpack(self, dtype):
        levels = torch.tensor(_NF4, device=self.packed_weight.device, dtype=dtype)
        # A byte lookup avoids expanding each 4-bit code to a separate int64 index.
        byte = torch.arange(256, device=levels.device)
        lookup = torch.stack((levels[byte >> 4], levels[byte & 15]), dim=1)
        weight = torch.empty((len(self.absmax), self.block_size), device=levels.device, dtype=dtype)
        packed = self.packed_weight.view(len(self.absmax), self.block_size // 2)
        for start in range(0, len(self.absmax), 2048):
            block = packed[start:start + 2048]
            values = lookup[block.long()].reshape(-1, self.block_size)
            torch.mul(values, self.absmax[start:start + len(block), None], out=weight[start:start + len(block)])
        return weight.flatten()[:self.in_features * self.out_features].view(self.out_features, self.in_features)

    def forward(self, value):
        def project(x):
            return F.linear(x, self.unpack(x.dtype))
        if torch.is_grad_enabled() and value.requires_grad:
            return checkpoint(project, value, use_reentrant=False)
        return project(value)

    def dense(self, dtype):
        with torch.device("meta"):
            linear = nn.Linear(self.in_features, self.out_features, bias=False)
        linear.weight = nn.Parameter(self.unpack(dtype), requires_grad=False)
        return linear
