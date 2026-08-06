"""
Encoder classes: CNN, Self-Attention, Classification Head

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dnn.util import complex_to_stacked

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pool_size, activation=nn.ReLU, pool_type=nn.AvgPool1d):
        """
        1D convolutional block: conv -> batchnorm -> relu -> pool
        """
        super().__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.activation = activation()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = pool_type(pool_size)

    def forward(self, x):
        """
        forward pass of the convolutional block:
        x -> conv -> bn -> relu -> (proj) -> pool
        """
        identity = self.proj(x)

        y = self.conv(x)
        y = self.bn(y)
        y = self.activation(y)
        y = y + identity
        y = self.pool(y)

        return y

# TCN Block
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, activation=nn.ReLU):
        """
        Temporal Convolutional Network block
        """
        super().__init__()

        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.activation = activation()
        self.proj = nn.Identity()

    def forward(self, x):
        """
        forward pass of the Temporal Convolutional Network block:
        x -> conv -> bn -> relu -> (proj)
        """
        identity = self.proj(x)

        y = self.conv(x)
        y = self.bn(y)
        y = self.activation(y)
        y = y + identity

        return y


# Encoder
class Encoder(nn.Module):
    def __init__(self, block_specs):
        """
        Assembles a sequence of ConvBlock / TCNBlock modules from a list of
        block specs, so depth/type/params are configurable without editing
        this class.

        block_specs: list of dicts, each either:
          {"type": "conv", "in_ch": int, "out_ch": int, "kernel_size": int,
           "pool_size": int, "activation": nn.Module (optional), "pool_type": nn.Module (optional)}
        or:
          {"type": "tcn", "channels": int, "kernel_size": int, "dilation": int,
           "activation": nn.Module (optional)}
        """
        super().__init__()

        blocks = []
        for spec in block_specs:
            spec = dict(spec)  # avoid mutating caller's dict
            block_type = spec.pop("type")
            if block_type == "conv":
                blocks.append(ConvBlock(**spec))
            elif block_type == "tcn":
                blocks.append(TCNBlock(**spec))
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        forward pass of the encoder:
        x -> blocks
        """
        for block in self.blocks:
            x = block(x)

        return x