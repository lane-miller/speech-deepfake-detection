"""
Utility functions for the DNN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_to_stacked(x):
    """
    Convert a complex tensor to a concatenated real tensor
    """

    return torch.cat([x.real, x.imag], dim=1)