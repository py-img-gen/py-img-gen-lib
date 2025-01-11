import gc
from typing import Tuple

import torch
import torch.nn as nn


def flush_gpu_memory(*nn_modules: Tuple[nn.Module, ...]) -> None:
    r"""Flush GPU memory.

    Args:
        *nn_modules (Tuple[torch.nn.Module, ...]): The torch.nn.Mdoule instances to flush the GPU memory.
    """
    for nn_module in nn_modules:
        module_name = nn_module.__class__.__name__
        print(f"Flushing GPU memory for `{module_name}` ...")

        del nn_module

    gc.collect()
    torch.cuda.empty_cache()
