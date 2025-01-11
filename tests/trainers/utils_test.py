import gc

import pytest
import torch
import torch.nn as nn

from py_img_gen.trainers import flush_gpu_memory, get_device


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield

    # en: Clean-up after the test
    torch.cuda.empty_cache()
    gc.collect()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No GPUs available for testing.",
)
def test_flush_gpu_memory():
    device = get_device()
    assert device.type == "cuda"

    # Define models for testing
    model1 = nn.Linear(in_features=100, out_features=100)
    model1 = model1.to(device)

    model2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    model2 = model2.to(device)

    # Record the initial GPU memory usage
    initial_memory = torch.cuda.memory_allocated()

    model1(input=torch.randn(32, 100, device=device))
    model2(input=torch.randn(32, 3, 224, 224, device=device))

    # Execute the `flush_gpu_memory` function to be tested
    flush_gpu_memory(model1, model2)

    # Check the GPU memory usage after memory flush
    final_memory = torch.cuda.memory_allocated()

    assert final_memory < initial_memory
