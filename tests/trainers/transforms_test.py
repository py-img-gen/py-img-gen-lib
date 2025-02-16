import numpy as np
import pytest
import torch
from PIL import Image

from py_img_gen.trainers import get_simple_resize_transforms


@pytest.mark.parametrize(
    argnames="input_size",
    argvalues=(16, 128),
)
@pytest.mark.parametrize(
    argnames="sample_size",
    argvalues=(32, 224, 512),
)
def test_get_simple_resize_transforms(input_size: int, sample_size: int):
    transforms = get_simple_resize_transforms(sample_size)

    dummy_image = Image.fromarray(
        np.random.randint(
            low=0,
            high=255,
            size=(input_size, input_size, 3),
            dtype=np.uint8,
        )
    )
    output = transforms(dummy_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, sample_size, sample_size)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0 and output.max() <= 1.0
