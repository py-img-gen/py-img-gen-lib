import pathlib
from typing import Annotated, Literal, TypedDict, Union

import torch
from PIL.Image import Image

PathLike = Union[str, pathlib.Path]

PilImage = Annotated[Image, "PIL Image"]

MixedPrecisionType = Literal["no", "fp16", "bf16"]

# Define the options for the characteristics to teach the model.
# Here, object and style are selectable.
LearnableProperty = Literal["object", "style"]


class SimpleTextToImageExamples(TypedDict):
    input_ids: torch.Tensor
    pixel_values: torch.Tensor


__all__ = [
    "PathLike",
    "PilImage",
    "SimpleTextToImageExamples",
]
