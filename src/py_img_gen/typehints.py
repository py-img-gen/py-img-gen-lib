import pathlib
from typing import Annotated, Union

from PIL.Image import Image

PathLike = Union[str, pathlib.Path]
PilImage = Annotated[Image, "PIL Image"]

__all__ = [
    "PathLike",
    "PilImage",
]
