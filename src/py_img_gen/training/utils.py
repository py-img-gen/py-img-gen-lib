import pathlib
from typing import Union

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid


def generate_validation_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    target_dir: pathlib.Path,
    num_images_per_prompt: int = 4,
    epoch: Union[int, str] = "latest",
    seed: int = 0,
) -> None:
    device: torch.device = pipe.device  # type: ignore[attr-defined]

    with torch.autocast(str(device)):
        output = pipe(  # type: ignore[operator]
            prompt=prompt,
            generator=torch.manual_seed(seed),
            num_images_per_prompt=num_images_per_prompt,
        )
    images = output.images
    image = make_image_grid(images=images, rows=1, cols=len(images))
    if epoch != "latest":
        epoch_dir = target_dir / f"{epoch=:03d}"
        epoch_dir.mkdir(exist_ok=True, parents=True)

        epoch_path = epoch_dir / f"{prompt}.png"
        image.save(epoch_path)

    latest_dir = target_dir / "epoch='latest'"
    latest_dir.mkdir(exist_ok=True, parents=True)

    latest_path = latest_dir / f"{prompt}.png"
    image.save(latest_path)
