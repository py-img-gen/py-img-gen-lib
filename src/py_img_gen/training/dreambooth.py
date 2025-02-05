import gc
import logging
import pathlib
from typing import Optional, Type, Union

import torch
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def generate_class_images(
    model_id: str,
    prompt: str,
    num_class_images: int,
    output_dir: Union[str, pathlib.Path],
    pipeline_cls: Type[StableDiffusionPipeline] = StableDiffusionPipeline,
    torch_dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
) -> None:
    if isinstance(output_dir, str):
        output_dir = pathlib.Path(output_dir)

    cur_class_images = len(list(output_dir.iterdir()))
    if cur_class_images >= num_class_images:
        return

    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    num_new_images = num_class_images - cur_class_images
    logger.info(f"Number of class images to sample: {num_new_images}")

    for idx in tqdm(
        range(0, num_new_images, batch_size),
        desc="Generating class images",
    ):
        output = pipe(
            prompt=prompt,
            num_images_per_prompt=batch_size,
        )
        images = output.images

        for i, image in enumerate(images):
            save_path = output_dir / f"{cur_class_images + idx + i}.png"
            print(f"Saving the image to `{save_path}`")
            image.save(save_path)

    # Clean-up the GPU memory
    pipe = pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
