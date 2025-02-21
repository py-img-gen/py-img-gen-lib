import gc
import logging
import pathlib
from tempfile import TemporaryDirectory
from typing import List, Optional, Type, Union

import diffusers
import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel

from py_img_gen.datasets.dreambooth import (
    DreamBoothExample,
)
from py_img_gen.typehints import SimpleTextToImageExamples

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
    pipe.set_progress_bar_config(leave=True)

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


def collate_fn(
    examples: List[DreamBoothExample],
    is_prior_preservation: bool,
) -> SimpleTextToImageExamples:
    input_ids = [example["instance"]["prompt_ids"] for example in examples]
    pixel_values = [example["instance"]["images"] for example in examples]

    # 事前保存のためにクラスとインスタンスの例を concat する
    # 以下の操作によりミニバッチ化して 2 回の順伝播を行うのを避ける
    if is_prior_preservation:
        for example in examples:
            class_example = example["classes"]
            assert class_example is not None
            input_ids.append(class_example["prompt_ids"])
            pixel_values.append(class_example["images"])

    pixel_values_th = torch.stack(pixel_values)
    pixel_values_th = pixel_values_th.to(memory_format=torch.contiguous_format).float()

    input_ids_th = torch.cat(input_ids, dim=0)

    return {
        "input_ids": input_ids_th,
        "pixel_values": pixel_values_th,
    }


def load_dreambooth_pipeline(
    model_id: str,
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    pipeline_cls: Type[StableDiffusionPipeline] = StableDiffusionPipeline,
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.float16,
    disable_progress_bar: bool = True,
) -> StableDiffusionPipeline:
    assert issubclass(pipeline_cls, StableDiffusionPipeline)
    if disable_progress_bar:
        diffusers.utils.logging.disable_progress_bar()

    pipe = pipeline_cls.from_pretrained(
        model_id,
        unet=unet,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )

    with TemporaryDirectory() as tmp_dir:
        pipe.save_pretrained(tmp_dir)

        pipe = pipeline_cls.from_pretrained(tmp_dir, torch_dtype=torch_dtype)

    pipe.set_progress_bar_config(disable=disable_progress_bar)
    pipe = pipe.to(device=device)

    return pipe
