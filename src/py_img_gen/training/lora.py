import pathlib
from typing import Optional, Type

import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from torchvision import transforms

from py_img_gen.typehints import (
    SimpleTextToImageExamples,
)


def get_lora_transforms(
    image_size: int, center_crop: bool, random_flip: bool
) -> transforms.Compose:
    resize_transform = transforms.Resize(
        image_size,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    crop_transform = (
        transforms.CenterCrop(image_size)
        if center_crop
        else transforms.RandomCrop(image_size)
    )
    flip_transform = (
        transforms.RandomHorizontalFlip()
        if random_flip
        else transforms.Lambda(lambda x: x)
    )
    train_transforms = transforms.Compose(
        [
            resize_transform,
            crop_transform,
            flip_transform,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return train_transforms


def collate_fn(examples) -> SimpleTextToImageExamples:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }


def save_lora_weights(
    unet: UNet2DConditionModel,
    output_dir: pathlib.Path,
    epoch: int,
    pipeline_cls: Type[StableDiffusionPipeline] = StableDiffusionPipeline,
):
    save_dir_1 = output_dir / f"{epoch=:03d}"
    save_dir_2 = output_dir / "epoch='latest'"

    state_dict = get_peft_model_state_dict(unet)
    layers = convert_state_dict_to_diffusers(state_dict)

    for save_dir in (save_dir_1, save_dir_2):
        save_dir.mkdir(exist_ok=True, parents=True)

        pipeline_cls.save_lora_weights(
            save_directory=save_dir,
            unet_lora_layers=layers,
        )


def load_pipeline_with_lora(
    model_id: str,
    target_dir: pathlib.Path,
    epoch: int,
    pipeline_cls: Type[StableDiffusionPipeline] = StableDiffusionPipeline,
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    assert issubclass(pipeline_cls, StableDiffusionPipeline)

    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)

    load_dir = target_dir / f"{epoch=:03d}"
    pipe.unet.load_attn_procs(load_dir)

    return pipe
