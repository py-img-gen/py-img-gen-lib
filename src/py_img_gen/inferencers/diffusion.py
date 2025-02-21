from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional, Union

import matplotlib.animation as animation
import torch
from diffusers.models import UNet2DModel
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
)
from diffusers.utils import make_image_grid
from diffusers.utils.torch_utils import randn_tensor
from ncsn.scheduler import (
    AnnealedLangevinDynamicsScheduler as ALDScheduler,
)
from PIL.Image import Image as PilImage
from tqdm.auto import tqdm

from py_img_gen.utils import (
    create_animation_gif,
    decode_images,
)

if TYPE_CHECKING:
    from py_img_gen.trainers import SchedulerUnion
    from py_img_gen.trainers.config import (
        BaseTrainConfig,
        EvalConfig,
    )


@torch.no_grad()
def inference(
    unet: UNet2DModel,
    noise_scheduler: SchedulerUnion,
    train_config: BaseTrainConfig,
    n_inference_steps: Optional[int] = None,
    only_final: bool = True,
) -> Union[List[PilImage], List[List[PilImage]]]:
    """Inference for generating images using the UNet denoiser model.

    Args:
        unet (UNet2DModel): The UNet denoiser model.
        noise_scheduler (Union[DDPMScheduler, DDIMScheduler]): The noise scheduler.
        train_config (TrainConfig): The training configuration.
        n_inference_steps (Optional[int], optional): The number of inference steps. Defaults to None.
        only_final (bool, optional): Whether to return only the final image. Defaults to True.

    Returns:
        Union[List[PilImage], List[List[PilImage]]]: The generated images.
            if only_final is True, it returns a batched (list of) PIL images, otherwise,
            it returns a list of list of PIL images that contains intermediate images.
    """
    # By importing the inferencers here, we avoid circular imports
    from py_img_gen.inferencers import (
        DDPMInference,
        NCSNInference,
    )

    # Set unet denoiser  to the eval mode
    unet.eval()  # type: ignore[attr-defined]

    # Set the timesteps of the scheduler for inference steps
    n_inference_steps = n_inference_steps or train_config.num_timesteps
    noise_scheduler.set_timesteps(n_inference_steps)  # type: ignore[union-attr]

    # Create random number generator for inference for reproducibility
    generator = torch.manual_seed(train_config.seed)

    # Set the shape of the noise and then generate random noise
    x_shape = (
        train_config.batch_size,
        unet.config.in_channels,  # type: ignore[attr-defined]
        unet.config.sample_size,  # type: ignore[attr-defined]
        unet.config.sample_size,  # type: ignore[attr-defined]
    )
    # Use torch.rand for ALD scheduler and randn_tensor for DDPM and DDIM schedulers
    randn_tensor_func = (
        torch.rand if isinstance(noise_scheduler, ALDScheduler) else randn_tensor
    )
    # Generate a random sample
    # NOTE: The behavior of random number generation is different between CPU and GPU,
    # so first generate random numbers on CPU and then move them to GPU (if available).
    x = randn_tensor_func(x_shape, generator=generator)  # type: ignore[operator]
    x = x.to(unet.device)  # type: ignore[attr-defined]

    # Set the inference module based on the scheduler type
    inferencer_module = (
        NCSNInference if isinstance(noise_scheduler, ALDScheduler) else DDPMInference
    )
    inferencer = inferencer_module(unet, noise_scheduler)

    # Perform the reverse diffusion process
    intermediate_images = []
    for t in tqdm(
        noise_scheduler.timesteps,  # type: ignore[union-attr]
        desc="Generating...",
        leave=False,
    ):
        x = inferencer(
            x=x,
            t=t,
            generator=generator,
            train_config=train_config,  # type: ignore[arg-type]
        )
        if not only_final:
            # Store the intermediate images if only_final is False
            intermediate_images.append(decode_images(x))

    # Return the final image or intermediate images based on `only_final`
    return decode_images(x) if only_final else intermediate_images


def animation_inference(
    train_config: BaseTrainConfig,
    eval_config: EvalConfig,
    unet: UNet2DModel,
    noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
    n_inference_steps: Optional[int] = None,
    n_frames: int = 10,
) -> animation.ArtistAnimation:
    """Generate an animation of the inference process.

    Args:
        train_config (TrainConfig): The training configuration.
        eval_config (EvalConfig): The evaluation configuration.
        unet (UNet2DModel): The UNet denoiser model.
        noise_scheduler (Union[DDPMScheduler, DDIMScheduler]): The noise scheduler.
        n_inference_steps (Optional[int], optional): The number of inference steps. Defaults to None.
        n_frames (int, optional): The number of frames for the animation. Defaults to 10.

    Returns:
        animation.ArtistAnimation: The animation of the inference
    """
    # Set `only_final` to `False` to get intermediate images and get a list of images
    images_list = inference(
        unet=unet,
        noise_scheduler=noise_scheduler,
        train_config=dataclasses.replace(
            train_config,
            batch_size=eval_config.num_generate_images,
        ),
        n_inference_steps=n_inference_steps,
        only_final=False,
    )

    # Create a grid of images for each timestep
    images = [
        make_image_grid(
            images=images,  # type: ignore[arg-type]
            rows=eval_config.num_grid_rows,
            cols=eval_config.num_grid_cols,
        )
        for images in images_list
    ]

    # Create an animation GIF from the animation images
    return create_animation_gif(images=images, num_frames=n_frames)
