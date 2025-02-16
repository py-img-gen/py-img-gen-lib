import dataclasses
from typing import Type, Union

import torch
from diffusers.models import UNet2DModel
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
)
from diffusers.utils import make_image_grid
from ncsn.scheduler import AnnealedLangevinDynamicsScheduler as ALDScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from py_img_gen.trainers.config import BaseTrainConfig, EvalConfig
from py_img_gen.trainers.loss_modules import LossDDPM, LossModule, LossNCSN

SchedulerUnion = Union[DDPMScheduler, DDIMScheduler, ALDScheduler]


def get_device() -> torch.device:
    """Get the device to use for training.

    Returns:
        torch.device: The device to use for training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_iteration(
    train_config: BaseTrainConfig,
    unet: UNet2DModel,
    noise_scheduler: SchedulerUnion,
    optim: Optimizer,
    loss_module: Type[LossModule],
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    """Train the UNet denoiser model over the entire data obtained from the data loader.

    Args:
        train_config (TrainConfig): The training configuration.
        unet (diffusers.UNet2DModel): The UNet denoiser model.
        noise_scheduler (Union[DDPMScheduler, DDIMScheduler, ANLDScheduler]): The noise scheduler.
        optim (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device to use for training
    """
    with tqdm(
        total=len(data_loader),
        desc="Iteration",
        leave=False,
    ) as pbar:
        for x, _ in data_loader:
            bsz = x.shape[0]
            x = x.to(device)

            t = torch.randint(
                low=0,
                high=train_config.num_timesteps,
                size=(bsz,),
                device=device,
            )
            z = torch.randn_like(x)
            x_noisy = noise_scheduler.add_noise(x, z, t)  # type: ignore
            loss = loss_module(unet)(x_noisy, z, t)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update()


def train(
    train_config: BaseTrainConfig,
    eval_config: EvalConfig,
    unet: UNet2DModel,
    noise_scheduler: SchedulerUnion,
    optim: Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch_filename_template: str = "{}.png",
    validation_filename: str = "validation.png",
) -> None:
    """Train the UNet denoiser model.

    Args:
        train_config (TrainConfig): The training configuration.
        eval_config (EvalConfig): The evaluation configuration.
        unet (diffusers.UNet2DModel): The UNet denoiser model.
        noise_scheduler (Union[DDPMScheduler, DDIMScheduler, ALDScheduler]): The noise scheduler.
        optim (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device to use for training
    """

    # Set unet denoiser model to the train mode
    unet.train()  # type: ignore[attr-defined]

    for epoch in tqdm(range(train_config.num_epochs), desc="Epoch"):
        loss_module = (
            LossNCSN if isinstance(noise_scheduler, ALDScheduler) else LossDDPM
        )
        train_iteration(
            train_config=train_config,
            unet=unet,
            noise_scheduler=noise_scheduler,
            optim=optim,
            data_loader=data_loader,
            loss_module=loss_module,
            device=device,
        )

        if epoch % eval_config.eval_epoch == 0:
            # By importing here, we avoid circular imports.
            from py_img_gen import inferencers

            images = inferencers.inference(
                unet=unet,
                noise_scheduler=noise_scheduler,
                train_config=dataclasses.replace(
                    train_config,
                    batch_size=eval_config.num_generate_images,
                ),
            )
            image = make_image_grid(
                images=images,  # type: ignore
                rows=eval_config.num_grid_rows,
                cols=eval_config.num_grid_cols,
            )
            image.save(train_config.output_dir / epoch_filename_template.format(epoch))
            image.save(train_config.output_dir / validation_filename)
