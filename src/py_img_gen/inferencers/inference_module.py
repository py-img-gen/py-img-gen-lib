from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch
import torch.nn as nn
from diffusers.models import UNet2DModel
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from diffusers.schedulers.scheduling_ddim import (
    DDIMSchedulerOutput,
)
from diffusers.schedulers.scheduling_ddpm import (
    DDPMSchedulerOutput,
)
from ncsn.scheduler import AnnealedLangevinDynamicsOutput as ALDOutput
from ncsn.scheduler import AnnealedLangevinDynamicsScheduler as ALDScheduler

if TYPE_CHECKING:
    from py_img_gen.trainers import SchedulerUnion
    from py_img_gen.trainers.config import BaseTrainConfig, TrainDDPMConfig


class InferenceModule(nn.Module):
    def __init__(self, unet: UNet2DModel, scheduler: SchedulerUnion) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        train_config: BaseTrainConfig,
        generator: torch.Generator,
    ) -> torch.Tensor:
        raise NotImplementedError


class DDPMInference(InferenceModule):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
    ) -> None:
        super().__init__(unet, scheduler)

    def __call__(  # type: ignore[override]
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        train_config: TrainDDPMConfig,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Predict noise `z` as `z_pred`
        z_pred = self.unet(x, t).sample  # type: ignore

        # Prepare the scheduler arguments. The arguments depend on the scheduler type.
        scheduler_kwargs = {
            "model_output": z_pred,
            "timestep": t,
            "sample": x,
            "generator": generator,
        }
        if isinstance(self.scheduler, DDIMScheduler):
            # Set the eta value for DDIMScheduler
            scheduler_kwargs["eta"] = train_config.eta_ddim

        # Compute previous image: x_{t} -> x_{t-1}
        output = self.scheduler.step(**scheduler_kwargs)  # type: ignore[union-attr]

        # Get previous sample from the output based on the output type
        x = (
            output.prev_sample
            if isinstance(
                output,
                (DDPMSchedulerOutput, DDIMSchedulerOutput),
            )
            else output[0]
        )
        return x


class NCSNInference(InferenceModule):
    scheduler: ALDScheduler

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: ALDScheduler,
    ) -> None:
        super().__init__(unet, scheduler)

    def __call__(  # type: ignore[override]
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        generator: torch.Generator,
        **kwargs,
    ) -> torch.Tensor:
        # Perform `num_annnealed_steps` annealing steps
        for _ in range(self.scheduler.num_annealed_steps):
            # Predict the score using the model
            model_output = self.unet(x, t).sample  # type: ignore

            # Perform the annealed langevin dynamics
            output = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=x,
                generator=generator,
            )
            x = output.prev_sample if isinstance(output, ALDOutput) else output[0]
        return x
