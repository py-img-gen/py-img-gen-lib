import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DModel
from einops import rearrange
from ncsn.unet import UNet2DModelForNCSN


class LossModule(nn.Module):
    r"""Base class for the loss module.

    Args:
        unet (UNet2DModel): The U-Net model.
    """

    def __init__(self, unet: UNet2DModel) -> None:
        super().__init__()
        self.unet = unet

    def __call__(
        self,
        x_noisy: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate the loss.

        Args:
            x_noisy (torch.Tensor): The noisy image.
            z (torch.Tensor): The target image.
            t (torch.Tensor): The time step.

        Returns:
            torch.Tensor: The loss value
        """
        raise NotImplementedError


class LossDDPM(LossModule):
    r"""Loss module for DDPM.

    Args:
        unet (UNet2DModel): The U-Net
    """

    def __init__(self, unet: UNet2DModel) -> None:
        super().__init__(unet)

    def __call__(
        self,
        x_noisy: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate the loss for DDPM.

        Args:
            x_noisy (torch.Tensor): The noisy image.
            z (torch.Tensor): The target image.
            t (torch.Tensor): The time step.

        Returns:
            torch.Tensor: The loss value
        """
        z_pred = self.unet(x_noisy, t).sample  # type: ignore
        loss = F.mse_loss(z_pred, z)
        return loss


class LossNCSN(LossModule):
    """Loss module for NCSN."""

    def __init__(self, unet: UNet2DModelForNCSN) -> None:
        super().__init__(unet)

    def __call__(
        self,
        x_noisy: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate the score using the model
        scores = self.unet(x_noisy, t).sample  # type: ignore

        # Calculate the target score
        used_sigmas = self.unet.sigmas[t]  # type: ignore
        used_sigmas = rearrange(used_sigmas, "b -> b 1 1 1")
        target = -1 / used_sigmas * z

        # Rearrange the tensors
        target = rearrange(target, "b c h w -> b (c h w)")
        scores = rearrange(scores, "b c h w -> b (c h w)")

        # Calculate the loss
        loss = F.mse_loss(scores, target, reduction="none")
        loss = loss.mean(dim=-1) * used_sigmas.squeeze() ** 2
        loss = loss.mean(dim=0)

        return loss
