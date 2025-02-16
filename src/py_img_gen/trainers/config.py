import pathlib
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BaseTrainConfig(object):
    """Base configuration for training."""

    output_dir: pathlib.Path = field(
        default=pathlib.Path.cwd() / "outputs",
        metadata={"help": "The output directory."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "The random seed for reproducibility."},
    )
    batch_size: int = field(
        default=256,
        metadata={"help": "The batch size."},
    )
    num_epochs: int = field(
        default=10,
        metadata={"help": "The number of epochs."},
    )
    num_timesteps: int = field(
        default=500,
        metadata={"help": "The number of timesteps."},
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "The learning rate."},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "The number of workers."},
    )

    def __post_init__(self) -> None:
        self.output_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class TrainDDPMConfig(BaseTrainConfig):
    """Configuration for training Denoising Diffusion Probabilistic Models (DDPM)."""

    beta_1: float = field(
        default=1e-4,
        metadata={"help": r"The $\beta_1$ value."},
    )
    beta_T: float = field(
        default=0.02,
        metadata={"help": r"The $\beta_T$ value."},
    )
    eta_ddim: float = field(
        default=0.0,
        metadata={
            "help": r"The $\eta$ value. When eta is 0, it is treated as DDIM, and when eta is 1, it is treated as DDPM. By setting a value between 0 and 1, it is possible to interpolate between DDIM and DDPM."
        },
    )


@dataclass
class TrainNCSNConfig(BaseTrainConfig):
    """Configuration for training Noise Conditional Score Networks (NCSN)."""

    num_epochs: int = field(
        default=150,
        metadata={"help": "The number of epochs."},
    )
    num_timesteps: int = field(
        default=10,
        metadata={"help": "The number of timesteps for NCSN."},
    )
    num_annealed_timesteps: int = field(
        default=100,
        metadata={"help": "The number of annealed timesteps for NCSN."},
    )
    sampling_eps: float = field(
        default=1e-5,
        metadata={"help": "Sampling epsilon"},
    )


@dataclass
class EvalConfig(object):
    """Configuration for evaluation."""

    eval_epoch: int = field(
        default=1,
        metadata={"help": "The evaluation epoch interval."},
    )
    num_generate_images: int = field(
        default=16,
        metadata={"help": "The number of images to generate."},
    )
    num_grid_rows: int = field(
        default=4,
        metadata={"help": "The number of rows in the grid."},
    )
    num_grid_cols: int = field(
        default=4,
        metadata={"help": "The number of columns in the grid."},
    )


@dataclass
class BaseModelConfig(object):
    """Base configuration for the model."""

    sample_size: int = field(
        default=32,
        metadata={"help": "Size of the input image"},
    )


@dataclass
class DDPMModelConfig(BaseModelConfig):
    """Configuration for the DDPM model."""

    in_channels: int = field(
        default=1,
        metadata={"help": "Number of input channels"},
    )
    out_channels: int = field(
        default=1,
        metadata={"help": "Number of output channels"},
    )
    block_out_channels: Tuple[int, ...] = field(
        default=(64, 128, 256, 512),
        metadata={"help": "Number of output channels for each block"},
    )
    layers_per_block: int = field(
        default=3,
        metadata={"help": "Number of layers per block"},
    )
    down_block_types: Tuple[str, ...] = field(
        default=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        metadata={"help": "Types of down blocks"},
    )
    up_block_types: Tuple[str, ...] = field(
        default=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        metadata={"help": "Types of up blocks"},
    )


@dataclass
class NCSNModelConfig(DDPMModelConfig):
    """Configuration for the NCSN model.

    This config is extended from the DDPM model config to include the sigma parameters for the NCSN.
    """

    sigma_min: float = field(
        default=0.005,
        metadata={"help": "Minimum value of sigma for the NCSN"},
    )
    sigma_max: float = field(
        default=10,
        metadata={"help": "Maximum value of sigma the NCSN"},
    )
