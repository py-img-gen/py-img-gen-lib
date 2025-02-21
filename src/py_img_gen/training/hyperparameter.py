import logging
import math
import pathlib
from dataclasses import dataclass

from py_img_gen.typehints import MixedPrecisionType

logger = logging.getLogger(__name__)


@dataclass
class BaseHyperparameter(object):
    train_batch_size: int
    num_train_epochs: int
    learning_rate: float
    output_dir: pathlib.Path

    seed: int = 19950815
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: MixedPrecisionType = "fp16"
    max_grad_norm: float = 1.0

    is_scale_lr: bool = False
    use_8bit_adam: bool = False
    validation_interval: int = 1
    num_workers: int = 4

    def __post_init__(self) -> None:
        logger.info(f"Make output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def total_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps

    def adjust_num_train_epochs(self, num_dataset: int) -> None:
        num_dataset_after_sharding = math.ceil(num_dataset)
        num_update_steps_per_epoch = math.ceil(
            num_dataset_after_sharding / self.gradient_accumulation_steps
        )
        max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        logger.info(
            f"Adjusted # epochs from {self.num_train_epochs} to {num_train_epochs}"
        )
        self.num_train_epochs = num_train_epochs

    def get_learning_rate(self) -> float:
        return (
            self.learning_rate
            * self.gradient_accumulation_steps
            * self.train_batch_size
            if self.is_scale_lr
            else self.learning_rate
        )
