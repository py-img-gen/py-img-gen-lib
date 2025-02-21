import logging
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import pytest
import torch
import torchvision
from diffusers.models import UNet2DModel
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from ncsn.scheduler import AnnealedLangevinDynamicsScheduler as ALDScheduler
from ncsn.unet import UNet2DModelForNCSN
from pytest_mock import MockerFixture
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import set_seed

from py_img_gen import trainers
from py_img_gen.trainers import get_device, get_simple_resize_transforms
from py_img_gen.trainers.config import (
    BaseModelConfig,
    BaseTrainConfig,
    DDPMModelConfig,
    EvalConfig,
    NCSNModelConfig,
    TrainDDPMConfig,
    TrainNCSNConfig,
)
from py_img_gen.utils.testing import PyImgGenTestCase

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    argnames="expected_device",
    argvalues=("cpu", "cuda"),
)
def test_get_device(mocker: MockerFixture, expected_device: str):
    mocker.patch(
        "torch.cuda.is_available",
        return_value=expected_device == "cuda",
    )
    device = get_device()
    assert device.type == expected_device


class TestDiffusionTrainer(PyImgGenTestCase):
    @pytest.fixture(autouse=True)
    def set_seed(self, train_config: TrainDDPMConfig):
        logger.info(f"Setting seed to {train_config.seed}")
        set_seed(train_config.seed)

    @pytest.fixture
    def device(self) -> torch.device:
        return get_device()

    @pytest.fixture
    def eval_config(self) -> EvalConfig:
        return EvalConfig()

    @pytest.fixture
    def optim(self, unet: UNet2DModel, train_config: BaseTrainConfig) -> Optimizer:
        return torch.optim.Adam(unet.parameters(), lr=train_config.lr)

    @pytest.fixture
    def dataset(
        self, model_config: BaseModelConfig, omit_size: Optional[int] = None
    ) -> torchvision.datasets.MNIST:
        dataset = torchvision.datasets.MNIST(
            root="~/.cache",
            train=True,
            download=True,
            transform=get_simple_resize_transforms(
                sample_size=model_config.sample_size,
            ),
        )
        if omit_size is None:
            return dataset

        indices = np.random.choice(
            np.arange(len(dataset)), size=omit_size, replace=False
        )
        dataset.data = dataset.data[indices]
        dataset.targets = dataset.targets[indices]
        assert len(dataset) == omit_size

        return dataset

    @pytest.fixture
    def data_loader(
        self, dataset: torchvision.datasets.MNIST, train_config: BaseTrainConfig
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_config.num_workers,
        )


class TestDDPMTrainer(TestDiffusionTrainer):
    @pytest.fixture
    def train_config(self) -> TrainDDPMConfig:
        return TrainDDPMConfig(
            output_dir=self.TEST_DIR,
            # batch_size=2,
            # num_epochs=1,
            # num_timesteps=1,
        )

    @pytest.fixture
    def model_config(self) -> DDPMModelConfig:
        return DDPMModelConfig()

    @pytest.fixture
    def unet(self, model_config: BaseModelConfig, device: torch.device) -> UNet2DModel:
        unet = UNet2DModel(**asdict(model_config))
        unet = unet.to(device)
        return unet

    @pytest.fixture
    def scheduler(self, train_config: TrainDDPMConfig) -> DDPMScheduler:
        return DDPMScheduler(
            num_train_timesteps=train_config.num_timesteps,
            beta_start=train_config.beta_1,
            beta_end=train_config.beta_T,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or bool(os.environ.get("CI", False)),
        reason="No GPUs available for testing.",
    )
    def test_train_function(
        self,
        train_config: TrainDDPMConfig,
        eval_config: EvalConfig,
        unet: UNet2DModel,
        scheduler: DDPMScheduler,
        optim: Optimizer,
        data_loader: DataLoader,
        device: torch.device,
        epoch_filename_template="{}.png",
        validation_filename: str = "validation.png",
    ) -> None:
        trainers.train(
            train_config=train_config,
            eval_config=eval_config,
            unet=unet,
            noise_scheduler=scheduler,
            optim=optim,
            data_loader=data_loader,
            device=device,
            validation_filename=validation_filename,
            epoch_filename_template=epoch_filename_template,
        )

        assert (self.TEST_DIR / validation_filename).exists()
        for epoch in range(train_config.num_epochs):
            if epoch % eval_config.eval_epoch != 0:
                continue

            assert (self.TEST_DIR / epoch_filename_template.format(epoch)).exists()

        breakpoint()


class TestDDIMTrainer(TestDDPMTrainer):
    @pytest.fixture
    def train_config(self) -> TrainDDPMConfig:
        return TrainDDPMConfig(
            output_dir=self.TEST_DIR,
            num_epochs=20,
            # batch_size=2,
            # num_timesteps=1,
        )

    @pytest.fixture
    def scheduler(  # type: ignore[override]
        self,
        train_config: TrainDDPMConfig,
    ) -> DDIMScheduler:
        return DDIMScheduler(
            num_train_timesteps=train_config.num_timesteps,
            beta_start=train_config.beta_1,
            beta_end=train_config.beta_T,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or bool(os.environ.get("CI", False)),
        reason="No GPUs available for testing.",
    )
    def test_train_function(
        self,
        train_config: TrainDDPMConfig,
        eval_config: EvalConfig,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,  # type: ignore[override]
        optim: Optimizer,
        data_loader: DataLoader,
        device: torch.device,
        epoch_filename_template="{}.png",
        validation_filename: str = "validation.png",
    ) -> None:
        super().test_train_function(
            train_config=train_config,
            eval_config=eval_config,
            unet=unet,
            scheduler=scheduler,  # type: ignore[arg-type]
            optim=optim,
            data_loader=data_loader,
            device=device,
            epoch_filename_template=epoch_filename_template,
            validation_filename=validation_filename,
        )


class TestNCSNTrainer(TestDiffusionTrainer):
    @pytest.fixture
    def train_config(self) -> TrainNCSNConfig:
        return TrainNCSNConfig(output_dir=self.TEST_DIR)

    @pytest.fixture
    def model_config(self) -> NCSNModelConfig:
        return NCSNModelConfig()

    @pytest.fixture
    def unet(
        self,
        train_config: TrainNCSNConfig,
        model_config: BaseModelConfig,
        device: torch.device,
    ) -> UNet2DModelForNCSN:
        unet = UNet2DModelForNCSN(
            num_train_timesteps=train_config.num_timesteps,
            **asdict(model_config),
        )
        unet = unet.to(device)
        return unet

    @pytest.fixture
    def scheduler(
        self, train_config: TrainNCSNConfig, model_config: NCSNModelConfig
    ) -> ALDScheduler:
        return ALDScheduler(
            num_train_timesteps=train_config.num_timesteps,
            num_annealed_steps=train_config.num_annealed_timesteps,
            sigma_min=model_config.sigma_min,
            sigma_max=model_config.sigma_max,
            sampling_eps=train_config.sampling_eps,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or bool(os.environ.get("CI", False)),
        reason="No GPUs available for testing.",
    )
    def test_train_function(
        self,
        train_config: TrainNCSNConfig,
        eval_config: EvalConfig,
        unet: UNet2DModelForNCSN,
        scheduler: ALDScheduler,
        optim: Optimizer,
        data_loader: DataLoader,
        device: torch.device,
        epoch_filename_template="{}.png",
        validation_filename: str = "validation.png",
    ) -> None:
        trainers.train(
            train_config=train_config,
            eval_config=eval_config,
            unet=unet,
            noise_scheduler=scheduler,
            optim=optim,
            data_loader=data_loader,
            device=device,
            validation_filename=validation_filename,
            epoch_filename_template=epoch_filename_template,
        )

        assert (self.TEST_DIR / validation_filename).exists()
        for epoch in range(train_config.num_epochs):
            if epoch % eval_config.eval_epoch != 0:
                continue

            assert (self.TEST_DIR / epoch_filename_template.format(epoch)).exists()

        breakpoint()
