from dataclasses import asdict

import pytest
import torch
from diffusers.models import UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from ncsn.unet import UNet2DModelForNCSN
from transformers import set_seed

from py_img_gen.trainers import (
    LossDDPM,
    LossModule,
    LossNCSN,
)
from py_img_gen.trainers.config import (
    BaseTrainConfig,
    DDPMModelConfig,
    NCSNModelConfig,
    TrainDDPMConfig,
    TrainNCSNConfig,
)
from py_img_gen.utils.testing import PyImgGenTestCase


class BaseLossModuleTest(PyImgGenTestCase):
    @pytest.fixture
    def train_config(self) -> BaseTrainConfig:
        return BaseTrainConfig(output_dir=self.TEST_DIR)

    @pytest.fixture(autouse=True)
    def set_seed(self, train_config: BaseTrainConfig):
        set_seed(train_config.seed)


class TestLossModule(BaseLossModuleTest):
    @pytest.fixture
    def unet(self) -> UNet2DModel:
        return UNet2DModel(
            sample_size=16,
            in_channels=1,
            out_channels=1,
            num_train_timesteps=10,
        )

    def test_loss_module(self, unet: UNet2DModel, bsz: int = 2):
        loss_module = LossModule(unet=unet)
        x_shape = (
            bsz,
            unet.config.in_channels,  # type: ignore[attr-defined]
            unet.config.sample_size,  # type: ignore[attr-defined]
            unet.config.sample_size,  # type: ignore[attr-defined]
        )
        x = randn_tensor(shape=x_shape)
        t = torch.randint(
            0,
            unet.config.num_train_timesteps,  # type: ignore[attr-defined]
            (bsz,),
        )
        z = torch.randn_like(x)

        with pytest.raises(NotImplementedError):
            loss_module(x_noisy=x, z=z, t=t)


class TestLossDDPM(BaseLossModuleTest):
    @pytest.fixture
    def train_config(self) -> TrainDDPMConfig:
        return TrainDDPMConfig(output_dir=self.TEST_DIR)

    @pytest.fixture
    def model_config(self) -> DDPMModelConfig:
        return DDPMModelConfig()

    @pytest.fixture
    def unet(self, model_config: DDPMModelConfig) -> UNet2DModel:
        unet = UNet2DModel(**asdict(model_config))
        unet.eval()  # Set the model to evaluation mode for testing
        return unet

    def test_loss_ddpm_module(
        self,
        train_config: TrainDDPMConfig,
        unet: UNet2DModel,
        bsz: int = 2,
    ):
        x_shape = (
            bsz,
            unet.config.in_channels,  # type: ignore[attr-defined]
            unet.config.sample_size,  # type: ignore[attr-defined]
            unet.config.sample_size,  # type: ignore[attr-defined]
        )
        x = randn_tensor(shape=x_shape)
        z = torch.randn_like(x)
        t = torch.randint(0, train_config.num_timesteps, size=(bsz,))

        loss_module = LossDDPM(unet=unet)
        loss = loss_module(x_noisy=x, z=z, t=t)
        assert loss.item() == 1.1753438711166382


class TestLossNCSN(BaseLossModuleTest):
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
        model_config: NCSNModelConfig,
    ) -> UNet2DModelForNCSN:
        unet = UNet2DModelForNCSN(
            **asdict(model_config),
            num_train_timesteps=train_config.num_timesteps,
        )
        unet.eval()  # Set the model to evaluation mode for testing
        return unet

    def test_loss_ncsn_module(
        self,
        train_config: TrainNCSNConfig,
        unet: UNet2DModelForNCSN,
        bsz: int = 2,
    ):
        x_shape = (
            bsz,
            unet.config.in_channels,  # type: ignore[attr-defined]
            unet.config.sample_size,  # type: ignore[attr-defined]
            unet.config.sample_size,  # type: ignore[attr-defined]
        )
        x = randn_tensor(shape=x_shape)
        z = torch.randn_like(x)
        t = torch.randint(0, train_config.num_timesteps, size=(bsz,))

        loss_module = LossNCSN(unet=unet)
        loss = loss_module(x_noisy=x, z=z, t=t)
        # assert loss.item() == 1.1753438711166382
        assert loss.item() == 1.0680458545684814
