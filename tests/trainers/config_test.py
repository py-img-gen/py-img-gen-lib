from dataclasses import asdict, fields

import pytest

from py_img_gen.trainers.config import (
    BaseTrainConfig,
    DDPMModelConfig,
    EvalConfig,
    NCSNModelConfig,
    TrainDDPMConfig,
    TrainNCSNConfig,
)
from py_img_gen.utils.testing import PyImgGenTestCase


class TestDiffusionTrainers(PyImgGenTestCase):
    @pytest.fixture
    def train_base_config(self) -> BaseTrainConfig:
        return BaseTrainConfig(output_dir=self.TEST_DIR)

    def test_train_base_config(self, train_base_config: BaseTrainConfig):
        expected = {
            "output_dir": self.TEST_DIR,
            "seed": 42,
            "batch_size": 256,
            "num_epochs": 10,
            "num_timesteps": 500,
            "lr": 1e-4,
            "num_workers": 4,
        }
        assert asdict(train_base_config).keys() == expected.keys()

        for field in fields(train_base_config):
            assert getattr(train_base_config, field.name) == expected[field.name]

        assert train_base_config.output_dir.exists()

    @pytest.fixture
    def train_ddpm_config(self) -> TrainDDPMConfig:
        return TrainDDPMConfig(output_dir=self.TEST_DIR)

    def test_train_ddpm_config(self, train_ddpm_config: TrainDDPMConfig):
        expected = {
            "output_dir": self.TEST_DIR,
            "seed": 42,
            "batch_size": 256,
            "num_epochs": 10,
            "num_timesteps": 500,
            "lr": 1e-4,
            "num_workers": 4,
            "beta_1": 1e-4,
            "beta_T": 0.02,
            "eta_ddim": 0.0,
        }
        assert asdict(train_ddpm_config).keys() == expected.keys()

        for field in fields(train_ddpm_config):
            assert getattr(train_ddpm_config, field.name) == expected[field.name]

        assert train_ddpm_config.output_dir.exists()

    @pytest.fixture
    def eval_config(self) -> EvalConfig:
        return EvalConfig()

    def test_eval_config(self, eval_config: EvalConfig):
        expected = {
            "eval_epoch": 1,
            "num_generate_images": 16,
            "num_grid_rows": 4,
            "num_grid_cols": 4,
        }
        assert asdict(eval_config).keys() == expected.keys()

        for field in fields(eval_config):
            assert getattr(eval_config, field.name) == expected[field.name]

    @pytest.fixture
    def train_ncsn_config(self) -> TrainNCSNConfig:
        return TrainNCSNConfig(output_dir=self.TEST_DIR)

    def test_train_ncsn_config(self, train_ncsn_config: TrainNCSNConfig):
        expected = {
            "output_dir": self.TEST_DIR,
            "seed": 42,
            "batch_size": 256,
            "num_epochs": 150,
            "num_timesteps": 10,
            "num_annealed_timesteps": 100,
            "sampling_eps": 1e-5,
            "lr": 1e-4,
            "num_workers": 4,
        }
        assert asdict(train_ncsn_config).keys() == expected.keys()

        for field in fields(train_ncsn_config):
            assert getattr(train_ncsn_config, field.name) == expected[field.name]

        assert train_ncsn_config.output_dir.exists()

    @pytest.fixture
    def ddpm_model_config(self) -> DDPMModelConfig:
        return DDPMModelConfig()

    def test_ddpm_model_config(self, ddpm_model_config: DDPMModelConfig):
        expected = {
            "sample_size": 32,
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": (64, 128, 256, 512),
            "layers_per_block": 3,
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        }
        assert asdict(ddpm_model_config).keys() == expected.keys()

        for field in fields(ddpm_model_config):
            assert getattr(ddpm_model_config, field.name) == expected[field.name]

    @pytest.fixture
    def ncsn_model_config(self) -> NCSNModelConfig:
        return NCSNModelConfig()

    def test_ncsn_model_config(self, ncsn_model_config: NCSNModelConfig):
        expected = {
            "sample_size": 32,
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": (64, 128, 256, 512),
            "layers_per_block": 3,
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            "sigma_min": 0.005,
            "sigma_max": 10,
        }
        assert asdict(ncsn_model_config).keys() == expected.keys()

        for field in fields(ncsn_model_config):
            assert getattr(ncsn_model_config, field.name) == expected[field.name]
