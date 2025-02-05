import pathlib
from dataclasses import dataclass
from typing import List, Optional, TypedDict

import torch
import torchvision.transforms as T
from diffusers.utils import load_image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from py_img_gen.utils import tokenize_prompt


class Example(TypedDict):
    images: torch.Tensor
    prompt_ids: torch.Tensor


class DreamBoothExample(TypedDict):
    instance: Example
    classes: Optional[Example]


@dataclass
class DreamBoothDataset(Dataset):
    instance_data_root: pathlib.Path
    instance_prompt: str
    tokenizer: CLIPTokenizer
    class_data_root: Optional[pathlib.Path] = None
    class_prompt: Optional[str] = None
    image_size: int = 512
    is_center_crop: bool = False

    _instance_image_paths: Optional[List[pathlib.Path]] = None
    _class_image_paths: Optional[List[pathlib.Path]] = None
    _image_transforms: Optional[T.Compose] = None

    def __post_init__(self) -> None:
        assert self.instance_data_root.exists()
        self._instance_image_paths = list(self.instance_data_root.iterdir())

    @property
    def instance_image_paths(self) -> List[pathlib.Path]:
        assert self._instance_image_paths is not None
        return self._instance_image_paths

    @property
    def class_image_paths(
        self,
    ) -> Optional[List[pathlib.Path]]:
        if self.class_data_root is None:
            return None

        return list(self.class_data_root.iterdir())

    @property
    def num_instance_images(self) -> int:
        return len(self.instance_image_paths)

    @property
    def num_class_images(self) -> int:
        return len(self.class_image_paths) if self.class_image_paths is not None else 0

    @property
    def dataset_length(self) -> int:
        return max(self.num_instance_images, self.num_class_images)

    @property
    def image_transforms(self) -> T.Compose:
        transforms = [
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size)
            if self.is_center_crop
            else T.RandomCrop(self.image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
        return T.Compose(transforms)

    def get_example(
        self,
        idx: int,
        image_paths: List[pathlib.Path],
        num_images: int,
        prompt: str,
    ) -> Example:
        #
        # 画像の読み込み
        #
        image_path = image_paths[idx % num_images]
        image = load_image(str(image_path))
        image_th = self.image_transforms(image)
        assert isinstance(image_th, torch.Tensor)

        #
        # プロンプトのトークナイズ
        #
        text_inputs = tokenize_prompt(prompt=prompt, tokenizer=self.tokenizer)

        return {
            "images": image_th,
            "prompt_ids": text_inputs.input_ids,
        }

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> DreamBoothExample:
        #
        # Instance データの取得
        #
        instance_example = self.get_example(
            idx,
            image_paths=self.instance_image_paths,
            num_images=self.num_instance_images,
            prompt=self.instance_prompt,
        )
        if self.class_data_root is None:
            return {
                "instance": instance_example,
                "classes": None,
            }
        #
        # Class データも使用する場合
        #
        assert self.class_image_paths is not None and self.class_prompt is not None
        class_example = self.get_example(
            idx,
            image_paths=self.class_image_paths,
            num_images=self.num_class_images,
            prompt=self.class_prompt,
        )
        return {
            "instance": instance_example,
            "classes": class_example,
        }
