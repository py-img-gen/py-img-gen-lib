import os
import random
from typing import Literal, Sequence, TypedDict, cast

import numpy as np
import torch
from diffusers.utils import PIL_INTERPOLATION
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import (
    PreTrainedTokenizer,
)

# 画像の拡大縮小方法の選択肢を定義
InterpolationType = Literal["linear", "bilinear", "bicubic", "lanczos", "nearest"]


#
# 加工したデータセットを辞書型のデータに加工する際に
# key の定義と対応する value の型アノテーションを宣言
#
# 以下のように宣言することで、想定とは異なるデータが
# 入ってきた場合にエラーを出すことができる
#
class Example(TypedDict):
    input_ids: torch.Tensor
    pixel_values: torch.Tensor


#
# 読み込んだデータセットを加工する Dataset クラス
#
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        tokenizer: PreTrainedTokenizer,
        # learnable_property: LearnableProperty,
        templates: Sequence[str],
        image_size: int = 512,
        repeats: int = 100,
        interpolation: InterpolationType = "bicubic",
        flip_proba: float = 0.5,
        split: str = "train",
        placeholder_token: str = "*",
        is_center_crop: bool = False,
    ) -> None:
        self.data_root = data_root
        self.tokenizer = tokenizer
        # self.learnable_property = learnable_property
        self.image_size = image_size
        self.placeholder_token = placeholder_token
        self.is_center_crop = is_center_crop
        self.flip_proba = flip_proba

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]

        self.num_images = len(self.image_paths)
        self.dataset_length = (
            self.num_images * repeats if split == "train" else self.num_images
        )

        self.interpolation = PIL_INTERPOLATION[interpolation]

        # self.templates = (
        #     IMAGENET_STYLE_TEMPLATES_SMALL
        #     if learnable_property == "style"
        #     else IMAGENET_TEMPLATES_SMALL
        # )
        self.templates = templates

        self.flip_transform = transforms.RandomHorizontalFlip(self.flip_proba)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Example:
        # 画像のパスから画像を読み込み
        image_pil = cast(
            Image.Image,
            Image.open(self.image_paths[idx % self.num_images]),
        )

        if not image_pil.mode == "RGB":
            image_pil = image_pil.convert("RGB")

        placeholder_string = self.placeholder_token
        # 上記で定義したプロンプトのテンプレートリストからランダムに
        # 1 つ取ってきて、placeholder_string を埋め込む
        text = random.choice(self.templates).format(placeholder_string)

        # tokenizer で文章をトークン列に変換
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        image_arr = np.array(image_pil).astype(np.uint8)

        if self.is_center_crop:
            crop = min(*image_arr.shape)
            (
                h,
                w,
            ) = (
                image_arr.shape[0],
                image_arr.shape[1],
            )
            image_arr = image_arr[
                (h - crop) // 2 : (h + crop) // 2,
                (w - crop) // 2 : (w + crop) // 2,
            ]

        image_pil = Image.fromarray(image_arr)
        image_pil = image_pil.resize(
            (self.image_size, self.image_size),
            resample=self.interpolation,
        )

        image_pil = self.flip_transform(image_pil)
        image_arr = np.array(image_pil).astype(np.uint8)
        image_arr = (image_arr / 127.5 - 1.0).astype(np.float32)

        pixel_values = torch.from_numpy(image_arr).permute(2, 0, 1)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
