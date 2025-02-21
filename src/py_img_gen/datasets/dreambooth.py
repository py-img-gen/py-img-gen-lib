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
    """A typed dictionary representing a data example containing image and prompt information.

    Attributes:
        images (torch.Tensor): A tensor containing image data.
        prompt_ids (torch.Tensor): A tensor containing prompt identifiers.
    """

    images: torch.Tensor
    prompt_ids: torch.Tensor


class DreamBoothExample(TypedDict):
    """A typed dictionary for representing an example used in the DreamBooth model, which includes both instance and optional class examples.

    Attributes:
        instance (Example): An instance of Example, typically representing the primary data example.
        classes (Optional[Example]): An optional instance of Example, representing class data examples
            that may be used for classification or other purposes.
    """

    instance: Example
    classes: Optional[Example]


@dataclass
class DreamBoothDataset(Dataset):
    """A dataset class for handling instances and associated class images used in the DreamBooth model,
    along with their corresponding prompts. The dataset facilitates loading and processing image data
    and tokenizing text prompts for input into machine learning models.

    Attributes:
        instance_data_root (pathlib.Path): Path to the directory containing instance images.
        instance_prompt (str): The text prompt associated with the instance images.
        tokenizer (CLIPTokenizer): The tokenizer used for processing text prompts.
        class_data_root (Optional[pathlib.Path], optional): Path to the directory for class images.
        class_prompt (Optional[str], optional): The text prompt for the class images.
        image_size (int, optional): The desired size of the images after processing. Default is 512.
        is_center_crop (bool, optional): Whether to apply a center crop to images. Default is False.

    Internal Attributes:
        _instance_image_paths (Optional[List[pathlib.Path]]): List of paths to instance images.
        _class_image_paths (Optional[List[pathlib.Path]]): List of paths to class images, if provided.
        _image_transforms (Optional[T.Compose]): Composed transformations to be applied to images.

    Methods:
        __post_init__(): Ensures that the instance data root exists and populates the instance image paths.
        instance_image_paths() -> List[pathlib.Path]: Returns the list of paths to instance images.
        class_image_paths() -> Optional[List[pathlib.Path]]: Returns the list of paths to class images, if provided.
        num_instance_images() -> int: Returns the number of instance images.
        num_class_images() -> int: Returns the number of class images, or 0 if not provided.
        dataset_length() -> int: Returns the maximum number of instance or class images to determine dataset length.
        image_transforms() -> T.Compose: Constructs and returns the image transformation pipeline.
        get_example(idx: int, image_paths: List[pathlib.Path], num_images: int, prompt: str) -> Example:
            Loads and transforms an image and tokenizes its prompt.
        __len__() -> int: Returns the length of the dataset, determined by the maximum of instance or class images.
        __getitem__(idx: int) -> DreamBoothExample: Retrieves a DreamBoothExample, including instance data and,
            if available, associated class data.
    """

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


# class DreamBoothDataset(Dataset):
#     """
#     A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
#     It pre-processes the images and the tokenizes prompts.
#     """

#     def __init__(
#         self,
#         instance_data_root,
#         instance_prompt,
#         tokenizer,
#         class_data_root=None,
#         class_prompt=None,
#         class_num=None,
#         size=512,
#         center_crop=False,
#         encoder_hidden_states=None,
#         class_prompt_encoder_hidden_states=None,
#         tokenizer_max_length=None,
#     ):
#         self.size = size
#         self.center_crop = center_crop
#         self.tokenizer = tokenizer
#         self.encoder_hidden_states = encoder_hidden_states
#         self.class_prompt_encoder_hidden_states = (
#             class_prompt_encoder_hidden_states
#         )
#         self.tokenizer_max_length = tokenizer_max_length

#         self.instance_data_root = pathlib.Path(
#             instance_data_root
#         )
#         if not self.instance_data_root.exists():
#             raise ValueError(
#                 f"Instance {self.instance_data_root} images root doesn't exists."
#             )

#         self.instance_images_path = list(
#             pathlib.Path(instance_data_root).iterdir()
#         )
#         self.num_instance_images = len(
#             self.instance_images_path
#         )
#         self.instance_prompt = instance_prompt
#         self._length = self.num_instance_images

#         if class_data_root is not None:
#             self.class_data_root = pathlib.Path(
#                 class_data_root
#             )
#             self.class_data_root.mkdir(
#                 parents=True, exist_ok=True
#             )
#             self.class_images_path = list(
#                 self.class_data_root.iterdir()
#             )
#             if class_num is not None:
#                 self.num_class_images = min(
#                     len(self.class_images_path), class_num
#                 )
#             else:
#                 self.num_class_images = len(
#                     self.class_images_path
#                 )
#             self._length = max(
#                 self.num_class_images,
#                 self.num_instance_images,
#             )
#             self.class_prompt = class_prompt
#         else:
#             self.class_data_root = None

#         self.image_transforms = transforms.Compose(
#             [
#                 transforms.Resize(
#                     size,
#                     interpolation=transforms.InterpolationMode.BILINEAR,
#                 ),
#                 transforms.CenterCrop(size)
#                 if center_crop
#                 else transforms.RandomCrop(size),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#         )

#     def __len__(self):
#         return self._length

#     def __getitem__(self, index):
#         example = {}
#         instance_image = Image.open(
#             self.instance_images_path[
#                 index % self.num_instance_images
#             ]
#         )
#         instance_image = exif_transpose(instance_image)

#         if not instance_image.mode == "RGB":
#             instance_image = instance_image.convert("RGB")
#         example["instance_images"] = self.image_transforms(
#             instance_image
#         )

#         if self.encoder_hidden_states is not None:
#             example["instance_prompt_ids"] = (
#                 self.encoder_hidden_states
#             )
#         else:
#             text_inputs = tokenize_prompt(
#                 self.tokenizer,
#                 self.instance_prompt,
#                 tokenizer_max_length=self.tokenizer_max_length,
#             )
#             example["instance_prompt_ids"] = (
#                 text_inputs.input_ids
#             )
#             example["instance_attention_mask"] = (
#                 text_inputs.attention_mask
#             )

#         if self.class_data_root:
#             class_image = Image.open(
#                 self.class_images_path[
#                     index % self.num_class_images
#                 ]
#             )
#             class_image = exif_transpose(class_image)

#             if not class_image.mode == "RGB":
#                 class_image = class_image.convert("RGB")
#             example["class_images"] = (
#                 self.image_transforms(class_image)
#             )

#             if (
#                 self.class_prompt_encoder_hidden_states
#                 is not None
#             ):
#                 example["class_prompt_ids"] = (
#                     self.class_prompt_encoder_hidden_states
#                 )
#             else:
#                 class_text_inputs = tokenize_prompt(
#                     self.tokenizer,
#                     self.class_prompt,
#                 )
#                 example["class_prompt_ids"] = (
#                     class_text_inputs.input_ids
#                 )
#                 example["class_attention_mask"] = (
#                     class_text_inputs.attention_mask
#                 )

#         return example
