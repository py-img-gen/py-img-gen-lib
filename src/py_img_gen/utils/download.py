import logging
import os
from typing import Sequence

from diffusers.utils import load_image
from tqdm.auto import tqdm

from py_img_gen.typehints import PathLike

logger = logging.getLogger(__name__)


def download_image(image_url: str, save_path: PathLike) -> None:
    logger.info(f"Downloading image from {image_url} to {save_path}")
    image = load_image(image_url)
    image.save(save_path)


def download_images(
    image_urls: Sequence[str],
    save_dir_path: PathLike,
) -> None:
    for i, image_url in enumerate(tqdm(image_urls)):
        save_path = os.path.join(save_dir_path, f"{i}.png")
        download_image(image_url=image_url, save_path=save_path)
