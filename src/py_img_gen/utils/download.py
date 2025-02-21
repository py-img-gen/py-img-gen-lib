import logging
import pathlib
from typing import Sequence

from diffusers.utils import load_image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def download_image(
    image_url: str,
    save_path: pathlib.Path,
    force_download: bool = False,
) -> None:
    if save_path.exists() and not force_download:
        return

    logger.info(f"Downloading image from {image_url} to {save_path}")
    image = load_image(image_url)
    image.save(save_path)


def download_images(
    image_urls: Sequence[str],
    save_dir_path: pathlib.Path,
) -> None:
    for i, image_url in enumerate(tqdm(image_urls, desc="Downloading images")):
        save_path = save_dir_path / f"{i}.png"
        download_image(image_url=image_url, save_path=save_path)
