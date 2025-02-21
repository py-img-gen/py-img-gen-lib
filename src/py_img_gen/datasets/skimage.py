import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import skimage
from diffusers.utils import load_image
from more_itertools import sort_together

from py_img_gen.typehints import PilImage


def get_skimage_descriptions() -> Dict[str, str]:
    """Get the descriptions of the images in the scikit-image dataset for use in CLIP zero-shot classification exercises.

    Returns:
        Dict[str, str]: The descriptions of the images in the scikit-image dataset.
    """
    return {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer",
    }


@dataclass
class SkimageImageTextPair(object):
    """A dataclass to hold the images and their corresponding text descriptions from the scikit-image dataset."""

    images: List[PilImage]
    texts: List[str]


def get_skimage_text_pairs(
    descriptions_dict: Optional[Dict[str, str]] = None,
) -> SkimageImageTextPair:
    """Get the images and their corresponding text descriptions from the scikit-image dataset.

    Args:
        descriptions_dict (Optional[Dict[str, str]], optional): A dictionary of image names and their corresponding text descriptions. Defaults to None.

    Returns:
        SkimageImageTextPair: A dataclass containing the images and their corresponding text descriptions.
    """
    # Use the default value of the descriptions_dict if it is not provided
    descriptions_dict = descriptions_dict or get_skimage_descriptions()

    original_imgs: List[PilImage] = []
    original_txts: List[str] = []

    # Get the paths of images from skimage that are either .png or .jpg
    skimage_data_dir = skimage.data_dir  # type: ignore[attr-defined]
    filenames = [
        fname
        for fname in os.listdir(skimage_data_dir)
        if fname.endswith(".png") or fname.endswith(".jpg")
    ]
    for fname in filenames:
        name, _ = os.path.splitext(fname)
        if name not in descriptions_dict:
            continue

        image_path = os.path.join(skimage_data_dir, fname)
        original_imgs.append(load_image(image_path))

        original_txts.append(descriptions_dict[name])

    # Ensure that the number of images and texts are the same
    assert len(original_txts) == len(original_imgs)

    # Sort the images and texts together
    original_txts, original_imgs = sort_together((original_txts, original_imgs))  # type: ignore
    return SkimageImageTextPair(images=original_imgs, texts=original_txts)
