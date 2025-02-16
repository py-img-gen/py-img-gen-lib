from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from diffusers.utils import numpy_to_pil
from PIL.Image import Image as PilImage


def normalize_images(image: torch.Tensor) -> torch.Tensor:
    """Normalize the image to be between 0 and 1 using min-max normalization manner.

    Args:
        image (torch.Tensor): The batch of images to normalize.

    Returns:
        torch.Tensor: The normalized image.
    """

    assert image.ndim == 4, image.ndim
    batch_size = image.shape[0]

    def _normalize(img: torch.Tensor) -> torch.Tensor:
        return (img - img.min()) / (img.max() - img.min())

    for i in range(batch_size):
        image[i] = _normalize(image[i])
    return image


def decode_images(
    image_tensor: torch.Tensor,
) -> List[PilImage]:
    """Decode the image tensor to a list of PIL images.

    Args:
        image_tensor (torch.Tensor): The image tensor to decode.

    Returns:
        List[PilImage]: The list of PIL images.
    """
    image_tensor = image_tensor.detach().clone()
    image_tensor = normalize_images(image_tensor)
    image_tensor = image_tensor.permute(0, 2, 3, 1)
    return numpy_to_pil(image_tensor.cpu().numpy())


def create_animation_gif(
    images: List[PilImage],
    num_frames: int = 10,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "gray",
    num_last_repeat: int = 5,
) -> animation.ArtistAnimation:
    """Create an animation gif from a list of images.

    Args:
        images (List[PilImage]): The list of images to animate.
        num_frames (int, optional): The number of frames to create the animation. Defaults to 50.
        figsize (Tuple[int, int], optional): The figure size of the animation. Defaults to (8, 8).
        cmap (str, optional): The color map of the images. Defaults to "gray".
        num_last_repeat (int, optional): The number of times to repeat the last image to pause the animation. Defaults to 10.

    Returns:
        animation.ArtistAnimation: The animation object.
    """

    def reduce_to_n_elements(lst: List[PilImage], n: int) -> List[PilImage]:
        """Reduce the number of elements in the list to `n` elements."""
        assert n > 0, f"`n` must be greater than 0, but got {n}"

        length = len(lst)
        step = max(1, length // n)
        result = lst[::step]

        if lst[-1] not in result:
            if len(result) >= n:
                result[-1] = lst[-1]

        assert len(result) == n, len(result)
        return result

    # Thinning the number of images to reduce the computational burden of creating the animation
    images = reduce_to_n_elements(images, n=num_frames)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    artists = [[ax.imshow(image, animated=True, cmap=cmap)] for image in images]
    # Repeat the last image for `num_last_repeat` times to make the animation pause at the end
    artists += [artists[-1]] * num_last_repeat

    return animation.ArtistAnimation(fig=fig, artists=artists, blit=True)
