from typing import Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from py_img_gen.typehints import PilImage


def visualize_grid_image_for_image_text_pairs(
    images: Sequence[PilImage],
    texts: Sequence[str],
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Visualize a grid of images and their corresponding text descriptions.

    Args:
        images (Sequence[PilImage]): A sequence of images.
        texts (Sequence[str]): A sequence of text descriptions.
        nrows (int): The number of rows in the grid.
        ncols (int): The number of columns in the grid.
        figsize (Tuple[int, int]): The size of the figure.

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: A tuple of the figure and the axes.
    """
    assert len(images) == len(texts), (
        "The number of images and texts should be the same."
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(images[i * ncols + j])
            axes[i][j].axis("off")
            axes[i][j].set_title(texts[i * ncols + j], fontsize=10)
    return fig, axes


def visualize_image_text_similarity(
    images: Sequence[PilImage],
    texts: Sequence[str],
    similarity: Union[np.ndarray, torch.Tensor],
    figsize: Tuple[int, int] = (20, 14),
    vmin: float = 0.1,
    vmax: float = 0.3,
    fontsize_title: int = 20,
    fontsize_ticks: int = 18,
    fontsize_text: int = 12,
    title_text: str = "Cosine similarity between text and image features",
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize the similarity between images and text descriptions.

    Args:
        images (Sequence[PilImage]): A sequence of images.
        texts (Sequence[str]): A sequence of text descriptions.
        similarity (Union[np.ndarray, torch.Tensor]): The similarity between images and text descriptions.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (20,14).
        vmin (float, optional): The minimum value of the similarity. Defaults to 0.1.
        vmax (float, optional): The maximum value of the similarity. Defaults to 0.3.
        fontsize_title (int, optional): The fontsize of the title. Defaults to 20.
        fontsize_ticks (int, optional): The fontsize of the ticks. Defaults to 18.
        fontsize_text (int, optional): The fontsize of the text. Defaults to 12.
        title_text (str, optional): The title of the plot. Defaults to "Cosine similarity between text and image features".

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple of the figure and the axes.
    """
    assert len(images) == len(texts)
    num_samples = len(images)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(similarity, vmin=vmin, vmax=vmax)

    ax.set_yticks(
        range(num_samples),
        labels=texts,
        fontsize=fontsize_ticks,
    )
    ax.set_xticks([])

    for i, img in enumerate(images):
        extent = (i - 0.5, i + 0.5, -1.6, -0.6)  # FIXME
        ax.imshow(img, extent=extent, origin="lower")

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            s = f"{similarity[y, x]:.2f}"
            a = "center"
            ax.text(x, y, s=s, ha=a, va=a, size=fontsize_text)

    for side in ("left", "top", "right", "bottom"):
        plt.gca().spines[side].set_visible(False)

    ax.set_xlim((-0.5, num_samples - 0.5))
    ax.set_ylim((num_samples - 0.5, -2))

    ax.set_title(title_text, size=fontsize_title)

    return fig, ax


def visualize_zero_shot_classification_results(
    images: Sequence[PilImage],
    texts: Sequence[str],
    top_probs: torch.Tensor,
    top_labels: torch.Tensor,
    classes: Sequence[str],
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int] = (16, 16),
) -> None:
    with plt.style.context("ggplot"):
        fig = plt.figure(figsize=figsize)
        y = np.arange(top_probs.shape[-1])

        for i, img in enumerate(images):
            ax1 = fig.add_subplot(nrows, ncols, 2 * i + 1)
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title(texts[i], fontsize=10)

            ax2 = fig.add_subplot(nrows, ncols, 2 * i + 2)
            ax2.barh(y, top_probs[i])

            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            ax2.set_yticks(
                ticks=y,
                labels=[classes[idx] for idx in top_labels[i]],
            )
            ax2.set_xlabel("Probability")

        fig.subplots_adjust(wspace=0.5)


def visualize_multiple_images_with_titles(
    images: Dict[str, PilImage],
    dpi: int = 300,
    fontsize_title: int = 5,
) -> None:
    """Visualize multiple images with their titles.

    Args:
        images (Dict[str, PilImage]): A dictionary of images and their titles.
        dpi (int, optional): The dots per inch of the figure. Defaults to 300.
        fontsize_title (int, optional): The fontsize of the title. Defaults to 5.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(images), dpi=dpi)

    for i, (k, image) in enumerate(images.items()):
        axes[i].imshow(image)
        axes[i].set_title(k, fontsize=fontsize_title)
        axes[i].axis("off")

    fig.tight_layout()
