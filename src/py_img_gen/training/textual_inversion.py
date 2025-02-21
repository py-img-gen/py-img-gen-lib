import logging
import pathlib
from typing import Final, Tuple

import torch
from transformers import CLIPTextModel

logger = logging.getLogger(__name__)

IMAGENET_TEMPLATES_SMALL: Final[Tuple[str, ...]] = (
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
)

IMAGENET_STYLE_TEMPLATES_SMALL: Final[Tuple[str, ...]] = (
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
)


def save_concept_embedding(
    text_encoder: CLIPTextModel,
    placeholder_token_id: int,
    accelerator,
    placeholder_token: str,
    save_path: pathlib.Path,
) -> None:
    # 新たに追加した概念に対応する埋め込みベクトルのみを保存する
    # `placeholder_token` の ID を指定することで対象のベクトルを取得可能
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[placeholder_token_id]
    )
    learned_embeds_dict = {placeholder_token: learned_embeds.clone().detach().cpu()}

    logger.info(f"Saving the learned embeddings to {save_path}")
    torch.save(learned_embeds_dict, save_path)
