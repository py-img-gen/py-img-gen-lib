from torchvision import transforms


def get_simple_resize_transforms(sample_size: int) -> transforms.Compose:
    r"""Get the simple resize transforms for the dataset.

    Args:
        sample_size (int): The size of the sample.
    Returns:
        torchvision.transforms.Compose: The composed transforms.
    """
    transforms_list = [
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transforms_list)
