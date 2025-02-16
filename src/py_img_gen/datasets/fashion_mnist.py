from typing import Tuple


def get_fashion_mnist_classes() -> Tuple[str, ...]:
    """Get the classes of the Fashion MNIST dataset.

    The labels defined here are based on the following URL:
    https://github.com/zalandoresearch/fashion-mnist/tree/master#labels

    Returns:
        Tuple[str]: The classes of the Fashion MNIST dataset.

    """
    return (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    )
