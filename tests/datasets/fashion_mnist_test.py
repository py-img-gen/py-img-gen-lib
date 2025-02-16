from py_img_gen.datasets import get_fashion_mnist_classes


def test_get_fashion_mnist_classes():
    classes = get_fashion_mnist_classes()

    assert len(classes) == 10
    assert classes == (
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
