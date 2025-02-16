from py_img_gen.datasets.skimage import (
    get_skimage_descriptions,
)


def test_get_skimage_descriptions():
    descriptions = get_skimage_descriptions()
    assert len(descriptions) == 8
    assert list(descriptions.keys()) == [
        "page",
        "chelsea",
        "astronaut",
        "rocket",
        "motorcycle_right",
        "camera",
        "horse",
        "coffee",
    ]
    assert list(descriptions.values()) == [
        "a page of text about segmentation",
        "a facial photo of a tabby cat",
        "a portrait of an astronaut with the American flag",
        "a rocket standing on a launchpad",
        "a red motorcycle standing in a garage",
        "a person looking at a camera on a tripod",
        "a black-and-white silhouette of a horse",
        "a cup of coffee on a saucer",
    ]
