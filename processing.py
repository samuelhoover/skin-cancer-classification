# pyright: basic
import json
import os
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import skimage
from PIL import Image
from skimage.color import rgb2gray

from consts import TRAIN_DIR

###########
### KEY ###
###########
# Benign tumor - 0
# Malignant tumor - 1


def calculate_moments(x, Px):
    """
    Calculate first, second, third, and fourth moments given data (x) and a
    distribution (Px).
    """
    mean = np.sum(x * Px)
    vari = np.sum(((x - mean) ** 2) * Px)

    z = (x - mean) / np.sqrt(vari)

    skew = np.sum((z**3) * Px)
    kurt = np.sum((z**4) * Px)

    return (mean, vari, skew, kurt)


def group_rbg_gray_img(img_path):
    """
    Concatenate original image with grayscaled image. Multiply grayscale image
    by 255 since `rgb2gray` outputs intensities (0.0 to 1.0) and I want to
    match RGB scale (0 to 255).
    """
    img = np.asarray(Image.open(img_path))
    img_gray = rgb2gray(img)[:, :, np.newaxis] * 255
    grouped_img = np.concat([img, img_gray], axis=-1)

    return grouped_img


def calculate_image_moments(grouped_img, feature_dict, **kwargs):
    for c, color in enumerate(["red", "green", "blue", "gray"]):
        channel = grouped_img[..., c]
        img_hist, bins = skimage.exposure.histogram(
            channel[channel > kwargs["black_pixel_threshold"]]
        )

        norm_img_hist = img_hist / img_hist.max()
        m, v, s, k = calculate_moments(bins, norm_img_hist)

        feature_dict[f"{kwargs['img_path']}"][f"mean_{color}"] = m
        feature_dict[f"{kwargs['img_path']}"][f"vari_{color}"] = v
        feature_dict[f"{kwargs['img_path']}"][f"skew_{color}"] = s
        feature_dict[f"{kwargs['img_path']}"][f"kurt_{color}"] = k

    return feature_dict


def create_features(black_pixel_threshold):
    features = defaultdict(dict)

    for path, dirnames, filenames in os.walk(TRAIN_DIR):
        # limit search only to image-containing directories (`dirnames` should be empty list)
        if not dirnames:
            for f in filenames:
                img_path = os.path.join(path, f)

                # extract targets (0 for benign, 1 for malignant)
                features[f"{img_path}"]["tumor"] = 0 if path.endswith("Benign") else 1

                # create grouped image from original image and grayscaled image
                grouped_img = group_rbg_gray_img(img_path)

                features = calculate_image_moments(
                    grouped_img,
                    features,
                    **{
                        "img_path": img_path,
                        "black_pixel_threshold": black_pixel_threshold,
                    },
                )

    with open("features.json", mode="w") as outfile:
        json.dump(features, outfile, indent=4)


def main():
    create_features(black_pixel_threshold=10)


if __name__ == "__main__":
    main()
