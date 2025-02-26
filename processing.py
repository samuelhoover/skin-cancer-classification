# pyright: basic
import json
import os
from collections import defaultdict

import numpy as np
import skimage
from PIL import Image

from consts import TRAIN_DIR

###########
### KEY ###
###########
# Benign tumor - 0
# Malignant tumor - 1


def compute_moments(x, Px):
    mean = np.sum(x * Px)
    vari = np.sum(((x - mean) ** 2) * Px)

    z = (x - mean) / np.sqrt(vari)

    skew = np.sum((z**3) * Px)
    kurt = np.sum((z**4) * Px)

    return (mean, vari, skew, kurt)


def create_dataset():
    features = defaultdict(dict)

    for path, dirnames, filenames in os.walk(TRAIN_DIR):
        # limit search only to image-containing directories (`dirnames` should be empty list)
        if not dirnames:
            for i, f in enumerate(filenames):
                if i >= 10:
                    break

                features[f"{os.path.join(path, f)}"]["tumor"] = (
                    0 if path.endswith("Benign") else 1
                )

                img = np.asarray(Image.open(os.path.join(path, f)))

                for c, color in enumerate(["red", "green", "blue"]):
                    img_hist, bins = skimage.exposure.histogram(img[..., c])

                    norm_img_hist = img_hist / img_hist.max()
                    m, v, s, k = compute_moments(bins, norm_img_hist)

                    features[f"{os.path.join(path, f)}"][f"mean_{color}"] = m
                    features[f"{os.path.join(path, f)}"][f"vari_{color}"] = v
                    features[f"{os.path.join(path, f)}"][f"skew_{color}"] = s
                    features[f"{os.path.join(path, f)}"][f"kurt_{color}"] = k

    with open("features.json", mode="w") as outfile:
        json.dump(features, outfile, indent=4)


def main():
    create_dataset()


if __name__ == "__main__":
    main()
