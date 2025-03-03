# pyright: basic
import os

import numpy as np

# import numpy.typing as npt
import pandas as pd
import skimage
from PIL import Image
from skimage.color import rgb2gray
from sklearn import decomposition

from utils.consts import SEED

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


def group_rbg_gray_img(img):
    """
    Concatenate original image with grayscaled image. Multiply grayscale image
    by 255 since `rgb2gray` outputs intensities (0.0 to 1.0) and I want to
    match RGB scale (0 to 255).
    """
    img_gray = rgb2gray(img)[:, :, np.newaxis] * 255
    grouped_img = np.concat([img, img_gray], axis=-1)

    return grouped_img


def calculate_color_moments(grouped_img, **kwargs):
    """
    Calculate the first four moments -- mean (mean), variance (vari), skewness
    (skew), and kurtosis (kurt) -- from the histogram of color values for each
    channel (red, green, blue, and gray).
    """
    color_moment_dict = {}
    for c, color in enumerate(["red", "green", "blue", "gray"]):
        channel = grouped_img[..., c]
        img_hist, bins = skimage.exposure.histogram(
            channel[channel > kwargs["black_pixel_threshold"]]
        )

        norm_img_hist = img_hist / img_hist.max()
        m, v, s, k = calculate_moments(bins, norm_img_hist)

        color_moment_dict[f"mean_{color}"] = m
        color_moment_dict[f"vari_{color}"] = v
        color_moment_dict[f"skew_{color}"] = s
        color_moment_dict[f"kurt_{color}"] = k

    return color_moment_dict


def extract_hog_features(img, orientations, pixels_per_cell, cells_per_block):
    """
    Calculate Histogram of Gradients (HOG) features from image, return 1d
    feature vector.
    """
    return skimage.feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        channel_axis=-1,
        feature_vector=True,
    )


def extract_lbp_features(img, num_points, radius):
    """
    Calculate Linear Binary Patterns (LBP) features from image, return 1d
    feature vector.
    """
    lbp = skimage.feature.local_binary_pattern(
        np.asarray(rgb2gray(img) * 255).astype(np.uint8), num_points, radius, "uniform"
    )
    counts, _ = np.histogram(
        lbp.ravel(),
        density=True,
        bins=int(lbp.max() + 1),
        range=(0, int(lbp.max() + 1)),
    )

    return counts


def create_full_feature_set(data_src, black_pixel_threshold, save_path=None):
    """
    Generate full feature set, pulling from image paths, tumor types,
    color/grayscale histograms (while optionally thresholding images), and
    computing LBP and (full set of) HOG features. Either save or return full
    feature set.
    """
    features_list = []

    for f in (x for x in os.listdir(data_src) if x.endswith(".jpg")):
        img_path = os.path.join(data_src, f)
        img = np.asarray(Image.open(img_path))

        img_dict = {"img_src": img_path}

        # extract targets (0 for benign, 1 for malignant)
        img_dict = img_dict | {"tumor": 0 if f.startswith("Benign") else 1}

        img_dict = img_dict | calculate_color_moments(
            group_rbg_gray_img(img),
            **{
                "img_path": img_path,
                "black_pixel_threshold": black_pixel_threshold,
            },
        )

        orientations = 8
        pixels_per_cell = (16, 16)
        cells_per_block = (2, 2)
        img_dict = img_dict | {
            f"hog_pca_{i}": v
            for i, v in enumerate(
                extract_hog_features(
                    img,
                    orientations,
                    pixels_per_cell,
                    cells_per_block,
                )
            )
        }

        radius = 8
        n_points = 3 * radius
        img_dict = img_dict | {
            f"lbp_{i}": v
            for i, v in enumerate(extract_lbp_features(img, n_points, radius))
        }

        features_list.append(img_dict)

    df = pd.DataFrame(features_list)
    df = df.set_index("img_src")

    if save_path is None:
        return df
    else:
        df.to_csv(save_path)


def calculate_pc_from_hog(X_hog, n_pc, svd_solver="auto", pca=None):
    """
    Reduce number of HOG features using PCA.
    """
    if pca is None:
        pca = decomposition.PCA(
            n_components=n_pc, svd_solver=svd_solver, random_state=SEED
        )
        pca.fit(X_hog)

    return pca.transform(X_hog), pca


def create_dataset(features, n_pc, svd_solver="auto", pca=None):
    """
    Create training/testing dataset by compressing HOG features using PCA.
    """
    if isinstance(features, pd.DataFrame):
        df = features
    else:
        try:
            df = pd.read_csv(features, index_col="img_src")
        except FileNotFoundError as err:
            print(f"{err}: {features} does not exist and is not a pandas DataFrame")

    y = df.pop("tumor")
    X = df

    hog_cols = [col for col in X.columns if col.startswith("hog")]
    hog_pca, pca = calculate_pc_from_hog(
        X.loc[:, hog_cols], n_pc=n_pc, svd_solver=svd_solver, pca=pca
    )

    X.drop(hog_cols, axis=1, inplace=True)
    X_reduced = pd.concat(
        [
            X,
            pd.DataFrame(
                hog_pca,
                index=X.index,
                columns=[f"hog_pca_{i}" for i in range(n_pc)],
            ),
        ],
        axis=1,
        join="inner",
    )

    return (X_reduced, y), pca


def main():
    # data_src = os.path.join(TRAIN_DIR)
    # threshold = 5
    # create_full_feature_set(
    #     data_src=data_src,
    #     black_pixel_threshold=threshold,
    #     save_path=f"data/features_threshold_{threshold}.csv",
    # )
    # load_features_into_dataframe("features.json")
    # calculate_pc_from_hog(
    #     pd.read_csv("data/features_threshold_5.csv"),
    #     n_pc=100,
    #     svd_solver="auto",
    # )
    # create_dataset("data/features_threshold_5.csv", n_pc=5)


if __name__ == "__main__":
    main()
