# pyright: basic

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
from PIL import Image

from consts import DATA_DIR, RNG, TRAIN_DIR


def check_img_dims():
    """
    Checking all image sizes.
    """
    for path, dirnames, filenames in os.walk(DATA_DIR):
        # limit search only to image-containing directories (`dirnames` should be empty list)
        if not dirnames:
            for f in filenames:
                w, h = Image.open(os.path.join(path, f)).size

                # for simplicity sake, only print those not 224 x 224 pixels
                if w != 224 & h != 224:
                    print(f"{path}/{f}: {w} x {h}")


def plot_example_images():
    """
    Plot example images (5 each) of benign and malignant tumors.
    """
    benign_img_paths = [
        os.path.join(TRAIN_DIR, "Benign", f)
        for f in np.random.choice(
            os.listdir(os.path.join(TRAIN_DIR, "Benign")), size=5, replace=False
        )
    ]
    malign_img_paths = [
        os.path.join(TRAIN_DIR, "Malignant", f)
        for f in np.random.choice(
            os.listdir(os.path.join(TRAIN_DIR, "Malignant")), size=5, replace=False
        )
    ]
    img_paths = benign_img_paths + malign_img_paths

    fig, axes = plt.subplots(nrows=2, ncols=5)
    axes = axes.flatten()

    for i, im in enumerate(img_paths):
        axes[i].imshow(Image.open(im))
        axes[i].axis("off")

    fig.suptitle("Benign (top row) and malignant (bottom row) tumors", fontsize=16)
    plt.show()
    fig.savefig("figs/training_set_example.png", dpi=200)


def create_color_histogram_examples(
    n_examples, ignore_black_pixels=False, black_pixel_threshold=20, show=False
):
    """
    Create a handful for color histograms for viewing.
    """
    for n in range(n_examples):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

        for i, tumor_type in enumerate(["Benign", "Malignant"]):
            dirpath = os.path.join(TRAIN_DIR, tumor_type)
            fname = RNG.choice(os.listdir(dirpath))
            img = np.asarray(Image.open(os.path.join(dirpath, fname)))

            # convert grayscale image from [0.0, 1.0] to [0, 255]
            img_gray = skimage.color.rgb2gray(img) * 255
            if ignore_black_pixels:
                img_gray = img_gray[img_gray >= black_pixel_threshold]

            axes[i, 0].imshow(img)
            axes[i, 0].axis("off")
            axes[i, 0].set_title(f"{os.path.join(dirpath, fname)}", fontsize=12)

            for c, color in enumerate(["red", "green", "blue"]):
                channel = img[..., c]
                if ignore_black_pixels:
                    channel = channel[channel >= black_pixel_threshold]

                img_hist, bins = skimage.exposure.histogram(channel)
                axes[i, c + 1].plot(bins, img_hist / img_hist.max(), color=color)
                img_cdf, bins = skimage.exposure.cumulative_distribution(channel)
                axes[i, c + 1].plot(bins, img_cdf, color=color, linestyle="dashed")

            img_gray_hist, bins = skimage.exposure.histogram(img_gray)
            axes[i, -1].plot(bins, img_gray_hist / img_gray_hist.max(), color="gray")
            img_gray_cdf, bins = skimage.exposure.cumulative_distribution(img_gray)
            axes[i, -1].plot(bins, img_gray_cdf, color="gray", linestyle="dashed")

        fig.tight_layout()

        fig_save_path = "figs/color-hists"
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)

        fig.savefig(os.path.join(fig_save_path, f"example_{n}.jpg"), dpi=200)

        if show:
            plt.show()

        plt.close(fig)


def visualize_pairplot(data, **kwargs):
    sns.pairplot(data, hue="tumor", **kwargs)
    plt.show()


def visualize_heatmap(data, **kwargs):
    sns.heatmap(data.corr(), annot=True, **kwargs)
    plt.show()


def main():
    test_img = np.asarray(Image.open(os.path.join(TRAIN_DIR, "Benign", "1000.jpg")))
    # # see if any images are not 224 x 224 pixels
    # check_img_dims()
    #
    # # visualize benign and malignant tumors (5 random samples, each)
    # plot_example_images()

    # make some color histogram examples
    # create_color_histogram_examples(
    #     20, ignore_black_pixels=True, black_pixel_threshold=10
    # )

    # visualize features
    df_from_json = pd.read_json("features.json", orient="index")
    # visualize_pairplot(
    #     df_from_json, **{"vars": ["mean_red", "mean_green", "mean_blue", "mean_gray"]}
    # )
    visualize_heatmap(df_from_json, **{"cmap": "vlag"})


if __name__ == "__main__":
    main()
