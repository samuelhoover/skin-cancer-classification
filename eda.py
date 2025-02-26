# pyright: basic

import os

import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image

from consts import DATA_DIR, TRAIN_DIR, RNG


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


def create_color_histogram_examples(n_examples, show=False):
    """
    Create a handful for color histograms for viewing.
    """
    for n in range(n_examples):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

        for i, tumor_type in enumerate(["Benign", "Malignant"]):
            dirpath = os.path.join(TRAIN_DIR, tumor_type)
            fname = RNG.choice(os.listdir(dirpath))
            img = np.asarray(Image.open(os.path.join(dirpath, fname)))

            axes[i, 0].imshow(img)
            axes[i, 0].axis("off")
            axes[i, 0].set_title(f"{os.path.join(dirpath, fname)}", fontsize=12)

            for c, color in enumerate(["red", "green", "blue"]):
                img_hist, bins = skimage.exposure.histogram(img[..., c])
                axes[i, c + 1].plot(bins, img_hist / img_hist.max(), color=color)
                img_cdf, bins = skimage.exposure.cumulative_distribution(img[..., c])
                axes[i, c + 1].plot(bins, img_cdf, color=color, linestyle="dashed")

            img_gray = skimage.color.rgb2gray(img)
            img_gray_hist, bins = skimage.exposure.histogram(img_gray)
            axes[i, -1].plot(
                bins * 256, img_gray_hist / img_gray_hist.max(), color="gray"
            )
            img_gray_cdf, bins = skimage.exposure.cumulative_distribution(img_gray)
            axes[i, -1].plot(bins * 256, img_gray_cdf, color="gray", linestyle="dashed")

        fig.tight_layout()
        fig.savefig(f"figs/color-hists/example_{n}.jpg", dpi=200)

        if show:
            plt.show()

        plt.close(fig)


def compute_moments(x, Px):
    mean = np.sum(x * Px)
    vari = np.sum(((x - mean) ** 2) * Px)

    z = (x - mean) / np.sqrt(vari)

    skew = np.sum((z**3) * Px)
    kurt = np.sum((z**4) * Px)

    return (mean, vari, skew, kurt)


def main():
    test_img = np.asarray(Image.open(os.path.join(TRAIN_DIR, "Benign", "1000.jpg")))
    # # see if any images are not 224 x 224 pixels
    # check_img_dims()
    #
    # # visualize benign and malignant tumors (5 random samples, each)
    # plot_example_images()

    # make some color histogram examples
    # create_color_histogram_examples(20)

    # red channel moments
    for i in range(3):
        img_hist, bins = skimage.exposure.histogram(test_img[..., i])
        print(compute_moments(bins, img_hist / img_hist.max()))


if __name__ == "__main__":
    main()
