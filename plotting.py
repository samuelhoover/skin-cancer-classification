# pyright: basic

import matplotlib.pyplot as plt
from sklearn import metrics
import json
from PIL import Image
import os


def plot_confusion_matrix(clf, X, y):
    metrics.ConfusionMatrixDisplay.from_estimator(
        clf,
        X,
        y,
        display_labels=["Benign", "Malignant"],
        normalize="true",
    )
    plt.tight_layout()
    plt.show()


def plot_ROC_and_PRC(clf, X, y):
    _, axes = plt.subplots(ncols=2)
    for ax, display, (xlabel, ylabel) in zip(
        axes,
        [metrics.RocCurveDisplay, metrics.PrecisionRecallDisplay],
        [
            ("FPR (Malignant)", "TPR (Malignant)"),
            ("Recall (Malignant)", "Precision (Malignant"),
        ],
    ):
        display.from_estimator(
            clf,
            X,
            y,
            plot_chance_level=True,
            pos_label=1,
            ax=ax,
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_misclassified_examples():
    with open("misclassified_examples.json", mode="r") as f:
        misclassified_dict = json.load(f)

    save_path = "figs/misclassified_examples"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, ex in enumerate(misclassified_dict):
        fig, ax = plt.subplots()
        ax.imshow(Image.open(ex["img_path"]))
        ax.set_title(
            f"Predicted: {ex['pred_class']}\nTrue: {ex['true_class']}", fontsize=16
        )
        plt.savefig(f"figs/misclassified_examples/misclassified_{i}.png")
        plt.close(fig)


def main():
    plot_misclassified_examples()


if __name__ == "__main__":
    main()
