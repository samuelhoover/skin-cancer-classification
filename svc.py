# pyright: basic

import json
import joblib

import scipy
from sklearn import metrics, model_selection, preprocessing, svm

from utils.consts import SEED
from utils.plotting import plot_confusion_matrix, plot_ROC_and_PRC
from utils.processing import create_dataset, create_full_feature_set

###########
### KEY ###
###########
# Benign tumor - 0
# Malignant tumor - 1


def print_performance_metrics(y_true, y_pred):
    metric_str = f"""
    Performance metrics
    ------------------
    Accuracy:          {metrics.accuracy_score(y_true, y_pred)}
    ROC AUC:           {metrics.roc_auc_score(y_true, y_pred)}
    PR AUC:            {metrics.average_precision_score(y_true, y_pred)}
    MCC:               {metrics.matthews_corrcoef(y_true, y_pred)}
    {metrics.classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])}
    """
    print(metric_str)


def train(black_pixel_threshold, n_pc, svd_solver, save_models=False):
    # read in data (index = file path) and split into training and validation sets
    (X, y), pca = create_dataset(
        create_full_feature_set(
            "data/train", black_pixel_threshold=black_pixel_threshold, save_path=None
        ),
        n_pc=n_pc,
        svd_solver=svd_solver,
    )

    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # fit data scaler to train set, transform both training and validation sets
    scaler = preprocessing.RobustScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # define cross-validation methods
    inner_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)

    # want to optimize the complexity of the decision boundary (C and gamma)
    search = model_selection.RandomizedSearchCV(
        svm.SVC(kernel="rbf", random_state=SEED),
        {
            "C": scipy.stats.loguniform(1e-3, 1e2),
            "gamma": scipy.stats.loguniform(1e-3, 1e2),
        },
        n_iter=10,
        scoring=metrics.make_scorer(metrics.matthews_corrcoef),
        refit=True,
        n_jobs=-1,
        cv=inner_cv,
        verbose=3,
        random_state=SEED,
    )
    cv_results = search.fit(X_train_sc, y_train)
    clf = cv_results.best_estimator_
    with open("best_params.json", mode="w") as param_file:
        json.dump(clf.get_params(), param_file, indent=4)

    # nested CV provides a less biased evaluation during model selection, helpful
    # for avoiding overfitting to training dataset when selecting parameters
    nested_score = model_selection.cross_val_score(
        search, X_train_sc, y_train, cv=outer_cv, n_jobs=-1
    ).mean()
    print(f"MCC (nested CV):   {nested_score}")

    # make predictions with tuned SVC, saved misclassified predictions to view
    # later
    y_pred = clf.predict(X_val_sc)
    misclassified = y_pred != y_val
    misclassified_list = [
        {"img_path": img, "pred_class": int(p), "true_class": int(t)}
        for (img, p, t) in zip(
            X_val.index.values[misclassified],
            y_pred[misclassified],
            y_val[misclassified].values,
        )
    ]
    with open("misclassified_examples.json", mode="w") as f:
        json.dump(
            misclassified_list,
            f,
            indent=4,
        )

    print_performance_metrics(y_val, y_pred)
    plot_confusion_matrix(clf, X_val_sc, y_val)
    plot_ROC_and_PRC(clf, X_val_sc, y_val)

    # save scaler, PCA, and SVC models
    if save_models:
        joblib.dump(scaler, "saved-models/robust_scaler.pkl", compress=True)
        joblib.dump(pca, "saved-models/pca.pkl", compress=True)
        joblib.dump(clf, "saved-models/svc.pkl", compress=True)

    return clf, scaler, pca


def test(clf, scaler, pca, black_pixel_threshold, n_pc, svd_solver):
    (X_test, y_test), pca = create_dataset(
        create_full_feature_set(
            "data/test",
            black_pixel_threshold=black_pixel_threshold,
            save_path=None,
        ),
        n_pc=n_pc,
        svd_solver=svd_solver,
        pca=pca,
    )

    X_test_sc = scaler.transform(X_test)

    print_performance_metrics(y_test, clf.predict(X_test_sc))
    plot_confusion_matrix(clf, X_test_sc, y_test)
    plot_ROC_and_PRC(clf, X_test_sc, y_test)


def main(run_train=True, run_test=True, save_models=False, load_models=True):
    black_pixel_threshold = 5
    n_pc = 100
    svd_solver = "randomized"

    if run_train:
        print("Training...")
        clf, scaler, pca = train(
            black_pixel_threshold,
            n_pc=n_pc,
            svd_solver=svd_solver,
            save_models=save_models,
        )
        print("Done!\n")

    if run_test:
        print("Testing...")
        if load_models:
            clf = joblib.load("saved-models/svc.pkl")
            scaler = joblib.load("saved-models/robust_scaler.pkl")
            pca = joblib.load("saved-models/pca.pkl")

        test(clf, scaler, pca, black_pixel_threshold, n_pc, svd_solver)

        print("Done!")


if __name__ == "__main__":
    main(run_train=False, run_test=True, save_models=False, load_models=True)
