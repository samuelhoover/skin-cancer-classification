# pyright: basic

import scipy
import json
from sklearn import metrics, model_selection, preprocessing, svm

from consts import SEED
from plotting import plot_confusion_matrix, plot_ROC_and_PRC
from processing import create_dataset

###########
### KEY ###
###########
# Benign tumor - 0
# Malignant tumor - 1

# read in data (index = file path) and split into training and validation sets
X, y = create_dataset(
    "features/features_threshold_5.csv", n_pc=100, svd_solver="randomized"
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

metric_str = f"""
Validation metrics
------------------
Best parameters:   {cv_results.best_params_}
Accuracy:          {metrics.accuracy_score(y_val, y_pred)}
ROC AUC:           {metrics.roc_auc_score(y_val, y_pred)}
PR AUC:            {metrics.average_precision_score(y_val, y_pred)}
MCC:               {metrics.matthews_corrcoef(y_val, y_pred)}
MCC (nested CV):   {nested_score}
{metrics.classification_report(y_val, y_pred, target_names=["Benign", "Malignant"])}
"""
print(metric_str)

plot_confusion_matrix(clf, X_val_sc, y_val)
plot_ROC_and_PRC(clf, X_val_sc, y_val)
