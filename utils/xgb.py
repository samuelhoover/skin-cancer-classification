# pyright: basic

from sklearn import metrics, model_selection, ensemble
import matplotlib.pyplot as plt
import scipy

from consts import SEED
from processing import create_dataset

###########
### KEY ###
###########
# Benign tumor - 0
# Malignant tumor - 1

# read in data (index = file path) and split into training and validation sets
X, y = create_dataset(
    "data/features_threshold_5.csv", n_pc=100, svd_solver="randomized"
)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# define cross-validation methods
inner_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)
outer_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)

search = model_selection.RandomizedSearchCV(
    ensemble.GradientBoostingClassifier(random_state=SEED),
    {
        "n_estimators": scipy.stats.randint(1e1, 1e3),
        "subsample": scipy.stats.uniform(),
    },
    n_iter=10,
    scoring=metrics.make_scorer(metrics.matthews_corrcoef),
    refit=True,
    n_jobs=-1,
    cv=inner_cv,
    verbose=3,
    random_state=SEED,
)
cv_results = search.fit(X_train, y_train)

clf = cv_results.best_estimator_
nested_score = model_selection.cross_val_score(
    clf, X_train, y_train, scoring="roc_auc", cv=outer_cv, n_jobs=-1
).mean()
y_pred = clf.predict(X_val)

metric_str = f"""
Validation metrics
------------------
Best parameters:   {cv_results.best_params_}
Accuracy:          {metrics.accuracy_score(y_val, y_pred)}
Accuracy (train):  {metrics.accuracy_score(y_train, clf.predict(X_train))}
ROC AUC:           {metrics.roc_auc_score(y_val, y_pred)}
Nested CV ROC AUC: {nested_score}
PR AUC:            {metrics.average_precision_score(y_val, y_pred)}
MCC:               {metrics.matthews_corrcoef(y_val, y_pred)}
{metrics.classification_report(y_val, y_pred, target_names=["Benign", "Malignant"])}
"""
print(metric_str)

metrics.ConfusionMatrixDisplay.from_estimator(
    clf,
    X_val,
    y_val,
    display_labels=["Benign", "Malignant"],
    normalize="true",
)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=2)
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
        X_val,
        y_val,
        plot_chance_level=True,
        pos_label=1,
        ax=ax,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
