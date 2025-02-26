# pyright: basic

import pandas as pd
from sklearn import metrics, model_selection, svm, preprocessing
import matplotlib.pyplot as plt
import scipy

from consts import SEED

###########
### KEY ###
###########
# Benign tumor - 0
# Malignant tumor - 1

df_from_json = pd.read_json("features.json", orient="index")
y = df_from_json.pop("tumor")
X = df_from_json.to_numpy()
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

scaler = preprocessing.RobustScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

clf = model_selection.RandomizedSearchCV(
    svm.SVC(kernel="rbf", random_state=SEED),
    {
        "C": scipy.stats.loguniform(1, 1e5),
        "gamma": scipy.stats.loguniform(1e-3, 1e-1),
    },
    n_iter=10,
    scoring="accuracy",
    refit=True,
    n_jobs=-1,
    verbose=3,
    random_state=SEED,
)
clf = clf.fit(X_train_sc, y_train)

y_pred = clf.predict(X_test_sc)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"ROC AUC:  {metrics.roc_auc_score(y_test, y_pred)}")
print(
    metrics.classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
)

metrics.ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test_sc,
    y_test,
    display_labels=["Benign", "Malignant"],
    normalize="true",
)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
metrics.RocCurveDisplay.from_estimator(
    clf,
    X_test_sc,
    y_test,
    plot_chance_level=True,
    ax=ax,
)
ax.set_xlabel("False positive rate (Positive label: 1)", fontsize=16)
ax.set_ylabel("True positive rate (Positive label: 1)", fontsize=16)
ax.legend(fontsize=14)

plt.tight_layout()
plt.show()
