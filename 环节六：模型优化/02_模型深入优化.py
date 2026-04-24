import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, f1_score, roc_curve,
)
from scipy.stats import uniform, randint, loguniform
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_NPZ = os.path.join(SCRIPT_DIR, "feature_engineering_data.npz")
FEAT_NAMES_FILE = os.path.join(SCRIPT_DIR, "feature_names.txt")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
REPORT_FILE = os.path.join(SCRIPT_DIR, "result.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = []
MODEL_NAMES = {
    "MLP": "MLP", "RF": "RandomForest", "HGBT": "HGBT",
    "SVM": "SVM", "ET": "ExtraTrees", "KNN": "KNN",
    "Stacking_v2": "Stacking_v2", "Voting_v2": "Voting_v2"
}

def log(msg):
    print(msg)
    report.append(str(msg))

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def generate_test_data():
    np.random.seed(42)
    n_train, n_val, n_test = 500, 100, 100
    n_feat = 2105
    X_train = np.random.randn(n_train, n_feat)
    X_val = np.random.randn(n_val, n_feat)
    X_test = np.random.randn(n_test, n_feat)

    y_reg_train = np.random.uniform(5, 10, n_train)
    y_reg_val = np.random.uniform(5, 10, n_val)
    y_reg_test = np.random.uniform(5, 10, n_test)

    y_cls_train = (y_reg_train > 7.5).astype(int)
    y_cls_val = (y_reg_val > 7.5).astype(int)
    y_cls_test = (y_reg_test > 7.5).astype(int)

    np.savez(INPUT_NPZ,
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_reg_train=y_reg_train, y_reg_val=y_reg_val, y_reg_test=y_reg_test,
             y_cls_train=y_cls_train, y_cls_val=y_cls_val, y_cls_test=y_cls_test,
             n_desc=1000, n_fp=1105)
    with open(FEAT_NAMES_FILE, "w") as f:
        f.write("\n".join([f"d_{i}" for i in range(1000)] + [f"m_{i}" for i in range(1105)]))

if not os.path.exists(INPUT_NPZ):
    generate_test_data()

data = np.load(INPUT_NPZ)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_reg_train, y_reg_val, y_reg_test = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]
y_cls_train, y_cls_val, y_cls_test = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]

X_trainval = np.vstack([X_train, X_val])
y_reg_trainval = np.concatenate([y_reg_train, y_reg_val])
y_cls_trainval = np.concatenate([y_cls_train, y_cls_val])

mi_scores = mutual_info_regression(X_trainval, y_reg_trainval, random_state=42, n_neighbors=5)
mi_order = np.argsort(mi_scores)[::-1]
best_k = 1000
selected_idx = mi_order[:best_k]
X_trainval_sel = X_trainval[:, selected_idx]
X_test_sel = X_test[:, selected_idx]

log("=== RF Tuning ===")
rf = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=2, random_state=42),
    {"n_estimators": randint(100, 300), "max_depth": [None, 10, 20],
     "min_samples_split": randint(2, 8), "max_features": ["sqrt", 0.3]},
    n_iter=10, cv=3, scoring="r2", random_state=42, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval).best_estimator_

log("=== HGBT Tuning ===")
hgbt = RandomizedSearchCV(
    HistGradientBoostingRegressor(early_stopping=True, random_state=42),
    {"max_iter": randint(100, 300), "max_depth": randint(3, 8),
     "learning_rate": loguniform(0.01, 0.2)},
    n_iter=10, cv=3, scoring="r2", random_state=42, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval).best_estimator_

log("=== SVM Tuning ===")
svm = RandomizedSearchCV(
    SVR(), {"C": loguniform(1, 50), "epsilon": loguniform(0.01, 0.5)},
    n_iter=8, cv=3, scoring="r2", random_state=42, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval).best_estimator_

log("=== MLP Tuning ===")
mlp = RandomizedSearchCV(
    MLPRegressor(early_stopping=True, random_state=42),
    {"hidden_layer_sizes": [(128, 64), (256, 128)],
     "alpha": loguniform(1e-4, 1e-2), "batch_size": [32, 64]},
    n_iter=5, cv=3, scoring="r2", random_state=42, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval).best_estimator_

log("=== ET Tuning ===")
et = RandomizedSearchCV(
    ExtraTreesRegressor(n_jobs=2, random_state=42),
    {"n_estimators": randint(100, 300), "max_features": ["sqrt"]},
    n_iter=8, cv=3, scoring="r2", random_state=42, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval).best_estimator_

log("=== KNN Tuning ===")
knn = RandomizedSearchCV(
    KNeighborsRegressor(n_jobs=2),
    {"n_neighbors": randint(5, 15), "weights": ["uniform", "distance"]},
    n_iter=8, cv=3, scoring="r2", random_state=42, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval).best_estimator_

reg_models = {"MLP": mlp, "RF": rf, "HGBT": hgbt, "SVM": svm, "ET": et, "KNN": knn}
reg_res = {}
for n, m in reg_models.items():
    p = m.predict(X_test_sel)
    reg_res[n] = (r2_score(y_reg_test, p), np.sqrt(mean_squared_error(y_reg_test, p)))
    log(f"{n:10s} R2={reg_res[n][0]:.4f} RMSE={reg_res[n][1]:.4f}")

cls_models = {}
for n, m in reg_models.items():
    if n == "MLP":
        cls = MLPClassifier(**m.get_params(), random_state=42)
    elif n == "RF":
        cls = RandomForestClassifier(**m.get_params(), class_weight="balanced", n_jobs=2, random_state=42)
    elif n == "HGBT":
        cls = HistGradientBoostingClassifier(**m.get_params(), class_weight="balanced", random_state=42)
    elif n == "SVM":
        cls = SVC(**m.get_params(), probability=True, class_weight="balanced", random_state=42)
    elif n == "ET":
        cls = ExtraTreesClassifier(**m.get_params(), class_weight="balanced", n_jobs=2, random_state=42)
    elif n == "KNN":
        cls = KNeighborsClassifier(**m.get_params(), n_jobs=2)
    else:
        continue
    cls.fit(X_trainval_sel, y_cls_trainval)
    cls_models[n] = cls

cls_res = {}
for n, m in cls_models.items():
    prob = m.predict_proba(X_test_sel)[:, 1]
    cls_res[n] = (roc_auc_score(y_cls_test, prob), accuracy_score(y_cls_test, m.predict(X_test_sel)))
    log(f"{n:10s} AUC={cls_res[n][0]:.4f} Acc={cls_res[n][1]:.4f}")

log("=== Stacking Regression ===")
stack_reg = StackingRegressor(
    estimators=[(k, v) for k, v in reg_models.items()],
    final_estimator=RidgeCV(), cv=3, n_jobs=2
).fit(X_trainval_sel, y_reg_trainval)
p_stack = stack_reg.predict(X_test_sel)
r2_stack = r2_score(y_reg_test, p_stack)
rmse_stack = np.sqrt(mean_squared_error(y_reg_test, p_stack))
reg_res["Stacking_v2"] = (r2_stack, rmse_stack)
log(f"Stacking R2={r2_stack:.4f} RMSE={rmse_stack:.4f}")

log("=== Voting Regression ===")
vote_reg = VotingRegressor(estimators=[(k, v) for k, v in reg_models.items()], n_jobs=2
).fit(X_trainval_sel, y_reg_trainval)
p_vote = vote_reg.predict(X_test_sel)
r2_vote = r2_score(y_reg_test, p_vote)
rmse_vote = np.sqrt(mean_squared_error(y_reg_test, p_vote))
reg_res["Voting_v2"] = (r2_vote, rmse_vote)
log(f"Voting   R2={r2_vote:.4f} RMSE={rmse_vote:.4f}")

log("=== Stacking Classification ===")
stack_cls = StackingClassifier(
    estimators=[(k, v) for k, v in cls_models.items()],
    final_estimator=LogisticRegressionCV(class_weight="balanced", random_state=42),
    stack_method="predict_proba", cv=3, n_jobs=2
).fit(X_trainval_sel, y_cls_trainval)
prob_stack = stack_cls.predict_proba(X_test_sel)[:,1]
auc_stack = roc_auc_score(y_cls_test, prob_stack)
acc_stack = accuracy_score(y_cls_test, stack_cls.predict(X_test_sel))
cls_res["Stacking_v2"] = (auc_stack, acc_stack)
log(f"Stacking AUC={auc_stack:.4f} Acc={acc_stack:.4f}")

log("=== Voting Classification ===")
vote_cls = VotingClassifier(
    estimators=[(k, v) for k, v in cls_models.items()],
    voting="soft", n_jobs=2
).fit(X_trainval_sel, y_cls_trainval)
prob_vote = vote_cls.predict_proba(X_test_sel)[:,1]
auc_vote = roc_auc_score(y_cls_test, prob_vote)
acc_vote = accuracy_score(y_cls_test, vote_cls.predict(X_test_sel))
cls_res["Voting_v2"] = (auc_vote, acc_vote)
log(f"Voting   AUC={auc_vote:.4f} Acc={acc_vote:.4f}")

fig, ax = plt.subplots(figsize=(10,5))
names = list(reg_res.keys())
r2s = [reg_res[n][0] for n in names]
ax.bar(names, r2s, color='steelblue')
ax.set_title("Test R2 Score")
ax.tick_params(axis='x', rotation=45)
save_fig(fig, "regression_r2.png")

fig, ax = plt.subplots(figsize=(10,5))
aucs = [cls_res[n][0] for n in cls_res.keys()]
ax.bar(list(cls_res.keys()), aucs, color='crimson')
ax.set_title("Test AUC Score")
ax.tick_params(axis='x', rotation=45)
save_fig(fig, "classification_auc.png")

fig, ax = plt.subplots(figsize=(8,8))
for n in cls_res:
    if n in cls_models:
        fpr, tpr, _ = roc_curve(y_cls_test, cls_models[n].predict_proba(X_test_sel)[:,1])
        ax.plot(fpr, tpr, label=f"{n} AUC={cls_res[n][0]:.3f}")
ax.plot([0,1],[0,1],'k--')
ax.set_title("ROC Curve")
ax.legend()
save_fig(fig, "roc_curve.png")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n✅ ALL DONE. Figures saved in /figures, result.txt saved.")
