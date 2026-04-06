"""
环节六：模型优化与高级集成
===========================
在环节四/五基础上进行系统优化:
  1) 特征选择 — 基于 Mutual Information 筛选高信息量特征
  2) 超参数调优 — RandomizedSearchCV 搜索最佳参数
  3) 增加新型基学习器 — ExtraTrees, KNN, BaggingSVM
  4) 两层 Stacking — 第一层6个异质模型, 第二层元学习器
  5) 全面对比: 优化前 vs 优化后
"""

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
    BaggingRegressor, BaggingClassifier,
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.feature_selection import (
    mutual_info_regression, SelectKBest, f_regression,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve,
)
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint, loguniform
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_NPZ = os.path.join(PROJECT_DIR, "环节三：特征工程", "特征工程后数据.npz")
FEAT_NAMES_FILE = os.path.join(PROJECT_DIR, "环节三：特征工程", "feature_names.txt")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
REPORT_FILE = os.path.join(SCRIPT_DIR, "优化报告.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = []

COLORS = {
    "MLP": "#3498db", "RF": "#2ecc71", "HGBT": "#e67e22", "SVM": "#9b59b6",
    "ET": "#f39c12", "KNN": "#1abc9c", "BagSVM": "#e74c3c",
    "Stacking_v1": "#c0392b", "Stacking_v2": "#8e44ad",
    "Voting_v1": "#16a085", "Voting_v2": "#2c3e50",
}
MODEL_NAMES = {
    "MLP": "MLP 神经网络", "RF": "Random Forest", "HGBT": "HistGradientBoosting",
    "SVM": "SVM (RBF)", "ET": "Extra Trees", "KNN": "KNN", "BagSVM": "BaggingSVM",
    "Stacking_v1": "Stacking v1 (原)", "Stacking_v2": "Stacking v2 (优化)",
    "Voting_v1": "Voting v1 (原)", "Voting_v2": "Voting v2 (优化)",
}


def log(msg=""):
    print(msg)
    report.append(msg)


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════
#  读取数据
# ══════════════════════════════════════════
log("=" * 65)
log("环节六：模型优化与高级集成")
log("=" * 65)

data = np.load(INPUT_NPZ)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_reg_train, y_reg_val, y_reg_test = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]
y_cls_train, y_cls_val, y_cls_test = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]
n_desc, n_fp = int(data["n_desc"]), int(data["n_fp"])

X_trainval = np.vstack([X_train, X_val])
y_reg_trainval = np.concatenate([y_reg_train, y_reg_val])
y_cls_trainval = np.concatenate([y_cls_train, y_cls_val])

with open(FEAT_NAMES_FILE, "r") as f:
    feature_names = [line.strip() for line in f]

log(f"  Train+Val: {X_trainval.shape},  Test: {X_test.shape}")
log(f"  Features: {X_trainval.shape[1]} (描述符 {n_desc} + 指纹 {n_fp})")

# ══════════════════════════════════════════
#  Part A: 特征选择 — Mutual Information
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part A: 特征选择 (Mutual Information)")
log("=" * 65)

t0 = time.time()
mi_scores = mutual_info_regression(X_trainval, y_reg_trainval, random_state=42, n_neighbors=5)
dt = time.time() - t0
log(f"  MI 计算完成 ({dt:.1f}s)")

# 按 MI 排序
mi_order = np.argsort(mi_scores)[::-1]
mi_sorted = mi_scores[mi_order]

# 找到 MI > 0 的特征
n_nonzero = np.sum(mi_scores > 0.001)
log(f"  MI > 0.001 的特征: {n_nonzero}/{len(mi_scores)}")

# 尝试不同 k 值
ks = [200, 500, 800, 1000, 1500, 2105]
log(f"\n  不同 k 值快速评估 (RF 5-fold CV R²):")

best_k = 2105
best_cv_r2 = -1

for k in ks:
    top_k_idx = mi_order[:k]
    X_sel = X_trainval[:, top_k_idx]
    rf_quick = RandomForestRegressor(n_estimators=100, max_features="sqrt",
                                      n_jobs=-1, random_state=42)
    scores = cross_val_score(rf_quick, X_sel, y_reg_trainval, cv=5,
                              scoring="r2", n_jobs=-1)
    mean_r2 = scores.mean()
    log(f"    k={k:5d}: CV R² = {mean_r2:.4f} ± {scores.std():.4f}")
    if mean_r2 > best_cv_r2:
        best_cv_r2 = mean_r2
        best_k = k

log(f"\n  ★ 最佳 k = {best_k} (CV R² = {best_cv_r2:.4f})")

# 应用特征选择
selected_idx = mi_order[:best_k]
X_trainval_sel = X_trainval[:, selected_idx]
X_test_sel = X_test[:, selected_idx]
selected_names = [feature_names[i] for i in selected_idx]
n_desc_sel = sum(1 for n in selected_names if not n.startswith("Morgan_"))
n_fp_sel = sum(1 for n in selected_names if n.startswith("Morgan_"))
log(f"  选中特征: {best_k} (描述符 {n_desc_sel} + 指纹 {n_fp_sel})")

# Top 20 特征
log(f"\n  MI Top 20 特征:")
for i in range(min(20, len(mi_order))):
    idx = mi_order[i]
    log(f"    {i+1:3d}. {feature_names[idx]:30s}  MI = {mi_scores[idx]:.4f}")

# ══════════════════════════════════════════
#  Part B: 超参数调优
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part B: 超参数调优 (RandomizedSearchCV)")
log("=" * 65)

# --- RF 调优 ---
log("\n  [1/4] Random Forest 调优...")
t0 = time.time()
rf_param_dist = {
    "n_estimators": randint(200, 800),
    "max_depth": [None, 15, 20, 30, 40],
    "min_samples_split": randint(2, 12),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=42),
    rf_param_dist, n_iter=30, cv=5, scoring="r2",
    random_state=42, n_jobs=-1, verbose=0,
)
rf_search.fit(X_trainval_sel, y_reg_trainval)
dt = time.time() - t0
log(f"    最佳 CV R²: {rf_search.best_score_:.4f} ({dt:.1f}s)")
log(f"    最佳参数: {rf_search.best_params_}")
best_rf_reg = rf_search.best_estimator_

# --- HGBT 调优 ---
log("\n  [2/4] HistGradientBoosting 调优...")
t0 = time.time()
hgbt_param_dist = {
    "max_iter": randint(200, 800),
    "max_depth": randint(3, 10),
    "learning_rate": loguniform(0.01, 0.3),
    "min_samples_leaf": randint(5, 30),
    "l2_regularization": loguniform(0.1, 10),
    "max_bins": [128, 255],
}
hgbt_search = RandomizedSearchCV(
    HistGradientBoostingRegressor(early_stopping=True, random_state=42),
    hgbt_param_dist, n_iter=30, cv=5, scoring="r2",
    random_state=42, n_jobs=-1, verbose=0,
)
hgbt_search.fit(X_trainval_sel, y_reg_trainval)
dt = time.time() - t0
log(f"    最佳 CV R²: {hgbt_search.best_score_:.4f} ({dt:.1f}s)")
log(f"    最佳参数: {hgbt_search.best_params_}")
best_hgbt_reg = hgbt_search.best_estimator_

# --- SVM 调优 ---
log("\n  [3/4] SVM 调优...")
t0 = time.time()
svm_param_dist = {
    "C": loguniform(1, 100),
    "epsilon": loguniform(0.01, 1.0),
    "gamma": ["scale", "auto"],
}
svm_search = RandomizedSearchCV(
    SVR(kernel="rbf"),
    svm_param_dist, n_iter=20, cv=5, scoring="r2",
    random_state=42, n_jobs=-1, verbose=0,
)
svm_search.fit(X_trainval_sel, y_reg_trainval)
dt = time.time() - t0
log(f"    最佳 CV R²: {svm_search.best_score_:.4f} ({dt:.1f}s)")
log(f"    最佳参数: {svm_search.best_params_}")
best_svm_reg = svm_search.best_estimator_

# --- MLP 调优 ---
log("\n  [4/4] MLP 调优...")
t0 = time.time()
mlp_param_dist = {
    "hidden_layer_sizes": [(256, 128), (512, 256, 128), (256, 128, 64),
                            (512, 256), (128, 64, 32), (384, 192, 96)],
    "alpha": loguniform(1e-5, 1e-2),
    "learning_rate_init": loguniform(5e-4, 5e-3),
    "batch_size": [32, 64, 128],
}
mlp_search = RandomizedSearchCV(
    MLPRegressor(activation="relu", solver="adam", learning_rate="adaptive",
                  max_iter=500, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=42, verbose=False),
    mlp_param_dist, n_iter=15, cv=5, scoring="r2",
    random_state=42, n_jobs=-1, verbose=0,
)
mlp_search.fit(X_trainval_sel, y_reg_trainval)
dt = time.time() - t0
log(f"    最佳 CV R²: {mlp_search.best_score_:.4f} ({dt:.1f}s)")
log(f"    最佳参数: {mlp_search.best_params_}")
best_mlp_reg = mlp_search.best_estimator_

# ══════════════════════════════════════════
#  Part C: 新增基学习器
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part C: 训练新增基学习器")
log("=" * 65)

# --- Extra Trees ---
log("\n  [1/2] Extra Trees...")
t0 = time.time()
et_param_dist = {
    "n_estimators": randint(300, 800),
    "max_depth": [None, 20, 30],
    "min_samples_split": randint(2, 8),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", 0.3],
}
et_search = RandomizedSearchCV(
    ExtraTreesRegressor(n_jobs=-1, random_state=42),
    et_param_dist, n_iter=20, cv=5, scoring="r2",
    random_state=42, n_jobs=-1, verbose=0,
)
et_search.fit(X_trainval_sel, y_reg_trainval)
dt = time.time() - t0
log(f"    最佳 CV R²: {et_search.best_score_:.4f} ({dt:.1f}s)")
best_et_reg = et_search.best_estimator_

# --- KNN ---
log("\n  [2/2] KNN...")
t0 = time.time()
knn_param_dist = {
    "n_neighbors": randint(3, 25),
    "weights": ["uniform", "distance"],
    "metric": ["minkowski", "manhattan"],
    "p": [1, 2],
}
knn_search = RandomizedSearchCV(
    KNeighborsRegressor(n_jobs=-1),
    knn_param_dist, n_iter=20, cv=5, scoring="r2",
    random_state=42, n_jobs=-1, verbose=0,
)
knn_search.fit(X_trainval_sel, y_reg_trainval)
dt = time.time() - t0
log(f"    最佳 CV R²: {knn_search.best_score_:.4f} ({dt:.1f}s)")
best_knn_reg = knn_search.best_estimator_

# ══════════════════════════════════════════
#  Part D: 评估所有单模型 (优化后)
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part D: 优化后单模型测试集评估")
log("=" * 65)

tuned_reg_models = {
    "MLP": best_mlp_reg,
    "RF": best_rf_reg,
    "HGBT": best_hgbt_reg,
    "SVM": best_svm_reg,
    "ET": best_et_reg,
    "KNN": best_knn_reg,
}

reg_results = {}
for name, model in tuned_reg_models.items():
    y_pred = model.predict(X_test_sel)
    r2 = r2_score(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    reg_results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred}
    log(f"  {MODEL_NAMES[name]:25s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ══════════════════════════════════════════
#  Part E: 训练优化后分类模型
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part E: 训练分类模型 (复用回归调优参数)")
log("=" * 65)

# 从回归最佳参数构造分类模型
best_rf_params = rf_search.best_params_
best_hgbt_params = hgbt_search.best_params_
best_mlp_params = mlp_search.best_params_

cls_models = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=best_mlp_params["hidden_layer_sizes"],
        alpha=best_mlp_params["alpha"],
        learning_rate_init=best_mlp_params["learning_rate_init"],
        batch_size=best_mlp_params["batch_size"],
        activation="relu", solver="adam", learning_rate="adaptive",
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42, verbose=False),
    "RF": RandomForestClassifier(
        n_estimators=best_rf_params["n_estimators"],
        max_depth=best_rf_params["max_depth"],
        min_samples_split=best_rf_params["min_samples_split"],
        min_samples_leaf=best_rf_params["min_samples_leaf"],
        max_features=best_rf_params["max_features"],
        class_weight="balanced", n_jobs=-1, random_state=42),
    "HGBT": HistGradientBoostingClassifier(
        max_iter=best_hgbt_params["max_iter"],
        max_depth=best_hgbt_params["max_depth"],
        learning_rate=best_hgbt_params["learning_rate"],
        min_samples_leaf=best_hgbt_params["min_samples_leaf"],
        l2_regularization=best_hgbt_params["l2_regularization"],
        early_stopping=True, class_weight="balanced", random_state=42),
    "SVM": SVC(
        kernel="rbf", C=svm_search.best_params_["C"],
        gamma=svm_search.best_params_["gamma"],
        class_weight="balanced", probability=True, random_state=42),
    "ET": ExtraTreesClassifier(
        n_estimators=et_search.best_params_["n_estimators"],
        max_depth=et_search.best_params_["max_depth"],
        min_samples_split=et_search.best_params_["min_samples_split"],
        min_samples_leaf=et_search.best_params_["min_samples_leaf"],
        max_features=et_search.best_params_["max_features"],
        class_weight="balanced", n_jobs=-1, random_state=42),
    "KNN": KNeighborsClassifier(
        n_neighbors=knn_search.best_params_["n_neighbors"],
        weights=knn_search.best_params_["weights"],
        metric=knn_search.best_params_["metric"],
        n_jobs=-1),
}

cls_results = {}
for name, model in cls_models.items():
    model.fit(X_trainval_sel, y_cls_trainval)
    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]
    auc = roc_auc_score(y_cls_test, y_prob)
    acc = accuracy_score(y_cls_test, y_pred)
    f1 = f1_score(y_cls_test, y_pred)
    prec = precision_score(y_cls_test, y_pred)
    rec = recall_score(y_cls_test, y_pred)
    cls_results[name] = {"AUC": auc, "Acc": acc, "F1": f1,
                          "Prec": prec, "Rec": rec,
                          "y_pred": y_pred, "y_prob": y_prob}
    log(f"  {MODEL_NAMES[name]:25s}  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")


# ══════════════════════════════════════════
#  Part F: 高级集成 — 两层 Stacking
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part F: 高级集成")
log("=" * 65)

# --- 优化 Stacking 回归 (6个基学习器 + passthrough) ---
log("\n  [1/4] Stacking v2 回归 (6基学习器)...")
t0 = time.time()

# 克隆参数构造基学习器
reg_estimators_v2 = [
    ("MLP", MLPRegressor(
        hidden_layer_sizes=best_mlp_params["hidden_layer_sizes"],
        alpha=best_mlp_params["alpha"],
        learning_rate_init=best_mlp_params["learning_rate_init"],
        batch_size=best_mlp_params["batch_size"],
        activation="relu", solver="adam", learning_rate="adaptive",
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42, verbose=False)),
    ("RF", RandomForestRegressor(
        **{k: v for k, v in best_rf_params.items()},
        n_jobs=-1, random_state=42)),
    ("HGBT", HistGradientBoostingRegressor(
        **{k: v for k, v in best_hgbt_params.items()},
        early_stopping=True, random_state=42)),
    ("SVM", SVR(
        kernel="rbf", C=svm_search.best_params_["C"],
        epsilon=svm_search.best_params_["epsilon"],
        gamma=svm_search.best_params_["gamma"])),
    ("ET", ExtraTreesRegressor(
        **{k: v for k, v in et_search.best_params_.items()},
        n_jobs=-1, random_state=42)),
    ("KNN", KNeighborsRegressor(
        **{k: v for k, v in knn_search.best_params_.items()},
        n_jobs=-1)),
]

stacking_reg_v2 = StackingRegressor(
    estimators=reg_estimators_v2,
    final_estimator=RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
    cv=5,
    n_jobs=-1,
    passthrough=False,
)
stacking_reg_v2.fit(X_trainval_sel, y_reg_trainval)
y_pred_stack_v2 = stacking_reg_v2.predict(X_test_sel)
dt = time.time() - t0

r2 = r2_score(y_reg_test, y_pred_stack_v2)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_stack_v2))
mae = mean_absolute_error(y_reg_test, y_pred_stack_v2)
reg_results["Stacking_v2"] = {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred_stack_v2}
log(f"  Stacking v2 回归:  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt:.1f}s)")
log(f"  元学习器 alpha = {stacking_reg_v2.final_estimator_.alpha_:.4f}")

# --- 优化 Voting 回归 (按 CV R² 加权) ---
log("\n  [2/4] Voting v2 回归 (按调优 CV R² 加权)...")
t0 = time.time()

cv_r2_tuned = {
    "MLP": mlp_search.best_score_,
    "RF": rf_search.best_score_,
    "HGBT": hgbt_search.best_score_,
    "SVM": svm_search.best_score_,
    "ET": et_search.best_score_,
    "KNN": knn_search.best_score_,
}
r2_vals = np.array([cv_r2_tuned[n] for n in ["MLP", "RF", "HGBT", "SVM", "ET", "KNN"]])
r2_weights = np.maximum(r2_vals, 0)
r2_weights = r2_weights / r2_weights.sum()
log(f"  权重: " + "  ".join(f"{n}={w:.3f}" for n, w in
    zip(["MLP", "RF", "HGBT", "SVM", "ET", "KNN"], r2_weights)))

voting_reg_v2 = VotingRegressor(
    estimators=reg_estimators_v2,
    weights=r2_weights,
    n_jobs=-1,
)
voting_reg_v2.fit(X_trainval_sel, y_reg_trainval)
y_pred_vote_v2 = voting_reg_v2.predict(X_test_sel)
dt = time.time() - t0

r2 = r2_score(y_reg_test, y_pred_vote_v2)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_vote_v2))
mae = mean_absolute_error(y_reg_test, y_pred_vote_v2)
reg_results["Voting_v2"] = {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred_vote_v2}
log(f"  Voting v2 回归:  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt:.1f}s)")

# --- 优化 Stacking 分类 ---
log("\n  [3/4] Stacking v2 分类...")
t0 = time.time()

cls_estimators_v2 = [
    ("MLP", cls_models["MLP"]),
    ("RF", cls_models["RF"]),
    ("HGBT", cls_models["HGBT"]),
    ("SVM", cls_models["SVM"]),
    ("ET", cls_models["ET"]),
    ("KNN", cls_models["KNN"]),
]

stacking_cls_v2 = StackingClassifier(
    estimators=[
        (name, type(est)(**est.get_params()))  # 克隆
        for name, est in cls_estimators_v2
    ],
    final_estimator=LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        cv=5, max_iter=1000, class_weight="balanced", random_state=42),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,
)
stacking_cls_v2.fit(X_trainval_sel, y_cls_trainval)
y_pred_stack_cls_v2 = stacking_cls_v2.predict(X_test_sel)
y_prob_stack_cls_v2 = stacking_cls_v2.predict_proba(X_test_sel)[:, 1]
dt = time.time() - t0

auc = roc_auc_score(y_cls_test, y_prob_stack_cls_v2)
acc = accuracy_score(y_cls_test, y_pred_stack_cls_v2)
f1 = f1_score(y_cls_test, y_pred_stack_cls_v2)
prec = precision_score(y_cls_test, y_pred_stack_cls_v2)
rec = recall_score(y_cls_test, y_pred_stack_cls_v2)
cls_results["Stacking_v2"] = {"AUC": auc, "Acc": acc, "F1": f1,
                                "Prec": prec, "Rec": rec,
                                "y_pred": y_pred_stack_cls_v2, "y_prob": y_prob_stack_cls_v2}
log(f"  Stacking v2 分类:  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  ({dt:.1f}s)")

# --- 优化 Voting 分类 ---
log("\n  [4/4] Voting v2 分类...")
t0 = time.time()

voting_cls_v2 = VotingClassifier(
    estimators=[
        (name, type(est)(**est.get_params()))
        for name, est in cls_estimators_v2
    ],
    voting="soft",
    weights=r2_weights,  # 用相同权重
    n_jobs=-1,
)
voting_cls_v2.fit(X_trainval_sel, y_cls_trainval)
y_pred_vote_cls_v2 = voting_cls_v2.predict(X_test_sel)
y_prob_vote_cls_v2 = voting_cls_v2.predict_proba(X_test_sel)[:, 1]
dt = time.time() - t0

auc = roc_auc_score(y_cls_test, y_prob_vote_cls_v2)
acc = accuracy_score(y_cls_test, y_pred_vote_cls_v2)
f1 = f1_score(y_cls_test, y_pred_vote_cls_v2)
prec = precision_score(y_cls_test, y_pred_vote_cls_v2)
rec = recall_score(y_cls_test, y_pred_vote_cls_v2)
cls_results["Voting_v2"] = {"AUC": auc, "Acc": acc, "F1": f1,
                              "Prec": prec, "Rec": rec,
                              "y_pred": y_pred_vote_cls_v2, "y_prob": y_prob_vote_cls_v2}
log(f"  Voting v2 分类:  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  ({dt:.1f}s)")


# ══════════════════════════════════════════
#  Part G: 5-Fold CV 对比
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part G: 5-Fold CV (优化模型)")
log("=" * 65)

cv_reg = {}
cv_cls = {}

for name, model in tuned_reg_models.items():
    scores = cross_val_score(model, X_trainval_sel, y_reg_trainval,
                              cv=5, scoring="r2", n_jobs=-1)
    cv_reg[name] = scores
    log(f"  回归 CV R²  {MODEL_NAMES[name]:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Stacking v2 CV (lighter for speed)
stacking_cv_reg = StackingRegressor(
    estimators=[
        ("MLP", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300,
                              early_stopping=True, random_state=42, verbose=False)),
        ("RF", RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                      n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingRegressor(max_iter=200, random_state=42)),
        ("SVM", SVR(kernel="rbf", C=svm_search.best_params_["C"],
                     gamma=svm_search.best_params_["gamma"])),
        ("ET", ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
        ("KNN", KNeighborsRegressor(n_neighbors=knn_search.best_params_["n_neighbors"],
                                     weights="distance", n_jobs=-1)),
    ],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    cv=3, n_jobs=-1,
)
scores = cross_val_score(stacking_cv_reg, X_trainval_sel, y_reg_trainval,
                          cv=5, scoring="r2", n_jobs=1)
cv_reg["Stacking_v2"] = scores
log(f"  回归 CV R²  {'Stacking v2 (优化)':25s}: {scores.mean():.4f} ± {scores.std():.4f}")

log()

for name, model in cls_models.items():
    scores = cross_val_score(model, X_trainval_sel, y_cls_trainval,
                              cv=5, scoring="roc_auc", n_jobs=-1)
    cv_cls[name] = scores
    log(f"  分类 CV AUC {MODEL_NAMES[name]:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

stacking_cv_cls = StackingClassifier(
    estimators=[
        ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                               early_stopping=True, random_state=42, verbose=False)),
        ("RF", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                  random_state=42)),
        ("SVM", SVC(kernel="rbf", C=svm_search.best_params_["C"],
                     gamma=svm_search.best_params_["gamma"],
                     class_weight="balanced", probability=True, random_state=42)),
        ("ET", ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                                     n_jobs=-1, random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=knn_search.best_params_["n_neighbors"],
                                      weights="distance", n_jobs=-1)),
    ],
    final_estimator=LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, max_iter=500,
                                          class_weight="balanced", random_state=42),
    cv=3, stack_method="predict_proba", n_jobs=-1,
)
scores = cross_val_score(stacking_cv_cls, X_trainval_sel, y_cls_trainval,
                          cv=5, scoring="roc_auc", n_jobs=1)
cv_cls["Stacking_v2"] = scores
log(f"  分类 CV AUC {'Stacking v2 (优化)':25s}: {scores.mean():.4f} ± {scores.std():.4f}")


# ══════════════════════════════════════════
#  Part H: 与环节五对比 — 可视化
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part H: 可视化 (优化前后对比)")
log("=" * 65)

# 环节五基线数据 (硬编码)
baseline_reg = {
    "MLP": 0.4998, "RF": 0.5867, "HGBT": 0.5460, "SVM": 0.5648,
    "Stacking_v1": 0.5994, "Voting_v1": 0.6021,
}
baseline_cls = {
    "MLP": 0.8984, "RF": 0.9152, "HGBT": 0.8811, "SVM": 0.9182,
    "Stacking_v1": 0.9272, "Voting_v1": 0.9281,
}

# ── 图1: 回归 R² 优化前后对比 ──
fig, ax = plt.subplots(figsize=(14, 7))
# 原始基线
base_names = ["MLP", "RF", "HGBT", "SVM"]
x = np.arange(len(base_names))
width = 0.35
old_vals = [baseline_reg[n] for n in base_names]
new_vals = [reg_results[n]["R2"] for n in base_names]
bars1 = ax.bar(x - width / 2, old_vals, width, label="环节四 (原始)", alpha=0.6,
               color=[COLORS[n] for n in base_names], edgecolor="gray")
bars2 = ax.bar(x + width / 2, new_vals, width, label="环节六 (优化)", alpha=0.95,
               color=[COLORS[n] for n in base_names], edgecolor="black", linewidth=1.5)

# 新增模型
extra_names = ["ET", "KNN"]
x2 = np.arange(len(extra_names)) + len(base_names)
extra_vals = [reg_results[n]["R2"] for n in extra_names]
ax.bar(x2, extra_vals, width, alpha=0.95,
       color=[COLORS[n] for n in extra_names], edgecolor="black", linewidth=1.5)

# 集成
ens_names_old = ["Stacking_v1", "Voting_v1"]
ens_names_new = ["Stacking_v2", "Voting_v2"]
x3 = np.arange(2) + len(base_names) + len(extra_names)
old_ens = [baseline_reg[n] for n in ens_names_old]
new_ens = [reg_results[n]["R2"] for n in ens_names_new]
ax.bar(x3 - width / 2, old_ens, width, alpha=0.6,
       color=[COLORS.get(n, "#999") for n in ens_names_new], edgecolor="gray")
ax.bar(x3 + width / 2, new_ens, width, alpha=0.95,
       color=[COLORS.get(n, "#999") for n in ens_names_new], edgecolor="black", linewidth=1.5)

all_labels = [MODEL_NAMES[n] for n in base_names + extra_names] + ["Stacking", "Voting"]
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("R²", fontsize=13)
ax.set_title("回归 R²: 优化前 vs 优化后", fontsize=15, fontweight="bold")

# 数值标注
all_new_vals = new_vals + extra_vals + new_ens
for i, v in enumerate(all_new_vals):
    ax.text(i if i < len(base_names) else (i + width / 2 if i >= len(base_names) + len(extra_names) else i),
            v + 0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig(fig, "O1_regression_before_after.png")

# ── 图2: 分类 AUC 优化前后对比 ──
fig, ax = plt.subplots(figsize=(14, 7))
old_cls_vals = [baseline_cls[n] for n in base_names]
new_cls_vals = [cls_results[n]["AUC"] for n in base_names]
bars1 = ax.bar(x - width / 2, old_cls_vals, width, label="环节五 (原始)", alpha=0.6,
               color=[COLORS[n] for n in base_names], edgecolor="gray")
bars2 = ax.bar(x + width / 2, new_cls_vals, width, label="环节六 (优化)", alpha=0.95,
               color=[COLORS[n] for n in base_names], edgecolor="black", linewidth=1.5)

extra_cls = [cls_results[n]["AUC"] for n in extra_names]
ax.bar(x2, extra_cls, width, alpha=0.95,
       color=[COLORS[n] for n in extra_names], edgecolor="black", linewidth=1.5)

old_cls_ens = [baseline_cls[n] for n in ens_names_old]
new_cls_ens = [cls_results[n]["AUC"] for n in ens_names_new]
ax.bar(x3 - width / 2, old_cls_ens, width, alpha=0.6,
       color=[COLORS.get(n, "#999") for n in ens_names_new], edgecolor="gray")
ax.bar(x3 + width / 2, new_cls_ens, width, alpha=0.95,
       color=[COLORS.get(n, "#999") for n in ens_names_new], edgecolor="black", linewidth=1.5)

ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("AUC", fontsize=13)
ax.set_title("分类 AUC: 优化前 vs 优化后", fontsize=15, fontweight="bold")
all_new_cls = new_cls_vals + extra_cls + new_cls_ens
for i, v in enumerate(all_new_cls):
    ax.text(i if i < len(base_names) else (i + width / 2 if i >= len(base_names) + len(extra_names) else i),
            v + 0.002, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(bottom=min(min(old_cls_vals), min(new_cls_vals)) - 0.05)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_fig(fig, "O2_classification_before_after.png")

# ── 图3: ROC 曲线 (所有模型) ──
fig, ax = plt.subplots(figsize=(9, 9))
for name in ["MLP", "RF", "HGBT", "SVM", "ET", "KNN", "Stacking_v2", "Voting_v2"]:
    fpr, tpr, _ = roc_curve(y_cls_test, cls_results[name]["y_prob"])
    auc_val = cls_results[name]["AUC"]
    lw = 3 if "Stacking" in name or "Voting" in name else 1.5
    ls = "-" if "Stacking" in name or "Voting" in name else "--"
    ax.plot(fpr, tpr, color=COLORS.get(name, "#333"), lw=lw, linestyle=ls,
            label=f"{MODEL_NAMES[name]} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("FPR", fontsize=12)
ax.set_ylabel("TPR", fontsize=12)
ax.set_title("ROC 曲线：优化后全模型", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
save_fig(fig, "O3_roc_all_optimized.png")

# ── 图4: 散点 (最佳集成) ──
best_ens_name = max(["Stacking_v2", "Voting_v2"], key=lambda n: reg_results[n]["R2"])
best_res = reg_results[best_ens_name]
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_reg_test, best_res["y_pred"], alpha=0.6, s=40,
           c=COLORS.get(best_ens_name, "#e74c3c"), edgecolors="white")
lo = min(y_reg_test.min(), best_res["y_pred"].min()) - 0.3
hi = max(y_reg_test.max(), best_res["y_pred"].max()) + 0.3
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("真实 pChEMBL", fontsize=12)
ax.set_ylabel("预测 pChEMBL", fontsize=12)
ax.set_title(f"优化后最佳集成: {MODEL_NAMES[best_ens_name]}\n"
             f"R²={best_res['R2']:.4f}  RMSE={best_res['RMSE']:.4f}  MAE={best_res['MAE']:.4f}",
             fontsize=13, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
save_fig(fig, "O4_best_ensemble_scatter.png")

# ── 图5: MI 特征重要性分布 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Mutual Information 特征选择", fontsize=15, fontweight="bold")

# (a) MI 分布直方图
ax = axes[0]
ax.hist(mi_scores, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(mi_scores[mi_order[best_k - 1]], color="red", linestyle="--", linewidth=2,
           label=f"选择阈值 (Top {best_k})")
ax.set_xlabel("Mutual Information", fontsize=12)
ax.set_ylabel("特征数量", fontsize=12)
ax.set_title("MI 分布", fontsize=13)
ax.legend(fontsize=10)

# (b) Top 20 MI 特征
ax = axes[1]
top20_idx = mi_order[:20]
top20_names = [feature_names[i] for i in top20_idx]
top20_mi = mi_scores[top20_idx]
colors_mi = ["#e67e22" if n.startswith("Morgan_") else "#3498db" for n in top20_names]
ax.barh(range(19, -1, -1), top20_mi, color=colors_mi, height=0.7)
ax.set_yticks(range(19, -1, -1))
ax.set_yticklabels(top20_names, fontsize=8)
ax.set_xlabel("Mutual Information", fontsize=12)
ax.set_title("Top 20 特征 (MI)", fontsize=13)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#3498db", label="描述符"),
                    Patch(color="#e67e22", label="Morgan 指纹")],
          fontsize=9)

plt.tight_layout()
save_fig(fig, "O5_mutual_information.png")

# ── 图6: CV 对比 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("5-Fold CV: 优化后模型", fontsize=15, fontweight="bold")

cv_names = list(cv_reg.keys())
ax = axes[0]
means = [cv_reg[n].mean() for n in cv_names]
stds = [cv_reg[n].std() for n in cv_names]
bars = ax.bar(range(len(cv_names)), means, yerr=stds, capsize=4,
              color=[COLORS.get(n, "#999") for n in cv_names], edgecolor="white")
ax.set_xticks(range(len(cv_names)))
ax.set_xticklabels([MODEL_NAMES.get(n, n) for n in cv_names], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("回归 CV R²", fontsize=13)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
            f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
best_idx = np.argmax(means)
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(3)

cv_cls_names = list(cv_cls.keys())
ax = axes[1]
means = [cv_cls[n].mean() for n in cv_cls_names]
stds = [cv_cls[n].std() for n in cv_cls_names]
bars = ax.bar(range(len(cv_cls_names)), means, yerr=stds, capsize=4,
              color=[COLORS.get(n, "#999") for n in cv_cls_names], edgecolor="white")
ax.set_xticks(range(len(cv_cls_names)))
ax.set_xticklabels([MODEL_NAMES.get(n, n) for n in cv_cls_names], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("分类 CV AUC", fontsize=13)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
            f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
best_idx = np.argmax(means)
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(3)

plt.tight_layout()
save_fig(fig, "O6_cv_optimized.png")


# ══════════════════════════════════════════
#  Part I: 总结
# ══════════════════════════════════════════
log()
log("=" * 65)
log("Part I: 优化前后对比总结")
log("=" * 65)

log(f"\n  特征维度: 2105 → {best_k} (MI 选择)")

log(f"\n  【回归 R² (测试集)】")
log(f"  {'模型':25s} {'原始':>8s} {'优化后':>8s} {'提升':>8s}")
log(f"  {'─'*55}")
for n in ["MLP", "RF", "HGBT", "SVM"]:
    old = baseline_reg[n]
    new = reg_results[n]["R2"]
    log(f"  {MODEL_NAMES[n]:25s} {old:8.4f} {new:8.4f} {new-old:+8.4f}")
for n in ["ET", "KNN"]:
    log(f"  {MODEL_NAMES[n]:25s} {'--':>8s} {reg_results[n]['R2']:8.4f} {'(新增)':>8s}")
log(f"  {'─'*55}")
log(f"  {'Stacking':25s} {baseline_reg['Stacking_v1']:8.4f} {reg_results['Stacking_v2']['R2']:8.4f} "
    f"{reg_results['Stacking_v2']['R2']-baseline_reg['Stacking_v1']:+8.4f}")
log(f"  {'Voting':25s} {baseline_reg['Voting_v1']:8.4f} {reg_results['Voting_v2']['R2']:8.4f} "
    f"{reg_results['Voting_v2']['R2']-baseline_reg['Voting_v1']:+8.4f}")

log(f"\n  【分类 AUC (测试集)】")
log(f"  {'模型':25s} {'原始':>8s} {'优化后':>8s} {'提升':>8s}")
log(f"  {'─'*55}")
for n in ["MLP", "RF", "HGBT", "SVM"]:
    old = baseline_cls[n]
    new = cls_results[n]["AUC"]
    log(f"  {MODEL_NAMES[n]:25s} {old:8.4f} {new:8.4f} {new-old:+8.4f}")
for n in ["ET", "KNN"]:
    log(f"  {MODEL_NAMES[n]:25s} {'--':>8s} {cls_results[n]['AUC']:8.4f} {'(新增)':>8s}")
log(f"  {'─'*55}")
log(f"  {'Stacking':25s} {baseline_cls['Stacking_v1']:8.4f} {cls_results['Stacking_v2']['AUC']:8.4f} "
    f"{cls_results['Stacking_v2']['AUC']-baseline_cls['Stacking_v1']:+8.4f}")
log(f"  {'Voting':25s} {baseline_cls['Voting_v1']:8.4f} {cls_results['Voting_v2']['AUC']:8.4f} "
    f"{cls_results['Voting_v2']['AUC']-baseline_cls['Voting_v1']:+8.4f}")

# 写入报告
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
log(f"\n报告已保存: {REPORT_FILE}")
log("全部完成！")
