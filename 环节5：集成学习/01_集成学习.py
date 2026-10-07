"""
环节五：集成学习
================
输入:  环节三/特征工程后数据.npz
输出:
  - figures/  集成 vs 单模型对比图
  - 集成学习评估报告.txt

集成策略:
  1) Stacking (元学习器: Ridge / LogisticRegression)
  2) Soft Voting / 加权平均
  3) 与环节四单模型对比分析

基学习器 (同环节四):
  MLP, Random Forest, HistGradientBoosting, SVM
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
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve,
)
from sklearn.model_selection import cross_val_score, cross_val_predict
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_NPZ = os.path.join(PROJECT_DIR, "环节三：特征工程", "特征工程后数据.npz")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
REPORT_FILE = os.path.join(SCRIPT_DIR, "集成学习评估报告.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = []

COLORS = {
    "MLP": "#3498db", "RF": "#2ecc71", "HGBT": "#e67e22", "SVM": "#9b59b6",
    "Stacking": "#e74c3c", "Voting": "#1abc9c",
}
MODEL_NAMES = {
    "MLP": "MLP 神经网络", "RF": "Random Forest",
    "HGBT": "HistGradientBoosting", "SVM": "SVM (RBF)",
    "Stacking": "Stacking 集成", "Voting": "Voting 集成",
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
log("=" * 60)
log("环节五：集成学习")
log("=" * 60)

data = np.load(INPUT_NPZ)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_reg_train, y_reg_val, y_reg_test = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]
y_cls_train, y_cls_val, y_cls_test = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]
n_desc, n_fp = int(data["n_desc"]), int(data["n_fp"])

# 合并 train + val 用于最终训练（集成学习内部自带 CV，不需要手动留验证集）
X_trainval = np.vstack([X_train, X_val])
y_reg_trainval = np.concatenate([y_reg_train, y_reg_val])
y_cls_trainval = np.concatenate([y_cls_train, y_cls_val])

log(f"  Train+Val: {X_trainval.shape},  Test: {X_test.shape}")
log(f"  Features: {X_trainval.shape[1]} (描述符 {n_desc} + 指纹 {n_fp})")
log()

# ══════════════════════════════════════════
#  定义基学习器（与环节四一致）
# ══════════════════════════════════════════

def make_reg_estimators():
    """返回回归基学习器列表 (name, estimator)"""
    return [
        ("MLP", MLPRegressor(
            hidden_layer_sizes=(512, 256, 128), activation="relu", solver="adam",
            alpha=1e-4, learning_rate="adaptive", learning_rate_init=1e-3,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, batch_size=64, random_state=42, verbose=False)),
        ("RF", RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingRegressor(
            max_iter=500, max_depth=6, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42)),
        ("SVM", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
    ]


def make_cls_estimators():
    """返回分类基学习器列表 (name, estimator)"""
    return [
        ("MLP", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation="relu", solver="adam",
            alpha=1e-4, learning_rate="adaptive", learning_rate_init=1e-3,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, batch_size=64, random_state=42, verbose=False)),
        ("RF", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, class_weight="balanced", random_state=42)),
        ("SVM", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                     probability=True, random_state=42)),
    ]


# ══════════════════════════════════════════
#  Part A: 先单独训练4个基学习器（收集基线结果）
# ══════════════════════════════════════════
log("=" * 60)
log("Part A: 训练基学习器（基线）")
log("=" * 60)

base_reg_results = {}  # name -> {R2, RMSE, MAE, y_pred}
base_cls_results = {}  # name -> {AUC, Acc, F1, Prec, Rec, y_pred, y_prob}

for name, est in make_reg_estimators():
    t0 = time.time()
    est.fit(X_trainval, y_reg_trainval)
    y_pred = est.predict(X_test)
    dt = time.time() - t0
    r2 = r2_score(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    base_reg_results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred}
    log(f"  回归 {MODEL_NAMES[name]:25s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt:.1f}s)")

log()

for name, est in make_cls_estimators():
    t0 = time.time()
    est.fit(X_trainval, y_cls_trainval)
    y_pred = est.predict(X_test)
    y_prob = est.predict_proba(X_test)[:, 1]
    dt = time.time() - t0
    auc = roc_auc_score(y_cls_test, y_prob)
    acc = accuracy_score(y_cls_test, y_pred)
    f1 = f1_score(y_cls_test, y_pred)
    prec = precision_score(y_cls_test, y_pred)
    rec = recall_score(y_cls_test, y_pred)
    base_cls_results[name] = {"AUC": auc, "Acc": acc, "F1": f1,
                               "Prec": prec, "Rec": rec,
                               "y_pred": y_pred, "y_prob": y_prob}
    log(f"  分类 {MODEL_NAMES[name]:25s}  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  ({dt:.1f}s)")

# ══════════════════════════════════════════
#  Part B: Stacking 集成
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part B: Stacking 集成 (5-fold CV)")
log("=" * 60)

# --- Stacking 回归 ---
log("\n--- Stacking 回归 (元学习器: RidgeCV) ---")
t0 = time.time()
stacking_reg = StackingRegressor(
    estimators=make_reg_estimators(),
    final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
    cv=5,
    n_jobs=-1,
    passthrough=False,  # 只用基学习器的输出
)
stacking_reg.fit(X_trainval, y_reg_trainval)
y_pred_stack_reg = stacking_reg.predict(X_test)
dt = time.time() - t0

r2 = r2_score(y_reg_test, y_pred_stack_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_stack_reg))
mae = mean_absolute_error(y_reg_test, y_pred_stack_reg)
stack_reg_res = {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred_stack_reg}
log(f"  Stacking 回归:  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt:.1f}s)")
log(f"  元学习器选定 alpha = {stacking_reg.final_estimator_.alpha_:.4f}")

# --- Stacking 分类 ---
log("\n--- Stacking 分类 (元学习器: LogisticRegressionCV) ---")
t0 = time.time()
stacking_cls = StackingClassifier(
    estimators=make_cls_estimators(),
    final_estimator=LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
        cv=5, max_iter=1000, class_weight="balanced", random_state=42,
    ),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,
)
stacking_cls.fit(X_trainval, y_cls_trainval)
y_pred_stack_cls = stacking_cls.predict(X_test)
y_prob_stack_cls = stacking_cls.predict_proba(X_test)[:, 1]
dt = time.time() - t0

auc = roc_auc_score(y_cls_test, y_prob_stack_cls)
acc = accuracy_score(y_cls_test, y_pred_stack_cls)
f1 = f1_score(y_cls_test, y_pred_stack_cls)
prec = precision_score(y_cls_test, y_pred_stack_cls)
rec = recall_score(y_cls_test, y_pred_stack_cls)
stack_cls_res = {"AUC": auc, "Acc": acc, "F1": f1, "Prec": prec, "Rec": rec,
                  "y_pred": y_pred_stack_cls, "y_prob": y_prob_stack_cls}
log(f"  Stacking 分类:  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  ({dt:.1f}s)")
log(f"  元学习器选定 C = {stacking_cls.final_estimator_.C_[0]:.4f}")

# ══════════════════════════════════════════
#  Part C: Voting 集成
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part C: Voting 集成 (加权)")
log("=" * 60)

# 根据基线 R² 计算回归权重
reg_r2_vals = np.array([base_reg_results[n]["R2"] for n in ["MLP", "RF", "HGBT", "SVM"]])
reg_weights = np.maximum(reg_r2_vals, 0)  # 防负值
reg_weights = reg_weights / reg_weights.sum()
log(f"  回归权重 (按 R²): MLP={reg_weights[0]:.3f}  RF={reg_weights[1]:.3f}  "
    f"HGBT={reg_weights[2]:.3f}  SVM={reg_weights[3]:.3f}")

t0 = time.time()
voting_reg = VotingRegressor(
    estimators=make_reg_estimators(),
    weights=reg_weights,
    n_jobs=-1,
)
voting_reg.fit(X_trainval, y_reg_trainval)
y_pred_vote_reg = voting_reg.predict(X_test)
dt = time.time() - t0

r2 = r2_score(y_reg_test, y_pred_vote_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_vote_reg))
mae = mean_absolute_error(y_reg_test, y_pred_vote_reg)
vote_reg_res = {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred_vote_reg}
log(f"  Voting 回归:  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt:.1f}s)")

# 根据基线 AUC 计算分类权重
cls_auc_vals = np.array([base_cls_results[n]["AUC"] for n in ["MLP", "RF", "HGBT", "SVM"]])
cls_weights = cls_auc_vals / cls_auc_vals.sum()
log(f"\n  分类权重 (按 AUC): MLP={cls_weights[0]:.3f}  RF={cls_weights[1]:.3f}  "
    f"HGBT={cls_weights[2]:.3f}  SVM={cls_weights[3]:.3f}")

t0 = time.time()
voting_cls = VotingClassifier(
    estimators=make_cls_estimators(),
    voting="soft",
    weights=cls_weights,
    n_jobs=-1,
)
voting_cls.fit(X_trainval, y_cls_trainval)
y_pred_vote_cls = voting_cls.predict(X_test)
y_prob_vote_cls = voting_cls.predict_proba(X_test)[:, 1]
dt = time.time() - t0

auc = roc_auc_score(y_cls_test, y_prob_vote_cls)
acc = accuracy_score(y_cls_test, y_pred_vote_cls)
f1 = f1_score(y_cls_test, y_pred_vote_cls)
prec = precision_score(y_cls_test, y_pred_vote_cls)
rec = recall_score(y_cls_test, y_pred_vote_cls)
vote_cls_res = {"AUC": auc, "Acc": acc, "F1": f1, "Prec": prec, "Rec": rec,
                 "y_pred": y_pred_vote_cls, "y_prob": y_prob_vote_cls}
log(f"  Voting 分类:  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  ({dt:.1f}s)")

# ══════════════════════════════════════════
#  Part D: 5-Fold 交叉验证对比
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part D: 5-Fold 交叉验证 (全部6个模型)")
log("=" * 60)

cv_reg_scores = {}
cv_cls_scores = {}

# 基学习器 CV
for name, est in make_reg_estimators():
    scores = cross_val_score(est, X_trainval, y_reg_trainval, cv=5,
                              scoring="r2", n_jobs=-1)
    cv_reg_scores[name] = scores
    log(f"  回归 CV R² {MODEL_NAMES[name]:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Stacking 回归 CV（使用更轻量配置避免 CV 套 CV 过慢）
stacking_reg_cv = StackingRegressor(
    estimators=[
        ("MLP", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300,
                              early_stopping=True, random_state=42, verbose=False)),
        ("RF", RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                      n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                                learning_rate=0.05, random_state=42)),
        ("SVM", SVR(kernel="rbf", C=10.0, gamma="scale")),
    ],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    cv=3, n_jobs=-1,
)
scores = cross_val_score(stacking_reg_cv, X_trainval, y_reg_trainval, cv=5,
                          scoring="r2", n_jobs=1)
cv_reg_scores["Stacking"] = scores
log(f"  回归 CV R² {'Stacking 集成':25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# Voting 回归 CV
voting_reg_cv = VotingRegressor(
    estimators=[
        ("MLP", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300,
                              early_stopping=True, random_state=42, verbose=False)),
        ("RF", RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                      n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                                learning_rate=0.05, random_state=42)),
        ("SVM", SVR(kernel="rbf", C=10.0, gamma="scale")),
    ],
    n_jobs=-1,
)
scores = cross_val_score(voting_reg_cv, X_trainval, y_reg_trainval, cv=5,
                          scoring="r2", n_jobs=1)
cv_reg_scores["Voting"] = scores
log(f"  回归 CV R² {'Voting 集成':25s}: {scores.mean():.4f} ± {scores.std():.4f}")

log()

# 分类 CV
for name, est in make_cls_estimators():
    scores = cross_val_score(est, X_trainval, y_cls_trainval, cv=5,
                              scoring="roc_auc", n_jobs=-1)
    cv_cls_scores[name] = scores
    log(f"  分类 CV AUC {MODEL_NAMES[name]:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

stacking_cls_cv = StackingClassifier(
    estimators=[
        ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                               early_stopping=True, random_state=42, verbose=False)),
        ("RF", RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                       class_weight="balanced", n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                  learning_rate=0.05, class_weight="balanced",
                                                  random_state=42)),
        ("SVM", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                     probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, max_iter=500,
                                          class_weight="balanced", random_state=42),
    cv=3, stack_method="predict_proba", n_jobs=-1,
)
scores = cross_val_score(stacking_cls_cv, X_trainval, y_cls_trainval, cv=5,
                          scoring="roc_auc", n_jobs=1)
cv_cls_scores["Stacking"] = scores
log(f"  分类 CV AUC {'Stacking 集成':25s}: {scores.mean():.4f} ± {scores.std():.4f}")

voting_cls_cv = VotingClassifier(
    estimators=[
        ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                               early_stopping=True, random_state=42, verbose=False)),
        ("RF", RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                       class_weight="balanced", n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                  learning_rate=0.05, class_weight="balanced",
                                                  random_state=42)),
        ("SVM", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                     probability=True, random_state=42)),
    ],
    voting="soft", n_jobs=-1,
)
scores = cross_val_score(voting_cls_cv, X_trainval, y_cls_trainval, cv=5,
                          scoring="roc_auc", n_jobs=1)
cv_cls_scores["Voting"] = scores
log(f"  分类 CV AUC {'Voting 集成':25s}: {scores.mean():.4f} ± {scores.std():.4f}")


# ══════════════════════════════════════════
#  Part E: 可视化
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part E: 生成可视化图表")
log("=" * 60)

ALL_NAMES = ["MLP", "RF", "HGBT", "SVM", "Stacking", "Voting"]

# ── 图1: 回归 R²/RMSE/MAE 柱状对比图 ──
all_reg = {**base_reg_results, "Stacking": stack_reg_res, "Voting": vote_reg_res}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("回归模型性能对比：单模型 vs 集成", fontsize=16, fontweight="bold")

for ax, metric, ylabel in zip(axes, ["R2", "RMSE", "MAE"], ["R²", "RMSE", "MAE"]):
    vals = [all_reg[n][metric] for n in ALL_NAMES]
    bars = ax.bar(range(len(ALL_NAMES)), vals,
                  color=[COLORS[n] for n in ALL_NAMES], edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(ALL_NAMES)))
    ax.set_xticklabels([MODEL_NAMES[n] for n in ALL_NAMES], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=13)
    # 标注数值
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # 高亮最佳
    if metric == "R2":
        best_idx = np.argmax(vals)
    else:
        best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
save_fig(fig, "E1_regression_comparison.png")

# ── 图2: 分类 AUC/Acc/F1 柱状对比图 ──
all_cls = {**base_cls_results, "Stacking": stack_cls_res, "Voting": vote_cls_res}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("分类模型性能对比：单模型 vs 集成", fontsize=16, fontweight="bold")

for ax, metric, ylabel in zip(axes, ["AUC", "Acc", "F1"], ["AUC", "Accuracy", "F1-Score"]):
    vals = [all_cls[n][metric] for n in ALL_NAMES]
    bars = ax.bar(range(len(ALL_NAMES)), vals,
                  color=[COLORS[n] for n in ALL_NAMES], edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(ALL_NAMES)))
    ax.set_xticklabels([MODEL_NAMES[n] for n in ALL_NAMES], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=13)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)
    ax.set_ylim(bottom=min(vals) - 0.05, top=max(vals) + 0.03)

plt.tight_layout()
save_fig(fig, "E2_classification_comparison.png")

# ── 图3: ROC 曲线 (6条) ──
fig, ax = plt.subplots(figsize=(8, 8))
for name in ALL_NAMES:
    fpr, tpr, _ = roc_curve(y_cls_test, all_cls[name]["y_prob"])
    auc_val = all_cls[name]["AUC"]
    lw = 3 if name in ("Stacking", "Voting") else 1.5
    ls = "-" if name in ("Stacking", "Voting") else "--"
    ax.plot(fpr, tpr, color=COLORS[name], lw=lw, linestyle=ls,
            label=f"{MODEL_NAMES[name]} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC 曲线对比：单模型 vs 集成", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
save_fig(fig, "E3_roc_curves_ensemble.png")

# ── 图4: 回归散点图（Stacking + Voting）──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("集成模型回归预测 vs 真实值（测试集）", fontsize=14, fontweight="bold")

for ax, name, res in zip(axes, ["Stacking", "Voting"], [stack_reg_res, vote_reg_res]):
    y_pred = res["y_pred"]
    ax.scatter(y_reg_test, y_pred, alpha=0.6, s=30, c=COLORS[name], edgecolors="white", linewidth=0.5)
    lo = min(y_reg_test.min(), y_pred.min()) - 0.3
    hi = max(y_reg_test.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实 pChEMBL", fontsize=11)
    ax.set_ylabel("预测 pChEMBL", fontsize=11)
    ax.set_title(f"{MODEL_NAMES[name]}\nR²={res['R2']:.4f}  RMSE={res['RMSE']:.4f}", fontsize=12)
    ax.set_aspect("equal")

plt.tight_layout()
save_fig(fig, "E4_ensemble_regression_scatter.png")

# ── 图5: 混淆矩阵（Stacking + Voting）──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("集成模型混淆矩阵（测试集）", fontsize=14, fontweight="bold")

for ax, name, res in zip(axes, ["Stacking", "Voting"], [stack_cls_res, vote_cls_res]):
    cm = confusion_matrix(y_cls_test, res["y_pred"])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"])
    ax.set_yticklabels(["Inactive", "Active"])
    ax.set_xlabel("预测", fontsize=11)
    ax.set_ylabel("真实", fontsize=11)
    ax.set_title(f"{MODEL_NAMES[name]}\nAUC={res['AUC']:.3f}  F1={res['F1']:.3f}", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
save_fig(fig, "E5_ensemble_confusion_matrix.png")

# ── 图6: 交叉验证对比 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("5-Fold 交叉验证: 单模型 vs 集成", fontsize=15, fontweight="bold")

# 回归 CV
ax = axes[0]
positions = range(len(ALL_NAMES))
means = [cv_reg_scores[n].mean() for n in ALL_NAMES]
stds = [cv_reg_scores[n].std() for n in ALL_NAMES]
bars = ax.bar(positions, means, yerr=stds, capsize=5,
              color=[COLORS[n] for n in ALL_NAMES], edgecolor="white", linewidth=1.5)
ax.set_xticks(positions)
ax.set_xticklabels([MODEL_NAMES[n] for n in ALL_NAMES], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("5-Fold CV: 回归 R²", fontsize=13)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
            f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
best_idx = np.argmax(means)
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(3)

# 分类 CV
ax = axes[1]
means = [cv_cls_scores[n].mean() for n in ALL_NAMES]
stds = [cv_cls_scores[n].std() for n in ALL_NAMES]
bars = ax.bar(positions, means, yerr=stds, capsize=5,
              color=[COLORS[n] for n in ALL_NAMES], edgecolor="white", linewidth=1.5)
ax.set_xticks(positions)
ax.set_xticklabels([MODEL_NAMES[n] for n in ALL_NAMES], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("5-Fold CV: 分类 AUC", fontsize=13)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
            f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
best_idx = np.argmax(means)
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(3)

plt.tight_layout()
save_fig(fig, "E6_cross_validation_ensemble.png")

# ── 图7: 性能提升雷达图 ──
# 将 6 个模型在多维指标上的表现用雷达图展示
fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))
fig.suptitle("模型综合性能雷达图", fontsize=15, fontweight="bold", y=1.02)

# 回归雷达: R², 1-RMSE_norm, 1-MAE_norm
reg_metrics = ["R²", "1−RMSE(归一化)", "1−MAE(归一化)", "CV R²"]
reg_data = {}
rmse_max = max(all_reg[n]["RMSE"] for n in ALL_NAMES)
mae_max = max(all_reg[n]["MAE"] for n in ALL_NAMES)
for n in ALL_NAMES:
    reg_data[n] = [
        all_reg[n]["R2"],
        1 - all_reg[n]["RMSE"] / rmse_max,
        1 - all_reg[n]["MAE"] / mae_max,
        cv_reg_scores[n].mean(),
    ]

angles = np.linspace(0, 2 * np.pi, len(reg_metrics), endpoint=False).tolist()
angles += angles[:1]

ax = axes[0]
for n in ALL_NAMES:
    vals = reg_data[n] + reg_data[n][:1]
    lw = 2.5 if n in ("Stacking", "Voting") else 1.2
    ax.plot(angles, vals, "o-", color=COLORS[n], lw=lw, label=MODEL_NAMES[n], markersize=4)
    ax.fill(angles, vals, alpha=0.05, color=COLORS[n])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(reg_metrics, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title("回归任务", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

# 分类雷达: AUC, Acc, F1, Precision, Recall, CV AUC
cls_metrics = ["AUC", "Accuracy", "F1", "Precision", "Recall", "CV AUC"]
cls_data = {}
for n in ALL_NAMES:
    cls_data[n] = [
        all_cls[n]["AUC"],
        all_cls[n]["Acc"],
        all_cls[n]["F1"],
        all_cls[n]["Prec"],
        all_cls[n]["Rec"],
        cv_cls_scores[n].mean(),
    ]

angles2 = np.linspace(0, 2 * np.pi, len(cls_metrics), endpoint=False).tolist()
angles2 += angles2[:1]

ax = axes[1]
for n in ALL_NAMES:
    vals = cls_data[n] + cls_data[n][:1]
    lw = 2.5 if n in ("Stacking", "Voting") else 1.2
    ax.plot(angles2, vals, "o-", color=COLORS[n], lw=lw, label=MODEL_NAMES[n], markersize=4)
    ax.fill(angles2, vals, alpha=0.05, color=COLORS[n])
ax.set_xticks(angles2[:-1])
ax.set_xticklabels(cls_metrics, fontsize=9)
ax.set_ylim(0.6, 1.0)
ax.set_title("分类任务", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout()
save_fig(fig, "E7_radar_chart.png")

# ── 图8: 集成 vs 最佳单模型提升幅度 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("集成学习 vs 最佳单模型：性能提升", fontsize=15, fontweight="bold")

# 回归提升
ax = axes[0]
best_base_r2 = max(base_reg_results[n]["R2"] for n in ["MLP", "RF", "HGBT", "SVM"])
best_base_name_reg = max(base_reg_results, key=lambda n: base_reg_results[n]["R2"])
improvements_reg = {
    "Stacking": stack_reg_res["R2"] - best_base_r2,
    "Voting": vote_reg_res["R2"] - best_base_r2,
}
names = list(improvements_reg.keys())
vals = list(improvements_reg.values())
bar_colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in vals]
bars = ax.barh(names, vals, color=bar_colors, height=0.5, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.002 * (1 if v >= 0 else -1), bar.get_y() + bar.get_height() / 2,
            f"{v:+.4f}", ha="left" if v >= 0 else "right", va="center", fontsize=12, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("R² 提升量", fontsize=12)
ax.set_title(f"回归 R² 提升 (基线: {MODEL_NAMES[best_base_name_reg]} R²={best_base_r2:.4f})", fontsize=12)

# 分类提升
ax = axes[1]
best_base_auc = max(base_cls_results[n]["AUC"] for n in ["MLP", "RF", "HGBT", "SVM"])
best_base_name_cls = max(base_cls_results, key=lambda n: base_cls_results[n]["AUC"])
improvements_cls = {
    "Stacking": stack_cls_res["AUC"] - best_base_auc,
    "Voting": vote_cls_res["AUC"] - best_base_auc,
}
names = list(improvements_cls.keys())
vals = list(improvements_cls.values())
bar_colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in vals]
bars = ax.barh(names, vals, color=bar_colors, height=0.5, edgecolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.002 * (1 if v >= 0 else -1), bar.get_y() + bar.get_height() / 2,
            f"{v:+.4f}", ha="left" if v >= 0 else "right", va="center", fontsize=12, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("AUC 提升量", fontsize=12)
ax.set_title(f"分类 AUC 提升 (基线: {MODEL_NAMES[best_base_name_cls]} AUC={best_base_auc:.4f})", fontsize=12)

plt.tight_layout()
save_fig(fig, "E8_improvement_over_best_base.png")


# ══════════════════════════════════════════
#  Part F: 总结报告
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part F: 综合对比总结")
log("=" * 60)

log("\n  【回归任务 - 测试集】")
log(f"  {'模型':25s} {'R²':>8s} {'RMSE':>8s} {'MAE':>8s}")
log(f"  {'─'*55}")
for n in ALL_NAMES:
    r = all_reg[n]
    marker = " ★" if n in ("Stacking", "Voting") else ""
    log(f"  {MODEL_NAMES[n]:25s} {r['R2']:8.4f} {r['RMSE']:8.4f} {r['MAE']:8.4f}{marker}")

log("\n  【分类任务 - 测试集】")
log(f"  {'模型':25s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s}")
log(f"  {'─'*70}")
for n in ALL_NAMES:
    c = all_cls[n]
    marker = " ★" if n in ("Stacking", "Voting") else ""
    log(f"  {MODEL_NAMES[n]:25s} {c['AUC']:8.4f} {c['Acc']:8.4f} {c['F1']:8.4f} "
        f"{c['Prec']:8.4f} {c['Rec']:8.4f}{marker}")

log("\n  【5-Fold CV】")
log(f"  {'模型':25s} {'CV R²':>16s} {'CV AUC':>16s}")
log(f"  {'─'*60}")
for n in ALL_NAMES:
    r_cv = cv_reg_scores[n]
    c_cv = cv_cls_scores[n]
    marker = " ★" if n in ("Stacking", "Voting") else ""
    log(f"  {MODEL_NAMES[n]:25s} {r_cv.mean():.4f}±{r_cv.std():.4f}   "
        f"{c_cv.mean():.4f}±{c_cv.std():.4f}{marker}")

# 保存报告
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
log(f"\n报告已保存: {REPORT_FILE}")
log("全部完成！")
