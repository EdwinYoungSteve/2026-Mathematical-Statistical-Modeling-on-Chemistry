"""
环节四：模型训练与评估
======================
输入:  环节三/特征工程后数据.npz
输出:
  - figures/  目录下的性能对比图、学习曲线、混淆矩阵等
  - 模型评估报告.txt

模型清单 (回归 + 分类):
  1) MLP 神经网络     (sklearn MLPRegressor / MLPClassifier)
  2) Random Forest
  3) HistGradientBoosting  (等价 XGBoost/LightGBM)
  4) SVM (RBF kernel)

评估:
  回归: R², RMSE, MAE  +  散点图
  分类: AUC, Accuracy, F1  +  ROC 曲线 + 混淆矩阵
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              roc_auc_score, accuracy_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, roc_curve)
from sklearn.model_selection import cross_val_score
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
REPORT_FILE = os.path.join(SCRIPT_DIR, "模型评估报告.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = []


def log(msg=""):
    print(msg)
    report.append(msg)


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}")


# ── 颜色 ──
COLORS = {"MLP": "#3498db", "RF": "#2ecc71", "HGBT": "#e67e22", "SVM": "#9b59b6"}
COLOR_ACTIVE = "#2ecc71"
COLOR_INACTIVE = "#e74c3c"

# ══════════════════════════════════════════
#  读取数据
# ══════════════════════════════════════════
log("=" * 60)
log("读取特征工程后数据")
log("=" * 60)

data = np.load(INPUT_NPZ)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_reg_train, y_reg_val, y_reg_test = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]
y_cls_train, y_cls_val, y_cls_test = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]
n_desc, n_fp = int(data["n_desc"]), int(data["n_fp"])

log(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
log(f"  Features: {X_train.shape[1]} (描述符 {n_desc} + 指纹 {n_fp})")

# ══════════════════════════════════════════
#  定义模型
# ══════════════════════════════════════════

# --- 回归模型 ---
reg_models = {
    "MLP": MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,           # L2 正则
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        batch_size=64,
        random_state=42,
        verbose=False,
    ),
    "RF": RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    "HGBT": HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    ),
    "SVM": SVR(
        kernel="rbf",
        C=10.0,
        epsilon=0.1,
        gamma="scale",
    ),
}

# --- 分类模型 ---
cls_models = {
    "MLP": MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        batch_size=64,
        random_state=42,
        verbose=False,
    ),
    "RF": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "HGBT": HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=42,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
}

MODEL_NAMES = {"MLP": "MLP 神经网络", "RF": "Random Forest",
               "HGBT": "HistGradientBoosting", "SVM": "SVM (RBF)"}

# ══════════════════════════════════════════
#  Part A: 回归任务
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part A: 回归任务 (预测 pChEMBL Value)")
log("=" * 60)

reg_results = {}
reg_preds = {}

for name, model in reg_models.items():
    log(f"\n  训练 {MODEL_NAMES[name]}...")
    t0 = time.time()
    model.fit(X_train, y_reg_train)
    train_time = time.time() - t0

    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    r2_val = r2_score(y_reg_val, y_pred_val)
    r2_test = r2_score(y_reg_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_pred_test))
    mae_test = mean_absolute_error(y_reg_test, y_pred_test)

    reg_results[name] = {
        "R² (val)": r2_val, "R² (test)": r2_test,
        "RMSE (test)": rmse_test, "MAE (test)": mae_test,
        "Time (s)": train_time,
    }
    reg_preds[name] = (y_pred_val, y_pred_test)

    log(f"    R²(val)={r2_val:.4f}  R²(test)={r2_test:.4f}  "
        f"RMSE={rmse_test:.4f}  MAE={mae_test:.4f}  [{train_time:.1f}s]")

# ── 回归结果表 ──
log("\n  --- 回归结果汇总 ---")
df_reg = pd.DataFrame(reg_results).T
log(df_reg.to_string())

# ── 图 R1: 各模型预测 vs 真实散点图 ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, (name, (y_pred_val, y_pred_test)) in enumerate(reg_preds.items()):
    ax = axes[i]
    ax.scatter(y_reg_test, y_pred_test, alpha=0.5, s=15,
               color=COLORS[name], edgecolors="none")
    lims = [min(y_reg_test.min(), y_pred_test.min()) - 0.3,
            max(y_reg_test.max(), y_pred_test.max()) + 0.3]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("真实 pChEMBL", fontsize=11)
    ax.set_ylabel("预测 pChEMBL", fontsize=11)
    ax.set_title(f"{MODEL_NAMES[name]}\nR²={reg_results[name]['R² (test)']:.3f}  "
                 f"RMSE={reg_results[name]['RMSE (test)']:.3f}", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
fig.suptitle("回归模型: 预测值 vs 真实值 (测试集)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "R1_regression_scatter.png")

# ── 图 R2: 回归指标柱状图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["R² (test)", "RMSE (test)", "MAE (test)"]
for i, metric in enumerate(metrics):
    ax = axes[i]
    names = list(reg_results.keys())
    vals = [reg_results[n][metric] for n in names]
    colors = [COLORS[n] for n in names]
    bars = ax.bar([MODEL_NAMES[n] for n in names], vals, color=colors,
                  edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("回归模型性能对比", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "R2_regression_metrics_comparison.png")

# ── 图 R3: 残差分布 ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, (name, (_, y_pred_test)) in enumerate(reg_preds.items()):
    ax = axes[i]
    residuals = y_reg_test - y_pred_test
    ax.hist(residuals, bins=30, color=COLORS[name], edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("残差 (真实 - 预测)", fontsize=10)
    ax.set_ylabel("频数", fontsize=10)
    ax.set_title(f"{MODEL_NAMES[name]}\nμ={residuals.mean():.3f}, σ={residuals.std():.3f}",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("回归模型残差分布 (测试集)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "R3_residual_distribution.png")

# ══════════════════════════════════════════
#  Part B: 分类任务
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part B: 分类任务 (Active / Inactive)")
log("=" * 60)

cls_results = {}
cls_preds = {}

for name, model in cls_models.items():
    log(f"\n  训练 {MODEL_NAMES[name]}...")
    t0 = time.time()
    model.fit(X_train, y_cls_train)
    train_time = time.time() - t0

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_cls_test, y_pred_test)
    auc = roc_auc_score(y_cls_test, y_prob_test)
    f1 = f1_score(y_cls_test, y_pred_test)
    prec = precision_score(y_cls_test, y_pred_test)
    rec = recall_score(y_cls_test, y_pred_test)

    cls_results[name] = {
        "AUC": auc, "Accuracy": acc, "F1": f1,
        "Precision": prec, "Recall": rec, "Time (s)": train_time,
    }
    cls_preds[name] = (y_pred_test, y_prob_test)

    log(f"    AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  "
        f"Prec={prec:.4f}  Rec={rec:.4f}  [{train_time:.1f}s]")

# ── 分类结果表 ──
log("\n  --- 分类结果汇总 ---")
df_cls = pd.DataFrame(cls_results).T
log(df_cls.to_string())

# ── 图 C1: ROC 曲线 ──
fig, ax = plt.subplots(figsize=(8, 8))
for name, (_, y_prob_test) in cls_preds.items():
    fpr, tpr, _ = roc_curve(y_cls_test, y_prob_test)
    auc_val = cls_results[name]["AUC"]
    ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
            label=f"{MODEL_NAMES[name]} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC 曲线对比", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.3)
ax.set_aspect("equal")
save_fig(fig, "C1_roc_curves.png")

# ── 图 C2: 混淆矩阵 ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, (name, (y_pred_test, _)) in enumerate(cls_preds.items()):
    ax = axes[i]
    cm = confusion_matrix(y_cls_test, y_pred_test)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"])
    ax.set_yticklabels(["Inactive", "Active"])
    ax.set_xlabel("预测", fontsize=11)
    ax.set_ylabel("真实", fontsize=11)
    ax.set_title(f"{MODEL_NAMES[name]}\nAcc={cls_results[name]['Accuracy']:.3f}", fontsize=11)
    for (row, col), val in np.ndenumerate(cm):
        ax.text(col, row, str(val), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if val > cm.max() / 2 else "black")
fig.suptitle("混淆矩阵 (测试集)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "C2_confusion_matrices.png")

# ── 图 C3: 分类指标柱状图 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(["AUC", "Accuracy", "F1"]):
    ax = axes[i]
    names = list(cls_results.keys())
    vals = [cls_results[n][metric] for n in names]
    colors = [COLORS[n] for n in names]
    bars = ax.bar([MODEL_NAMES[n] for n in names], vals, color=colors,
                  edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("分类模型性能对比", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "C3_classification_metrics_comparison.png")

# ══════════════════════════════════════════
#  Part C: 特征重要性分析 (RF)
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part C: 特征重要性分析 (Random Forest)")
log("=" * 60)

feat_names_path = os.path.join(PROJECT_DIR, "环节三：特征工程", "feature_names.txt")
with open(feat_names_path, "r", encoding="utf-8") as f:
    feature_names = [line.strip() for line in f.readlines()]

rf_reg = reg_models["RF"]
importances = rf_reg.feature_importances_

# Top 30 重要特征
top_idx = np.argsort(importances)[::-1][:30]
top_names = [feature_names[i] for i in top_idx]
top_vals = importances[top_idx]

log(f"  Top 10 重要特征:")
for i in range(min(10, len(top_names))):
    log(f"    {i+1:2d}. {top_names[i]}: {top_vals[i]:.4f}")

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = range(len(top_names))
colors = ["#3498db" if not n.startswith("Morgan_") else "#95a5a6" for n in top_names]
ax.barh(y_pos, top_vals[::-1], color=colors[::-1], edgecolor="white", height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_names[::-1], fontsize=9)
ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
ax.set_title("Top 30 特征重要性 (Random Forest 回归)", fontsize=14, fontweight="bold")
legend_elements = [
    Line2D([0], [0], color="#3498db", lw=8, label="分子描述符"),
    Line2D([0], [0], color="#95a5a6", lw=8, label="Morgan 指纹位"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_fig(fig, "F1_feature_importance_top30.png")

# ── 描述符 vs 指纹 贡献比 ──
desc_importance = importances[:n_desc].sum()
fp_importance = importances[n_desc:].sum()
log(f"\n  描述符总贡献: {desc_importance:.4f} ({desc_importance/(desc_importance+fp_importance):.1%})")
log(f"  指纹总贡献:   {fp_importance:.4f} ({fp_importance/(desc_importance+fp_importance):.1%})")

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie([desc_importance, fp_importance],
       labels=[f"描述符 ({desc_importance:.1%})", f"指纹 ({fp_importance:.1%})"],
       colors=["#3498db", "#95a5a6"],
       autopct="%1.1f%%", startangle=90,
       textprops={"fontsize": 13},
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("描述符 vs 指纹 特征贡献比", fontsize=14, fontweight="bold")
save_fig(fig, "F2_descriptor_vs_fingerprint_importance.png")

# ══════════════════════════════════════════
#  Part D: 5-fold 交叉验证 (在 train+val 上)
# ══════════════════════════════════════════
log()
log("=" * 60)
log("Part D: 5-fold 交叉验证")
log("=" * 60)

X_trainval = np.vstack([X_train, X_val])
y_reg_trainval = np.concatenate([y_reg_train, y_reg_val])
y_cls_trainval = np.concatenate([y_cls_train, y_cls_val])

cv_results_reg = {}
cv_results_cls = {}

# 交叉验证用轻量参数
cv_reg_models = {
    "MLP": MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300,
                         early_stopping=True, random_state=42, verbose=False),
    "RF": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "HGBT": HistGradientBoostingRegressor(max_iter=200, early_stopping=True, random_state=42),
    "SVM": SVR(kernel="rbf", C=10.0),
}
cv_cls_models = {
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                          early_stopping=True, random_state=42, verbose=False),
    "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  n_jobs=-1, random_state=42),
    "HGBT": HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                            early_stopping=True, random_state=42),
    "SVM": SVC(kernel="rbf", C=10.0, class_weight="balanced",
               probability=True, random_state=42),
}

for name in ["MLP", "RF", "HGBT", "SVM"]:
    log(f"\n  {MODEL_NAMES[name]} 5-fold CV...")

    # 回归 CV
    scores_r2 = cross_val_score(cv_reg_models[name], X_trainval, y_reg_trainval,
                                 cv=5, scoring="r2", n_jobs=-1)
    scores_neg_rmse = cross_val_score(cv_reg_models[name], X_trainval, y_reg_trainval,
                                       cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_results_reg[name] = {
        "R² mean": scores_r2.mean(), "R² std": scores_r2.std(),
        "RMSE mean": -scores_neg_rmse.mean(), "RMSE std": scores_neg_rmse.std(),
    }
    log(f"    回归 R²: {scores_r2.mean():.4f} ± {scores_r2.std():.4f}")

    # 分类 CV
    scores_auc = cross_val_score(cv_cls_models[name], X_trainval, y_cls_trainval,
                                  cv=5, scoring="roc_auc", n_jobs=-1)
    scores_f1 = cross_val_score(cv_cls_models[name], X_trainval, y_cls_trainval,
                                 cv=5, scoring="f1", n_jobs=-1)
    cv_results_cls[name] = {
        "AUC mean": scores_auc.mean(), "AUC std": scores_auc.std(),
        "F1 mean": scores_f1.mean(), "F1 std": scores_f1.std(),
    }
    log(f"    分类 AUC: {scores_auc.mean():.4f} ± {scores_auc.std():.4f}")

# ── 图 CV1: 交叉验证箱线图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 回归 R² CV
ax = axes[0]
names = list(cv_results_reg.keys())
means = [cv_results_reg[n]["R² mean"] for n in names]
stds = [cv_results_reg[n]["R² std"] for n in names]
colors = [COLORS[n] for n in names]
bars = ax.bar([MODEL_NAMES[n] for n in names], means, yerr=stds,
              color=colors, edgecolor="white", width=0.6, capsize=5)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
            f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
ax.set_title("5-Fold CV: 回归 R²", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

# 分类 AUC CV
ax = axes[1]
means = [cv_results_cls[n]["AUC mean"] for n in names]
stds = [cv_results_cls[n]["AUC std"] for n in names]
bars = ax.bar([MODEL_NAMES[n] for n in names], means, yerr=stds,
              color=colors, edgecolor="white", width=0.6, capsize=5)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
            f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
ax.set_title("5-Fold CV: 分类 AUC", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("5-fold 交叉验证结果", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "CV1_cross_validation.png")

# ══════════════════════════════════════════
#  总结
# ══════════════════════════════════════════
log()
log("=" * 60)
log("总结")
log("=" * 60)

best_reg = max(reg_results, key=lambda k: reg_results[k]["R² (test)"])
best_cls = max(cls_results, key=lambda k: cls_results[k]["AUC"])

log(f"  回归最佳: {MODEL_NAMES[best_reg]} (R²={reg_results[best_reg]['R² (test)']:.4f})")
log(f"  分类最佳: {MODEL_NAMES[best_cls]} (AUC={cls_results[best_cls]['AUC']:.4f})")
log()
log(f"  共生成 {len(os.listdir(FIG_DIR))} 张图表 → {FIG_DIR}")

# 保存报告
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
log(f"  ✓ {os.path.basename(REPORT_FILE)}")
log()
log("模型训练与评估完成!")
