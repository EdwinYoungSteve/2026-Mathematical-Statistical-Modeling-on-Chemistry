"""
数据演示：抗癌药物虚拟筛选 Demo
================================
完整流程展示：
  1) 从清洗数据中恢复测试集对应分子信息
  2) 训练最佳集成模型 (Voting)
  3) 对测试集分子进行虚拟筛选打分
  4) 展示个案预测 + 筛选排名 + 可视化
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
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
CLEAN_FILE = os.path.join(PROJECT_DIR, "环节二：数据清洗与EDA", "清洗后数据.xlsx")
NPZ_FILE = os.path.join(PROJECT_DIR, "环节三：特征工程", "特征工程后数据.npz")
FEAT_NAMES_FILE = os.path.join(PROJECT_DIR, "环节三：特征工程", "feature_names.txt")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "active": "#2ecc71", "inactive": "#e74c3c",
    "MLP": "#3498db", "RF": "#2ecc71", "HGBT": "#e67e22", "SVM": "#9b59b6",
    "Voting": "#1abc9c",
}

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════
#  Step 1: 恢复测试集分子信息
# ══════════════════════════════════════════
print("=" * 65)
print("  STEP 1  恢复测试集分子身份信息")
print("=" * 65)

# 读取清洗后数据
df = pd.read_excel(CLEAN_FILE, engine="openpyxl")
print(f"  清洗后数据: {df.shape[0]} 个分子")

# 重做完全相同的特征工程pipeline来恢复索引
info_cols = ["Molecule ChEMBL ID", "Smiles", "Molecular Weight", "AlogP",
             "Standard Type", "Standard Value", "Standard Units",
             "pChEMBL Value", "Activity_Label"]
fp_cols = sorted([c for c in df.columns if c.startswith("Morgan_")])
desc_cols_raw = sorted([c for c in df.columns if c not in info_cols and c not in fp_cols])

# 去冗余描述符 (与环节三完全相同)
corr_matrix = df[desc_cols_raw].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
desc_cols = [c for c in desc_cols_raw if c not in to_drop]

y_cls = df["Activity_Label"].values.astype(np.int32)

# 用相同 random_state 恢复划分索引
indices = np.arange(len(df))
idx_train, idx_tmp = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_cls)
y_cls_tmp = y_cls[idx_tmp]
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=42, stratify=y_cls_tmp)

df_test = df.iloc[idx_test].reset_index(drop=True)
print(f"  测试集: {len(df_test)} 个分子 (Active: {(df_test['Activity_Label']==1).sum()}, "
      f"Inactive: {(df_test['Activity_Label']==0).sum()})")

# ══════════════════════════════════════════
#  Step 2: 加载特征数据 & 训练集成模型
# ══════════════════════════════════════════
print()
print("=" * 65)
print("  STEP 2  训练集成模型 (Voting Ensemble)")
print("=" * 65)

data = np.load(NPZ_FILE)
X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
y_reg_train, y_reg_val, y_reg_test = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]
y_cls_train, y_cls_val, y_cls_test = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]

X_trainval = np.vstack([X_train, X_val])
y_reg_trainval = np.concatenate([y_reg_train, y_reg_val])
y_cls_trainval = np.concatenate([y_cls_train, y_cls_val])

# 训练 Voting 回归
print("  训练回归集成模型...")
voting_reg = VotingRegressor(
    estimators=[
        ("MLP", MLPRegressor(hidden_layer_sizes=(512, 256, 128), activation="relu",
                              solver="adam", alpha=1e-4, learning_rate="adaptive",
                              learning_rate_init=1e-3, max_iter=500, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=20,
                              batch_size=64, random_state=42, verbose=False)),
        ("RF", RandomForestRegressor(n_estimators=500, min_samples_split=5,
                                      min_samples_leaf=2, max_features="sqrt",
                                      n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingRegressor(max_iter=500, max_depth=6,
                                                learning_rate=0.05, min_samples_leaf=10,
                                                l2_regularization=1.0, early_stopping=True,
                                                random_state=42)),
        ("SVM", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
    ],
    n_jobs=-1,
)
voting_reg.fit(X_trainval, y_reg_trainval)
y_pred_reg = voting_reg.predict(X_test)
print(f"  回归 R² = {r2_score(y_reg_test, y_pred_reg):.4f}")

# 训练 Voting 分类
print("  训练分类集成模型...")
voting_cls = VotingClassifier(
    estimators=[
        ("MLP", MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation="relu",
                               solver="adam", alpha=1e-4, learning_rate="adaptive",
                               learning_rate_init=1e-3, max_iter=500, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=20,
                               batch_size=64, random_state=42, verbose=False)),
        ("RF", RandomForestClassifier(n_estimators=500, min_samples_split=5,
                                       min_samples_leaf=2, max_features="sqrt",
                                       class_weight="balanced", n_jobs=-1, random_state=42)),
        ("HGBT", HistGradientBoostingClassifier(max_iter=500, max_depth=6,
                                                  learning_rate=0.05, min_samples_leaf=10,
                                                  l2_regularization=1.0, early_stopping=True,
                                                  class_weight="balanced", random_state=42)),
        ("SVM", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
                     probability=True, random_state=42)),
    ],
    voting="soft", n_jobs=-1,
)
voting_cls.fit(X_trainval, y_cls_trainval)
y_pred_cls = voting_cls.predict(X_test)
y_prob_cls = voting_cls.predict_proba(X_test)[:, 1]
print(f"  分类 AUC = {roc_auc_score(y_cls_test, y_prob_cls):.4f}")
print(f"  分类 Acc = {accuracy_score(y_cls_test, y_pred_cls):.4f}")

# 同时收集各基学习器的单独预测（用于展示）
base_preds_reg = {}
for name, est in voting_reg.named_estimators_.items():
    base_preds_reg[name] = est.predict(X_test)

base_probs_cls = {}
for name, est in voting_cls.named_estimators_.items():
    base_probs_cls[name] = est.predict_proba(X_test)[:, 1]

# ══════════════════════════════════════════
#  Step 3: 组装预测结果
# ══════════════════════════════════════════
print()
print("=" * 65)
print("  STEP 3  虚拟筛选结果")
print("=" * 65)

df_test = df_test.copy()
df_test["Pred_pChEMBL"] = y_pred_reg
df_test["Pred_Label"] = y_pred_cls
df_test["Pred_Prob_Active"] = y_prob_cls
df_test["Actual_pChEMBL"] = y_reg_test
df_test["Actual_Label"] = y_cls_test

# 按预测活性概率排序（虚拟筛选排名）
df_ranked = df_test.sort_values("Pred_Prob_Active", ascending=False).reset_index(drop=True)
df_ranked.index = df_ranked.index + 1  # 排名从1开始
df_ranked.index.name = "Rank"

# 打印前 15 名（最可能有活性的分子）
print("\n  ╔═══ 筛选排名 TOP 15 (最可能具有抗癌活性的分子) ═══╗\n")
print(f"  {'Rank':>4s}  {'ChEMBL ID':>15s}  {'预测pIC50':>8s}  "
      f"{'真实pIC50':>8s}  {'活性概率':>7s}  {'预测':>4s}  {'真实':>4s}  {'✓/✗':>3s}")
print(f"  {'─'*75}")
for i in range(15):
    row = df_ranked.iloc[i]
    correct = "✓" if row["Pred_Label"] == row["Actual_Label"] else "✗"
    pred_lbl = "活性" if row["Pred_Label"] == 1 else "无活性"
    actual_lbl = "活性" if row["Actual_Label"] == 1 else "无活性"
    print(f"  {i+1:4d}  {row['Molecule ChEMBL ID']:>15s}  "
          f"{row['Pred_pChEMBL']:8.3f}  {row['Actual_pChEMBL']:8.3f}  "
          f"{row['Pred_Prob_Active']:7.1%}  {pred_lbl:>4s}  {actual_lbl:>4s}  {correct:>3s}")

# 打印后 15 名（最可能无活性的分子）
print(f"\n  ╔═══ 筛选排名 后15 (最可能无抗癌活性的分子) ═══╗\n")
print(f"  {'Rank':>4s}  {'ChEMBL ID':>15s}  {'预测pIC50':>8s}  "
      f"{'真实pIC50':>8s}  {'活性概率':>7s}  {'预测':>4s}  {'真实':>4s}  {'✓/✗':>3s}")
print(f"  {'─'*75}")
for i in range(len(df_ranked) - 15, len(df_ranked)):
    row = df_ranked.iloc[i]
    correct = "✓" if row["Pred_Label"] == row["Actual_Label"] else "✗"
    pred_lbl = "活性" if row["Pred_Label"] == 1 else "无活性"
    actual_lbl = "活性" if row["Actual_Label"] == 1 else "无活性"
    print(f"  {i+1:4d}  {row['Molecule ChEMBL ID']:>15s}  "
          f"{row['Pred_pChEMBL']:8.3f}  {row['Actual_pChEMBL']:8.3f}  "
          f"{row['Pred_Prob_Active']:7.1%}  {pred_lbl:>4s}  {actual_lbl:>4s}  {correct:>3s}")

# 统计预测准确率
n_correct = (df_test["Pred_Label"] == df_test["Actual_Label"]).sum()
print(f"\n  测试集总体准确率: {n_correct}/{len(df_test)} = {n_correct/len(df_test):.1%}")

# ══════════════════════════════════════════
#  Step 4: 个案深度分析（挑选有代表性的分子）
# ══════════════════════════════════════════
print()
print("=" * 65)
print("  STEP 4  个案深度分析")
print("=" * 65)

# 挑选5个有代表性的分子
cases = []
# 1) 预测最佳命中 (高活性高置信)
top_active = df_ranked.iloc[0]
cases.append(("最佳命中 (高活性)", top_active))

# 2) 排名中间的分子
mid_idx = len(df_ranked) // 2
mid_mol = df_ranked.iloc[mid_idx]
cases.append(("中等活性", mid_mol))

# 3) 预测无活性 (低活性高置信)
bottom_inactive = df_ranked.iloc[-1]
cases.append(("预测无活性", bottom_inactive))

# 4) 预测正确的困难案例 (预测概率在0.4-0.6之间但正确的)
borderline = df_ranked[(df_ranked["Pred_Prob_Active"] > 0.35) &
                        (df_ranked["Pred_Prob_Active"] < 0.65) &
                        (df_ranked["Pred_Label"] == df_ranked["Actual_Label"])]
if len(borderline) > 0:
    cases.append(("边界案例(正确)", borderline.iloc[0]))

# 5) 预测错误的案例
wrong = df_ranked[df_ranked["Pred_Label"] != df_ranked["Actual_Label"]]
if len(wrong) > 0:
    cases.append(("预测失误", wrong.iloc[0]))

for title, mol in cases:
    print(f"\n  ┌── {title} ──")
    print(f"  │ ChEMBL ID:   {mol['Molecule ChEMBL ID']}")
    smiles = mol['Smiles']
    if len(smiles) > 60:
        smiles = smiles[:57] + "..."
    print(f"  │ SMILES:      {smiles}")
    print(f"  │ 分子量:       {mol['Molecular Weight']:.1f} Da")
    print(f"  │ AlogP:       {mol['AlogP']:.2f}")
    print(f"  │ ─────────────────────────────────")
    print(f"  │ pIC50 真实:   {mol['Actual_pChEMBL']:.3f}")
    print(f"  │ pIC50 预测:   {mol['Pred_pChEMBL']:.3f}  (误差: {abs(mol['Pred_pChEMBL']-mol['Actual_pChEMBL']):.3f})")
    print(f"  │ 活性概率:     {mol['Pred_Prob_Active']:.1%}")
    actual = "✅ 活性" if mol['Actual_Label'] == 1 else "❌ 无活性"
    pred = "✅ 活性" if mol['Pred_Label'] == 1 else "❌ 无活性"
    match = "✓ 正确" if mol['Pred_Label'] == mol['Actual_Label'] else "✗ 错误"
    print(f"  │ 真实标签:     {actual}")
    print(f"  │ 预测标签:     {pred}  ({match})")
    print(f"  └────────────────────────────────────")

# ══════════════════════════════════════════
#  Step 5: 可视化
# ══════════════════════════════════════════
print()
print("=" * 65)
print("  STEP 5  生成可视化图表")
print("=" * 65)

# ── 图1: 虚拟筛选排名瀑布图 ──
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(df_ranked))
colors_bar = [COLORS["active"] if lbl == 1 else COLORS["inactive"]
              for lbl in df_ranked["Actual_Label"]]
ax.bar(x, df_ranked["Pred_Prob_Active"], color=colors_bar, width=1.0, alpha=0.8)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="分类阈值 (0.5)")
ax.set_xlabel("筛选排名（按预测活性概率降序）", fontsize=12)
ax.set_ylabel("预测活性概率", fontsize=12)
ax.set_title("虚拟筛选瀑布图：183 个测试分子按预测活性排名", fontsize=14, fontweight="bold")
ax.set_xlim(-1, len(df_ranked))
ax.set_ylim(0, 1.05)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["active"], label="真实活性"),
    Patch(facecolor=COLORS["inactive"], label="真实无活性"),
    plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="分类阈值"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=11)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "D1_screening_waterfall.png")

# ── 图2: 预测 vs 真实 pChEMBL 散点（带分子标注）──
fig, ax = plt.subplots(figsize=(9, 9))
scatter_colors = [COLORS["active"] if lbl == 1 else COLORS["inactive"]
                  for lbl in df_test["Actual_Label"]]
ax.scatter(df_test["Actual_pChEMBL"], df_test["Pred_pChEMBL"],
           c=scatter_colors, alpha=0.6, s=40, edgecolors="white", linewidth=0.5)

# 对角线
lo = min(df_test["Actual_pChEMBL"].min(), df_test["Pred_pChEMBL"].min()) - 0.3
hi = max(df_test["Actual_pChEMBL"].max(), df_test["Pred_pChEMBL"].max()) + 0.3
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1.5)

# 标注个案分子
for title, mol in cases[:3]:
    ax.annotate(mol["Molecule ChEMBL ID"],
                xy=(mol["Actual_pChEMBL"], mol["Pred_pChEMBL"]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

ax.set_xlabel("真实 pChEMBL Value (pIC50)", fontsize=12)
ax.set_ylabel("Voting 集成预测 pChEMBL Value", fontsize=12)
ax.set_title("集成模型预测 vs 真实值（测试集 183 分子）", fontsize=14, fontweight="bold")
r2 = r2_score(df_test["Actual_pChEMBL"], df_test["Pred_pChEMBL"])
ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=13,
        fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
legend_elements = [
    Patch(facecolor=COLORS["active"], label="真实活性 (pChEMBL ≥ 6.0)"),
    Patch(facecolor=COLORS["inactive"], label="真实无活性 (pChEMBL < 6.0)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
save_fig(fig, "D2_prediction_scatter_annotated.png")

# ── 图3: 各基学习器 vs 集成对比（选3个分子展示）──
demo_molecules = [cases[0], cases[1], cases[2]]
model_names_ordered = ["MLP", "RF", "HGBT", "SVM"]

fig, axes = plt.subplots(1, len(demo_molecules), figsize=(6 * len(demo_molecules), 7))
if len(demo_molecules) == 1:
    axes = [axes]

for ax, (title, mol) in zip(axes, demo_molecules):
    # 找到该分子在 df_test 中的原始行号（非排名后的）
    mol_id = mol["Molecule ChEMBL ID"]
    test_idx = df_test[df_test["Molecule ChEMBL ID"] == mol_id].index[0]

    actual = mol["Actual_pChEMBL"]
    preds = {name: base_preds_reg[name][test_idx] for name in model_names_ordered}
    preds["Voting"] = mol["Pred_pChEMBL"]

    names = model_names_ordered + ["Voting"]
    vals = [preds[n] for n in names]
    colors_b = [COLORS.get(n, "#333") for n in names]

    bars = ax.barh(range(len(names)), vals, color=colors_b, height=0.6, alpha=0.85)
    ax.axvline(actual, color="red", linestyle="-", linewidth=2.5, label=f"真实值 {actual:.2f}")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([{"MLP": "MLP", "RF": "RF", "HGBT": "HGBT",
                          "SVM": "SVM", "Voting": "Voting\n(集成)"}[n] for n in names],
                        fontsize=10)
    ax.set_xlabel("pChEMBL Value (pIC50)", fontsize=10)
    ax.set_title(f"{title}\n{mol_id}", fontsize=11, fontweight="bold")

    # 标注数值
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)

    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("各模型单独预测 vs 集成预测 vs 真实值", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "D3_model_comparison_cases.png")

# ── 图4: 预测概率分布 ──
fig, ax = plt.subplots(figsize=(10, 6))
probs_active = df_test[df_test["Actual_Label"] == 1]["Pred_Prob_Active"]
probs_inactive = df_test[df_test["Actual_Label"] == 0]["Pred_Prob_Active"]

ax.hist(probs_active, bins=25, alpha=0.7, color=COLORS["active"],
        label=f"真实活性 (n={len(probs_active)})", edgecolor="white")
ax.hist(probs_inactive, bins=25, alpha=0.7, color=COLORS["inactive"],
        label=f"真实无活性 (n={len(probs_inactive)})", edgecolor="white")
ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="分类阈值")
ax.set_xlabel("预测活性概率", fontsize=12)
ax.set_ylabel("分子数量", fontsize=12)
ax.set_title("预测概率分布（活性 vs 无活性分子）", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "D4_probability_distribution.png")

# ── 图5: 筛选效率曲线 (Enrichment) ──
fig, ax = plt.subplots(figsize=(10, 7))
n_total = len(df_ranked)
n_active = (df_ranked["Actual_Label"] == 1).sum()
x_pct = np.arange(1, n_total + 1) / n_total * 100
y_recall = np.cumsum(df_ranked["Actual_Label"].values) / n_active * 100

ax.plot(x_pct, y_recall, "b-", linewidth=2.5, label="集成模型筛选", color=COLORS["Voting"])
ax.plot([0, 100], [0, 100], "k--", alpha=0.4, lw=1.5, label="随机筛选")
ax.fill_between(x_pct, y_recall, x_pct, alpha=0.15, color=COLORS["Voting"])

# 标注关键节点
for pct in [10, 20, 50]:
    idx = int(n_total * pct / 100) - 1
    recall = y_recall[idx]
    ax.plot(pct, recall, "ro", markersize=8, zorder=5)
    ax.annotate(f"筛选 {pct}% → 命中 {recall:.1f}%",
                xy=(pct, recall), xytext=(pct + 5, recall - 8),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

ax.set_xlabel("筛选比例 (% of 全部分子)", fontsize=12)
ax.set_ylabel("活性分子累计召回率 (%)", fontsize=12)
ax.set_title("虚拟筛选富集曲线 (Enrichment Curve)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
save_fig(fig, "D5_enrichment_curve.png")

# ── 图6: 分子性质 vs 预测准确性 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("分子理化性质 vs 预测表现", fontsize=14, fontweight="bold")

df_test["Abs_Error"] = abs(df_test["Pred_pChEMBL"] - df_test["Actual_pChEMBL"])
df_test["Correct"] = (df_test["Pred_Label"] == df_test["Actual_Label"]).astype(int)

# (a) 分子量 vs 绝对误差
ax = axes[0]
ax.scatter(df_test["Molecular Weight"], df_test["Abs_Error"], alpha=0.5,
           c=[COLORS["active"] if c else COLORS["inactive"] for c in df_test["Correct"]],
           s=30, edgecolors="white", linewidth=0.3)
ax.set_xlabel("分子量 (Da)", fontsize=11)
ax.set_ylabel("预测绝对误差", fontsize=11)
ax.set_title("分子量 vs 预测误差", fontsize=12)
ax.grid(True, alpha=0.3)

# (b) AlogP vs 绝对误差
ax = axes[1]
ax.scatter(df_test["AlogP"], df_test["Abs_Error"], alpha=0.5,
           c=[COLORS["active"] if c else COLORS["inactive"] for c in df_test["Correct"]],
           s=30, edgecolors="white", linewidth=0.3)
ax.set_xlabel("AlogP", fontsize=11)
ax.set_ylabel("预测绝对误差", fontsize=11)
ax.set_title("AlogP vs 预测误差", fontsize=12)
ax.grid(True, alpha=0.3)

# (c) 实际 pChEMBL vs 误差
ax = axes[2]
ax.scatter(df_test["Actual_pChEMBL"], df_test["Abs_Error"], alpha=0.5,
           c=[COLORS["active"] if c else COLORS["inactive"] for c in df_test["Correct"]],
           s=30, edgecolors="white", linewidth=0.3)
ax.set_xlabel("真实 pChEMBL", fontsize=11)
ax.set_ylabel("预测绝对误差", fontsize=11)
ax.set_title("真实活性 vs 预测误差", fontsize=12)
ax.grid(True, alpha=0.3)

legend_elements = [
    Patch(facecolor=COLORS["active"], label="分类正确"),
    Patch(facecolor=COLORS["inactive"], label="分类错误"),
]
axes[2].legend(handles=legend_elements, fontsize=10)
plt.tight_layout()
save_fig(fig, "D6_property_vs_error.png")


# ══════════════════════════════════════════
#  Step 6: 导出筛选报告
# ══════════════════════════════════════════
print()
print("=" * 65)
print("  STEP 6  导出筛选结果")
print("=" * 65)

# 导出完整排名表
output_cols = ["Molecule ChEMBL ID", "Smiles", "Molecular Weight", "AlogP",
               "Actual_pChEMBL", "Pred_pChEMBL", "Actual_Label", "Pred_Label",
               "Pred_Prob_Active"]
df_export = df_ranked[output_cols].copy()
df_export.columns = ["ChEMBL_ID", "SMILES", "MW", "AlogP",
                      "Actual_pIC50", "Pred_pIC50", "Actual_Active",
                      "Pred_Active", "Prob_Active"]
export_path = os.path.join(SCRIPT_DIR, "虚拟筛选结果.csv")
df_export.to_csv(export_path, index=True, encoding="utf-8-sig")
print(f"  筛选结果已导出: {export_path}")

# 汇总统计
print(f"\n  ┌── 筛选效率汇总 ──")
for pct in [5, 10, 20, 30, 50]:
    n_screen = int(len(df_ranked) * pct / 100)
    n_hit = df_ranked.iloc[:n_screen]["Actual_Label"].sum()
    ef = (n_hit / n_screen) / (n_active / n_total) if n_screen > 0 else 0
    print(f"  │ 筛选 Top {pct:2d}% ({n_screen:3d} 个): "
          f"命中 {n_hit:3d} 个活性分子, 召回率 {n_hit/n_active:.1%}, "
          f"富集因子 EF={ef:.2f}")
print(f"  └──────────────────────────────")
print(f"\n  全部演示完成！共生成 6 张图表于 {FIG_DIR}")
