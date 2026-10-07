"""
环节二 · Step 2：探索性数据分析 (EDA)
====================================
输入:  清洗后数据.xlsx
输出:  figures/ 目录下的多张可视化图表

图表清单:
  01 - pChEMBL 值分布直方图
  02 - 活性/非活性类别饼图
  03 - 分子量 vs pChEMBL 散点图
  04 - AlogP vs pChEMBL 散点图
  05 - Lipinski 类药五规则分析
  06 - 关键理化性质箱线图 (按活性分组)
  07 - 描述符相关性热力图 (Top 30)
  08 - 描述符与 pChEMBL 相关性条形图 (Top 20)
  09 - Morgan 指纹 t-SNE 可视化
  10 - 描述符 PCA 累计方差解释图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互后端, 适合脚本运行
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ── 中文字体配置 ──
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

# ── 路径配置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "清洗后数据.xlsx")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 颜色方案 ──
COLOR_ACTIVE = "#2ecc71"
COLOR_INACTIVE = "#e74c3c"
COLOR_MAIN = "#3498db"
PALETTE = [COLOR_INACTIVE, COLOR_ACTIVE]

# ══════════════════════════════════════════
#  读取数据
# ══════════════════════════════════════════
print("正在读取清洗后数据...")
df = pd.read_excel(INPUT_FILE, engine="openpyxl")
print(f"  数据维度: {df.shape[0]} × {df.shape[1]}")

# 列分组
info_cols = ["Molecule ChEMBL ID", "Smiles", "Molecular Weight", "AlogP",
             "Standard Type", "Standard Value", "Standard Units", "pChEMBL Value"]
fp_cols = [c for c in df.columns if c.startswith("Morgan_")]
desc_cols = [c for c in df.columns if c not in info_cols
             and not c.startswith("Morgan_") and c != "Activity_Label"]

THRESHOLD = 6.0  # 与清洗脚本一致


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════
#  图 01: pChEMBL 值分布直方图
# ══════════════════════════════════════════
print("\n绘制图表...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df["pChEMBL Value"], bins=50, color=COLOR_MAIN, edgecolor="white",
        alpha=0.85, linewidth=0.5)
ax.axvline(x=THRESHOLD, color=COLOR_INACTIVE, linestyle="--", linewidth=2,
           label=f"活性阈值 (pChEMBL = {THRESHOLD})")
ax.set_xlabel("pChEMBL Value", fontsize=13)
ax.set_ylabel("分子数量", fontsize=13)
ax.set_title("pChEMBL Value 分布", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "01_pchembl_distribution.png")

# ══════════════════════════════════════════
#  图 02: 活性/非活性类别饼图
# ══════════════════════════════════════════
n_active = (df["Activity_Label"] == 1).sum()
n_inactive = (df["Activity_Label"] == 0).sum()

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    [n_active, n_inactive],
    labels=[f"Active (n={n_active})", f"Inactive (n={n_inactive})"],
    colors=[COLOR_ACTIVE, COLOR_INACTIVE],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 13},
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts:
    t.set_fontsize(14)
    t.set_fontweight("bold")
ax.set_title("活性 / 非活性分子分布", fontsize=15, fontweight="bold")
save_fig(fig, "02_activity_pie.png")

# ══════════════════════════════════════════
#  图 03: 分子量 vs pChEMBL 散点图
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df["Molecular Weight"], df["pChEMBL Value"],
    c=df["Activity_Label"], cmap=matplotlib.colors.ListedColormap(PALETTE),
    alpha=0.5, s=20, edgecolors="none"
)
ax.axhline(y=THRESHOLD, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("Molecular Weight (Da)", fontsize=13)
ax.set_ylabel("pChEMBL Value", fontsize=13)
ax.set_title("分子量 vs 生物活性", fontsize=15, fontweight="bold")
# 自定义图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ACTIVE,
           markersize=8, label='Active'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_INACTIVE,
           markersize=8, label='Inactive'),
]
ax.legend(handles=legend_elements, fontsize=11)
ax.grid(alpha=0.3)
save_fig(fig, "03_mw_vs_pchembl.png")

# ══════════════════════════════════════════
#  图 04: AlogP vs pChEMBL 散点图
# ══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
mask_valid = df["AlogP"].notna()
ax.scatter(
    df.loc[mask_valid, "AlogP"], df.loc[mask_valid, "pChEMBL Value"],
    c=df.loc[mask_valid, "Activity_Label"],
    cmap=matplotlib.colors.ListedColormap(PALETTE),
    alpha=0.5, s=20, edgecolors="none"
)
ax.axhline(y=THRESHOLD, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(x=5, color="orange", linestyle="--", linewidth=1, alpha=0.7,
           label="Lipinski AlogP ≤ 5")
ax.set_xlabel("AlogP", fontsize=13)
ax.set_ylabel("pChEMBL Value", fontsize=13)
ax.set_title("脂溶性 vs 生物活性", fontsize=15, fontweight="bold")
ax.legend(handles=legend_elements + [
    Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='Lipinski AlogP ≤ 5')
], fontsize=10)
ax.grid(alpha=0.3)
save_fig(fig, "04_alogp_vs_pchembl.png")

# ══════════════════════════════════════════
#  图 05: Lipinski 类药五规则
# ══════════════════════════════════════════
# 使用描述符中的 RDKit 版本
lipinski_map = {
    "分子量 ≤ 500": ("MolWt", 500),
    "LogP ≤ 5": ("MolLogP", 5),
    "氢键供体 ≤ 5": ("NumHDonors", 5),
    "氢键受体 ≤ 10": ("NumHAcceptors", 10),
}

lipinski_data = {}
for label, (col, threshold) in lipinski_map.items():
    if col in df.columns:
        lipinski_data[label] = {
            "pass": (df[col] <= threshold).sum(),
            "fail": (df[col] > threshold).sum(),
        }

if lipinski_data:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (label, data) in enumerate(lipinski_data.items()):
        ax = axes[i]
        bars = ax.bar(["通过", "不通过"], [data["pass"], data["fail"]],
                      color=[COLOR_ACTIVE, COLOR_INACTIVE], edgecolor="white", width=0.5)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylabel("分子数量", fontsize=11)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                    f"{int(height)}", ha="center", va="bottom", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Lipinski 类药五规则分析", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "05_lipinski_rules.png")

# ══════════════════════════════════════════
#  图 06: 关键理化性质箱线图 (按活性分组)
# ══════════════════════════════════════════
physico_cols = []
col_labels = []
for col, label in [("MolWt", "分子量"), ("MolLogP", "LogP"),
                    ("TPSA", "TPSA"), ("NumHDonors", "氢键供体"),
                    ("NumHAcceptors", "氢键受体"), ("NumRotatableBonds", "可旋转键")]:
    if col in df.columns:
        physico_cols.append(col)
        col_labels.append(label)

if physico_cols:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, (col, label) in enumerate(zip(physico_cols, col_labels)):
        ax = axes[i]
        data_active = df.loc[df["Activity_Label"] == 1, col].dropna()
        data_inactive = df.loc[df["Activity_Label"] == 0, col].dropna()
        bp = ax.boxplot(
            [data_inactive, data_active],
            labels=["Inactive", "Active"],
            patch_artist=True,
            widths=0.5,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )
        bp["boxes"][0].set_facecolor(COLOR_INACTIVE)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLOR_ACTIVE)
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # 隐藏多余的 axes
    for j in range(len(physico_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("理化性质分布 (按活性分组)", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "06_physicochemical_boxplots.png")

# ══════════════════════════════════════════
#  图 07: 描述符相关性热力图 (Top 30 与 pChEMBL 最相关的)
# ══════════════════════════════════════════
print("  计算描述符相关性...")

# 选出与 pChEMBL 相关性最强的 30 个描述符
corr_with_target = df[desc_cols].corrwith(df["pChEMBL Value"]).abs().dropna()
top30_descs = corr_with_target.nlargest(30).index.tolist()

corr_matrix = df[top30_descs].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            annot=False, vmin=-1, vmax=1)
ax.set_title("Top 30 描述符相关性热力图\n(与 pChEMBL 相关性最强)", fontsize=14, fontweight="bold")
ax.tick_params(axis="both", labelsize=8)
fig.tight_layout()
save_fig(fig, "07_descriptor_correlation_heatmap.png")

# ══════════════════════════════════════════
#  图 08: 描述符与 pChEMBL 相关性条形图
# ══════════════════════════════════════════
corr_signed = df[desc_cols].corrwith(df["pChEMBL Value"]).dropna()
top20_pos = corr_signed.nlargest(10)
top20_neg = corr_signed.nsmallest(10)
top20 = pd.concat([top20_pos, top20_neg]).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = [COLOR_INACTIVE if v < 0 else COLOR_ACTIVE for v in top20.values]
ax.barh(range(len(top20)), top20.values, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=10)
ax.set_xlabel("Pearson 相关系数", fontsize=13)
ax.set_title("Top 20 描述符与 pChEMBL 的相关性", fontsize=15, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
save_fig(fig, "08_descriptor_pchembl_correlation.png")

# ══════════════════════════════════════════
#  图 09: Morgan 指纹 t-SNE 可视化
# ══════════════════════════════════════════
print("  计算 t-SNE (可能需要较长时间)...")

# 如果数据量大, 随机采样以加速
MAX_TSNE_SAMPLES = 2000
if len(df) > MAX_TSNE_SAMPLES:
    df_sample = df.sample(n=MAX_TSNE_SAMPLES, random_state=42)
    print(f"    采样 {MAX_TSNE_SAMPLES}/{len(df)} 个分子用于 t-SNE")
else:
    df_sample = df

fp_data = df_sample[fp_cols].values.astype(np.float32)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000,
            learning_rate="auto", init="pca")
tsne_result = tsne.fit_transform(fp_data)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 左图: 按分类着色
ax = axes[0]
for label, color, name in [(0, COLOR_INACTIVE, "Inactive"), (1, COLOR_ACTIVE, "Active")]:
    mask = df_sample["Activity_Label"] == label
    ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
               c=color, alpha=0.5, s=15, label=name, edgecolors="none")
ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.set_title("Morgan 指纹 t-SNE (活性分类)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)

# 右图: 按 pChEMBL 连续值着色
ax = axes[1]
sc = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                c=df_sample["pChEMBL Value"].values, cmap="RdYlGn",
                alpha=0.6, s=15, edgecolors="none")
plt.colorbar(sc, ax=ax, label="pChEMBL Value", shrink=0.8)
ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.set_title("Morgan 指纹 t-SNE (pChEMBL 连续值)", fontsize=14, fontweight="bold")

fig.tight_layout()
save_fig(fig, "09_tsne_morgan_fingerprint.png")

# ══════════════════════════════════════════
#  图 10: 描述符 PCA 累计方差解释比
# ══════════════════════════════════════════
print("  计算 PCA...")

scaler = StandardScaler()
desc_scaled = scaler.fit_transform(df[desc_cols].values)

pca_full = PCA(random_state=42)
pca_full.fit(desc_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumvar, 0.95) + 1
n_90 = np.searchsorted(cumvar, 0.90) + 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(cumvar) + 1), cumvar, color=COLOR_MAIN, linewidth=2)
ax.axhline(y=0.95, color=COLOR_INACTIVE, linestyle="--", linewidth=1,
           label=f"95% 方差 (需 {n_95} 个主成分)")
ax.axhline(y=0.90, color="orange", linestyle="--", linewidth=1,
           label=f"90% 方差 (需 {n_90} 个主成分)")
ax.scatter([n_95], [0.95], color=COLOR_INACTIVE, s=60, zorder=5)
ax.scatter([n_90], [0.90], color="orange", s=60, zorder=5)
ax.set_xlabel("主成分数量", fontsize=13)
ax.set_ylabel("累计方差解释比", fontsize=13)
ax.set_title("分子描述符 PCA 累计方差解释", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, min(len(cumvar), 100))
ax.set_ylim(0, 1.05)
save_fig(fig, "10_pca_cumulative_variance.png")

# ══════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════
print()
print("=" * 50)
print("EDA 可视化完成!")
print(f"所有图表保存在: {FIG_DIR}")
print(f"共生成 {len(os.listdir(FIG_DIR))} 张图表")
print("=" * 50)
