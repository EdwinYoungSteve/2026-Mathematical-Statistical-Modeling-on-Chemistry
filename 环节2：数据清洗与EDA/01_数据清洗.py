"""
环节二 · Step 1：数据清洗
========================
输入:  环节一/描述符与指纹合并.xlsx
输出:
  - 清洗后数据.xlsx          : 去重、去缺失、去异常后的完整数据
  - 清洗后数据_仅特征.xlsx    : 仅保留特征列 + 标签列（直接供模型训练）
  - 清洗报告.txt             : 每步清洗的统计信息
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── 路径配置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(PROJECT_DIR, "环节一：数据收集与整理", "描述符与指纹合并.xlsx")
OUTPUT_FULL = os.path.join(SCRIPT_DIR, "清洗后数据.xlsx")
OUTPUT_FEATURES = os.path.join(SCRIPT_DIR, "清洗后数据_仅特征.xlsx")
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "清洗报告.txt")

# 将日志同时输出到控制台和文件
report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ══════════════════════════════════════════
#  1. 读取数据
# ══════════════════════════════════════════
log("=" * 60)
log("步骤 1: 读取数据")
log("=" * 60)

df = pd.read_excel(INPUT_FILE, engine="openpyxl")
log(f"  原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

# 识别列组
info_cols = ["Molecule ChEMBL ID", "Smiles", "Molecular Weight", "AlogP",
             "Standard Type", "Standard Value", "Standard Units", "pChEMBL Value"]
desc_cols = [c for c in df.columns if c not in info_cols and not c.startswith("Morgan_")]
fp_cols = [c for c in df.columns if c.startswith("Morgan_")]

log(f"  信息列: {len(info_cols)}")
log(f"  描述符列: {len(desc_cols)}")
log(f"  指纹列: {len(fp_cols)}")

# ══════════════════════════════════════════
#  2. 去除 pChEMBL Value 缺失的行
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 2: 去除 pChEMBL Value 缺失值")
log("=" * 60)

n_before = len(df)
df = df.dropna(subset=["pChEMBL Value"]).copy()
df["pChEMBL Value"] = pd.to_numeric(df["pChEMBL Value"], errors="coerce")
df = df.dropna(subset=["pChEMBL Value"])
n_after = len(df)
log(f"  去除 pChEMBL 缺失: {n_before} → {n_after} (移除 {n_before - n_after} 行)")

# ══════════════════════════════════════════
#  3. 去除异常值 (pChEMBL 范围检查)
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 3: 去除活性值异常样本")
log("=" * 60)

# pChEMBL 值通常在 2~12 之间, 超出此范围的可能是数据错误
n_before = len(df)
mask_valid = (df["pChEMBL Value"] >= 3.0) & (df["pChEMBL Value"] <= 12.0)
df = df[mask_valid].copy()
n_after = len(df)
log(f"  过滤 pChEMBL ∈ [3.0, 12.0]: {n_before} → {n_after} (移除 {n_before - n_after} 行)")

# Standard Value 极端值检查 (IC50 > 100 μM = 100000 nM 通常无意义)
n_before = len(df)
df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
mask_sv = df["Standard Value"] <= 100000
df = df[mask_sv].copy()
n_after = len(df)
log(f"  过滤 IC50 ≤ 100 μM: {n_before} → {n_after} (移除 {n_before - n_after} 行)")

# ══════════════════════════════════════════
#  4. 处理重复分子 (同一分子取 pChEMBL 中位数)
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 4: 处理重复分子")
log("=" * 60)

n_before = len(df)
n_unique_before = df["Molecule ChEMBL ID"].nunique()

# 对数值列按分子 ID 取中位数聚合
numeric_cols = desc_cols + ["pChEMBL Value", "Standard Value", "Molecular Weight", "AlogP"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

# 指纹列对同一分子应完全相同 (来自同一 SMILES), 取 first
non_numeric_agg = {c: "first" for c in ["Smiles", "Standard Type", "Standard Units"] + fp_cols
                   if c in df.columns}
numeric_agg = {c: "median" for c in numeric_cols if c in df.columns}
agg_dict = {**non_numeric_agg, **numeric_agg}

df_dedup = df.groupby("Molecule ChEMBL ID", as_index=False).agg(agg_dict)
n_unique_after = len(df_dedup)

log(f"  聚合前: {n_before} 行, {n_unique_before} 个唯一分子")
log(f"  聚合后: {n_unique_after} 行 (每个分子一条记录, pChEMBL 取中位数)")

df = df_dedup.copy()

# ══════════════════════════════════════════
#  5. 处理描述符中的缺失值和无穷值
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 5: 处理描述符缺失值 / 无穷值")
log("=" * 60)

# 替换 inf 为 NaN
df[desc_cols] = df[desc_cols].replace([np.inf, -np.inf], np.nan)

# 统计各描述符缺失率
desc_nan_rate = df[desc_cols].isna().mean()
high_nan_descs = desc_nan_rate[desc_nan_rate > 0.3].index.tolist()
log(f"  缺失率 > 30% 的描述符: {len(high_nan_descs)} 个")
if high_nan_descs:
    for d in high_nan_descs:
        log(f"    - {d}: {desc_nan_rate[d]:.1%}")

# 移除缺失率过高的描述符列
df = df.drop(columns=high_nan_descs)
desc_cols = [c for c in desc_cols if c not in high_nan_descs]
log(f"  移除后剩余描述符: {len(desc_cols)} 个")

# 剩余缺失值: 用该列中位数填充
n_filled = df[desc_cols].isna().sum().sum()
for c in desc_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())
log(f"  中位数填充: 共填充 {n_filled} 个缺失值")

# AlogP 缺失处理 (可用描述符中的 MolLogP 替代)
if df["AlogP"].isna().any():
    n_alogp_nan = df["AlogP"].isna().sum()
    if "MolLogP" in df.columns:
        df["AlogP"] = df["AlogP"].fillna(df["MolLogP"])
        log(f"  AlogP 缺失 {n_alogp_nan} 个, 已用 MolLogP 替代")
    else:
        df["AlogP"] = df["AlogP"].fillna(df["AlogP"].median())
        log(f"  AlogP 缺失 {n_alogp_nan} 个, 已用中位数填充")

# ══════════════════════════════════════════
#  6. 去除零方差特征
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 6: 去除零方差特征")
log("=" * 60)

feature_cols = desc_cols + fp_cols
feature_cols = [c for c in feature_cols if c in df.columns]

variances = df[feature_cols].var()
zero_var_cols = variances[variances == 0].index.tolist()
log(f"  零方差特征: {len(zero_var_cols)} 个")

if zero_var_cols:
    df = df.drop(columns=zero_var_cols)
    desc_cols = [c for c in desc_cols if c not in zero_var_cols]
    fp_cols = [c for c in fp_cols if c not in zero_var_cols]
    log(f"  移除后: 描述符 {len(desc_cols)} 个, 指纹 {len(fp_cols)} 位")

# ══════════════════════════════════════════
#  7. 添加分类标签
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 7: 添加分类标签")
log("=" * 60)

# pChEMBL >= 6.0  ⟹  IC50 ≤ 1000 nM  ⟹  Active
ACTIVITY_THRESHOLD = 6.0
df["Activity_Label"] = (df["pChEMBL Value"] >= ACTIVITY_THRESHOLD).astype(int)
n_active = df["Activity_Label"].sum()
n_inactive = len(df) - n_active
log(f"  阈值: pChEMBL ≥ {ACTIVITY_THRESHOLD} → Active")
log(f"  Active:   {n_active} ({n_active / len(df):.1%})")
log(f"  Inactive: {n_inactive} ({n_inactive / len(df):.1%})")

# ══════════════════════════════════════════
#  8. 输出最终统计 & 保存
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 8: 最终统计 & 保存")
log("=" * 60)

final_desc_cols = [c for c in df.columns if c not in info_cols
                   and not c.startswith("Morgan_") and c != "Activity_Label"]
final_fp_cols = [c for c in df.columns if c.startswith("Morgan_")]
final_info_cols = [c for c in info_cols if c in df.columns]

log(f"  最终数据: {len(df)} 行")
log(f"  信息列: {len(final_info_cols)}")
log(f"  描述符: {len(final_desc_cols)}")
log(f"  指纹位: {len(final_fp_cols)}")
log(f"  标签列: pChEMBL Value (回归), Activity_Label (分类)")
log()

# pChEMBL 统计
log("  pChEMBL Value 统计:")
log(f"    Mean  = {df['pChEMBL Value'].mean():.3f}")
log(f"    Std   = {df['pChEMBL Value'].std():.3f}")
log(f"    Min   = {df['pChEMBL Value'].min():.3f}")
log(f"    Max   = {df['pChEMBL Value'].max():.3f}")

# 保存完整数据
log()
log("正在保存文件...")

df.to_excel(OUTPUT_FULL, index=False, engine="openpyxl")
log(f"  ✓ {os.path.basename(OUTPUT_FULL)} ({df.shape[0]} × {df.shape[1]})")

# 保存仅特征+标签 (供模型直接使用)
feature_label_cols = final_desc_cols + final_fp_cols + ["pChEMBL Value", "Activity_Label"]
df_features = df[feature_label_cols]
df_features.to_excel(OUTPUT_FEATURES, index=False, engine="openpyxl")
log(f"  ✓ {os.path.basename(OUTPUT_FEATURES)} ({df_features.shape[0]} × {df_features.shape[1]})")

# 保存报告
with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"  ✓ {os.path.basename(OUTPUT_REPORT)}")

log()
log("数据清洗完成!")
