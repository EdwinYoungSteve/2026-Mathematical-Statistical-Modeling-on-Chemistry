"""
环节三：特征工程
================
输入:  环节二/清洗后数据.xlsx
输出:
  - 特征工程后数据.npz       : 含训练/验证/测试集的 numpy 数组
  - 特征工程报告.txt          : 各步操作的统计
  - feature_names.txt        : 特征名称列表 (与数组列对应)

步骤:
  1) 去除高相关冗余描述符 (|r| > 0.95)
  2) 描述符 StandardScaler 标准化, 指纹保持原样
  3) 8:1:1 分层划分 train / val / test
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ── 路径配置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_FILE = os.path.join(PROJECT_DIR, "环节二：数据清洗与EDA", "清洗后数据.xlsx")
OUTPUT_NPZ = os.path.join(SCRIPT_DIR, "特征工程后数据.npz")
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "特征工程报告.txt")
OUTPUT_FEAT_NAMES = os.path.join(SCRIPT_DIR, "feature_names.txt")

report = []


def log(msg=""):
    print(msg)
    report.append(msg)


# ══════════════════════════════════════════
#  1. 读取 & 分列
# ══════════════════════════════════════════
log("=" * 60)
log("步骤 1: 读取数据")
log("=" * 60)

df = pd.read_excel(INPUT_FILE, engine="openpyxl")
log(f"  原始: {df.shape[0]} 行 × {df.shape[1]} 列")

info_cols = ["Molecule ChEMBL ID", "Smiles", "Molecular Weight", "AlogP",
             "Standard Type", "Standard Value", "Standard Units",
             "pChEMBL Value", "Activity_Label"]
fp_cols = sorted([c for c in df.columns if c.startswith("Morgan_")])
desc_cols = sorted([c for c in df.columns if c not in info_cols and c not in fp_cols])

log(f"  描述符: {len(desc_cols)}, 指纹: {len(fp_cols)}")

# ══════════════════════════════════════════
#  2. 去除高相关冗余描述符
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 2: 去除高相关冗余描述符 (|r| > 0.95)")
log("=" * 60)

corr_matrix = df[desc_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

log(f"  发现 {len(to_drop)} 个冗余描述符")
if to_drop:
    log(f"  移除: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")

desc_cols = [c for c in desc_cols if c not in to_drop]
log(f"  保留描述符: {len(desc_cols)}")

# ══════════════════════════════════════════
#  3. 标准化描述符, 指纹不动
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 3: 标准化")
log("=" * 60)

X_desc = df[desc_cols].values.astype(np.float64)
X_fp = df[fp_cols].values.astype(np.float32)
y_reg = df["pChEMBL Value"].values.astype(np.float64)
y_cls = df["Activity_Label"].values.astype(np.int32)

scaler = StandardScaler()
X_desc_scaled = scaler.fit_transform(X_desc).astype(np.float32)

# 合并特征
X = np.hstack([X_desc_scaled, X_fp])
feature_names = desc_cols + fp_cols
log(f"  合并特征矩阵: {X.shape}")
log(f"  描述符 (标准化): {X_desc_scaled.shape[1]}")
log(f"  指纹 (原始 0/1):  {X_fp.shape[1]}")

# ══════════════════════════════════════════
#  4. 数据集划分 8:1:1
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 4: 数据集划分 (8:1:1, 分层)")
log("=" * 60)

# 先划出 20% 作为 val+test
X_train, X_tmp, y_reg_train, y_reg_tmp, y_cls_train, y_cls_tmp = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
# 再对半分成 val 和 test
X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
    X_tmp, y_reg_tmp, y_cls_tmp, test_size=0.5, random_state=42, stratify=y_cls_tmp
)

log(f"  Train: {X_train.shape[0]} ({X_train.shape[0]/len(X):.1%})")
log(f"  Val:   {X_val.shape[0]} ({X_val.shape[0]/len(X):.1%})")
log(f"  Test:  {X_test.shape[0]} ({X_test.shape[0]/len(X):.1%})")
log()
log(f"  Train Active/Inactive: {y_cls_train.sum()}/{len(y_cls_train)-y_cls_train.sum()}")
log(f"  Val   Active/Inactive: {y_cls_val.sum()}/{len(y_cls_val)-y_cls_val.sum()}")
log(f"  Test  Active/Inactive: {y_cls_test.sum()}/{len(y_cls_test)-y_cls_test.sum()}")

# ══════════════════════════════════════════
#  5. 保存
# ══════════════════════════════════════════
log()
log("=" * 60)
log("步骤 5: 保存")
log("=" * 60)

np.savez_compressed(
    OUTPUT_NPZ,
    X_train=X_train, X_val=X_val, X_test=X_test,
    y_reg_train=y_reg_train, y_reg_val=y_reg_val, y_reg_test=y_reg_test,
    y_cls_train=y_cls_train, y_cls_val=y_cls_val, y_cls_test=y_cls_test,
    scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
    n_desc=len(desc_cols), n_fp=len(fp_cols),
)
log(f"  ✓ {os.path.basename(OUTPUT_NPZ)}")

with open(OUTPUT_FEAT_NAMES, "w", encoding="utf-8") as f:
    f.write("\n".join(feature_names))
log(f"  ✓ {os.path.basename(OUTPUT_FEAT_NAMES)} ({len(feature_names)} features)")

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
log(f"  ✓ {os.path.basename(OUTPUT_REPORT)}")

log()
log("特征工程完成!")
