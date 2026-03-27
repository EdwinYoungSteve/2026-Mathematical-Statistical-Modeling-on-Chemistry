"""
将 SMILES 转化为分子描述符（Molecular Descriptors）和分子指纹（Morgan Fingerprint）
用于后续基团-推测模型的建立

输入: 整理后原始数据.xlsx（含 Smiles 列）
输出:
  - 分子描述符.xlsx    : 包含 208 个 RDKit 2D 分子描述符
  - 分子指纹.xlsx      : 包含 Morgan 指纹 (半径2, 2048位)
  - 描述符与指纹合并.xlsx : 合并上述两者 + 原始活性数据
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

warnings.filterwarnings("ignore")

# ── 路径配置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "整理后原始数据.xlsx")
OUTPUT_DESC = os.path.join(SCRIPT_DIR, "分子描述符.xlsx")
OUTPUT_FP = os.path.join(SCRIPT_DIR, "分子指纹.xlsx")
OUTPUT_MERGED = os.path.join(SCRIPT_DIR, "描述符与指纹合并.xlsx")

# ── 读取数据 ──
print("正在读取数据...")
df = pd.read_excel(INPUT_FILE, engine="openpyxl")
print(f"  共 {len(df)} 条记录, {df['Smiles'].notna().sum()} 条含有 SMILES")

# ── SMILES → RDKit Mol 对象 ──
print("正在解析 SMILES...")
df["Mol"] = df["Smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
valid_mask = df["Mol"].notna()
n_invalid = (~valid_mask).sum()
if n_invalid > 0:
    print(f"  警告: {n_invalid} 条 SMILES 解析失败, 将被跳过")
    invalid_ids = df.loc[~valid_mask, "Molecule ChEMBL ID"].tolist()
    print(f"  失败的 ID: {invalid_ids[:10]}{'...' if len(invalid_ids) > 10 else ''}")

df_valid = df[valid_mask].copy().reset_index(drop=True)
print(f"  有效分子数: {len(df_valid)}")

# ── 计算 208 个 2D 分子描述符 ──
print("正在计算 2D 分子描述符 (208个)...")
desc_names = [name for name, _ in Descriptors.descList]
calculator = MolecularDescriptorCalculator(desc_names)

desc_data = []
for i, mol in enumerate(df_valid["Mol"]):
    try:
        desc_values = calculator.CalcDescriptors(mol)
        desc_data.append(desc_values)
    except Exception as e:
        desc_data.append([np.nan] * len(desc_names))
        print(f"  第 {i} 行描述符计算出错: {e}")
    if (i + 1) % 500 == 0:
        print(f"  已处理 {i + 1}/{len(df_valid)}...")

df_desc = pd.DataFrame(desc_data, columns=desc_names)
print(f"  描述符矩阵: {df_desc.shape}")

# ── 计算 Morgan 分子指纹 (ECFP4, 2048位) ──
print("正在计算 Morgan 分子指纹 (radius=2, nBits=2048)...")
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

fp_data = []
for i, mol in enumerate(df_valid["Mol"]):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        fp_data.append(list(fp.ToBitString()))
    except Exception as e:
        fp_data.append([np.nan] * MORGAN_NBITS)
        print(f"  第 {i} 行指纹计算出错: {e}")
    if (i + 1) % 500 == 0:
        print(f"  已处理 {i + 1}/{len(df_valid)}...")

fp_cols = [f"Morgan_{j}" for j in range(MORGAN_NBITS)]
df_fp = pd.DataFrame(fp_data, columns=fp_cols).astype("int8")
print(f"  指纹矩阵: {df_fp.shape}")

# ── 组装输出 ──
# 保留的原始信息列
info_cols = ["Molecule ChEMBL ID", "Smiles", "Molecular Weight", "AlogP",
             "Standard Type", "Standard Value", "Standard Units", "pChEMBL Value"]
info_cols = [c for c in info_cols if c in df_valid.columns]
df_info = df_valid[info_cols].reset_index(drop=True)

# 1) 纯描述符表
df_desc_out = pd.concat([df_info, df_desc], axis=1)

# 2) 纯指纹表
df_fp_out = pd.concat([df_info, df_fp], axis=1)

# 3) 合并表
df_merged = pd.concat([df_info, df_desc, df_fp], axis=1)

# ── 导出 ──
print("正在导出 Excel 文件...")

df_desc_out.to_excel(OUTPUT_DESC, index=False, engine="openpyxl")
print(f"  ✓ {OUTPUT_DESC}  ({df_desc_out.shape[0]} 行 × {df_desc_out.shape[1]} 列)")

df_fp_out.to_excel(OUTPUT_FP, index=False, engine="openpyxl")
print(f"  ✓ {OUTPUT_FP}  ({df_fp_out.shape[0]} 行 × {df_fp_out.shape[1]} 列)")

df_merged.to_excel(OUTPUT_MERGED, index=False, engine="openpyxl")
print(f"  ✓ {OUTPUT_MERGED}  ({df_merged.shape[0]} 行 × {df_merged.shape[1]} 列)")

# ── 统计摘要 ──
print("\n========== 描述符统计摘要 ==========")
print(f"有效分子总数: {len(df_valid)}")
print(f"2D 描述符数量: {len(desc_names)}")
print(f"Morgan 指纹维度: {MORGAN_NBITS}")

# 检查描述符中的异常值 (NaN / Inf)
nan_counts = df_desc.isna().sum()
inf_counts = (df_desc == np.inf).sum() + (df_desc == -np.inf).sum()
problem_cols = nan_counts[nan_counts > 0].index.tolist() + inf_counts[inf_counts > 0].index.tolist()
if problem_cols:
    print(f"含 NaN/Inf 的描述符列: {list(set(problem_cols))}")
else:
    print("所有描述符均无 NaN/Inf 异常值")

print("\n完成!")
