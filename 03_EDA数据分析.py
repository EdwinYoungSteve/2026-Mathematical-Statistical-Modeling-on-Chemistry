import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_excel("清洗后数据.xlsx", engine="openpyxl")
os.makedirs("figures", exist_ok=True)

target = "pChEMBL Value"
label = "Activity_Label"
info_cols = [c for c in df.columns if "Molecule" in c or "Smiles" in c or "Standard" in c]
fp_cols = [c for c in df.columns if c.startswith("Morgan_")]
desc_cols = [c for c in df.columns if c not in info_cols + fp_cols + [target, label]]

missing = df.isnull().sum()[df.isnull().sum() > 0]
if len(missing) > 0:
    plt.figure(figsize=(10,5))
    missing.head(20).plot(kind="bar")
    plt.title("缺失值统计")
    plt.savefig("figures/缺失值.png")
    plt.close()

var_feat = df[desc_cols+fp_cols].var()
plt.figure(figsize=(10,5))
plt.hist(var_feat, bins=50, log=True)
plt.title("特征方差分布")
plt.savefig("figures/方差分布.png")
plt.close()

sample = np.random.choice(desc_cols, min(20, len(desc_cols)))
df[sample].hist(figsize=(16,12), bins=30)
plt.savefig("figures/描述符分布.png")
plt.close()

top_corr = df[desc_cols].corrwith(df[target]).abs().nlargest(6).index
plt.figure(figsize=(18,10))
for i, c in enumerate(top_corr):
    plt.subplot(2,3,i+1)
    sns.boxplot(x=label, y=c, data=df)
plt.savefig("figures/活性差异.png")
plt.close()

top50 = df[desc_cols].corrwith(df[target]).abs().nlargest(50).index
Xv = df[top50].dropna()
vif = pd.DataFrame({"特征":top50, "VIF":[variance_inflation_factor(Xv.values,i) for i in range(len(top50))]})
vif = vif.sort_values("VIF", ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x="VIF", y="特征", data=vif.head(15))
plt.axvline(10, c="k", ls="--")
plt.savefig("figures/VIF共线性.png")
plt.close()

plt.figure(figsize=(7,5))
sns.countplot(x=label, data=df)
plt.title("样本分布")
plt.savefig("figures/样本不平衡.png")
plt.close()

plt.figure(figsize=(10,5))
sns.histplot(df[target], bins=40, kde=True)
plt.axvline(6, c="r", ls="--")
plt.savefig("figures/目标变量分布.png")
plt.close()

top30 = df[desc_cols].corrwith(df[target]).abs().nlargest(30).index
corr = df[top30].corr()
plt.figure(figsize=(14,12))
mask = np.triu(corr)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0)
plt.savefig("figures/相关性热力图.png")
plt.close()

report = f"""数据规模：{df.shape}
描述符数量：{len(desc_cols)}
指纹数量：{len(fp_cols)}
缺失值特征数：{len(missing)}
零方差特征数：{len(var_feat[var_feat==0])}
高共线性特征数(VIF>10)：{len(vif[vif.VIF>10])}
活性样本：{df[label].sum()}
非活性样本：{(df[label]==0).sum()}
"""

with open("EDA报告.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("✅ EDA 全部完成")

