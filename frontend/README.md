# MolSight 前端

这是一个无构建依赖的静态前端，面向当前项目的分子活性预测与虚拟筛选流程。

## 使用

最简单的方式是双击仓库根目录的 `启动前端.bat`，它会自动打开 MolSight 页面，不需要启动 Python 或 Node。

也可以直接双击本目录中的 `index.html`，体验内置演示数据。为了让页面读取仓库中的完整 CSV 和报告，建议在仓库根目录启动静态服务器：

```bash
python -m http.server 8000
```

然后打开 <http://localhost:8000/frontend/>。

页面包含：

- 项目总览：候选数量、活性比例、分布图和高潜力分子；
- 单分子预测：输入 SMILES，展示 pIC50、活性概率和分子属性；
- 批量虚拟筛选：导入 CSV、按概率过滤、分页、排序和导出；
- 模型评估：展示现有 Stacking / Voting / 基模型指标；
- 实验报告：链接到仓库已有的报告和图表。

## 接入真实模型 API

页面会自动尝试调用 `POST /api/predict`。请求体：

```json
{
  "smiles": "CCOc1ccc2[nH]c(C(=O)NCC)cc2c1",
  "model": "Stacking",
  "threshold": 6.0
}
```

接口返回至少包含 `prediction` 和 `probability`，也可以补充 `confidence`、`mw`、`aromatic`、`hetero`、`rings`、`logp`、`uncertainty`。接口不可用时，页面会回退到本地演示估算，并在结果卡片标注“演示估算”。
