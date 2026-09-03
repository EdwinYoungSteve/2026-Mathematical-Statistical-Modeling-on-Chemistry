# MolSight 前端

这是一个无构建依赖的静态前端，面向当前项目的分子活性预测与虚拟筛选流程。

## 使用

最简单的方式是双击 `frontend/启动前端.bat`，它会自动打开 MolSight 页面，不需要启动 Python 或 Node。

也可以直接双击本目录中的 `index.html`，体验内置演示数据。为了让页面读取仓库中的完整 CSV 和报告，建议在仓库根目录启动静态服务器：

```bash
python -m http.server 8000
```

然后打开 <http://localhost:8000/frontend/>。

页面包含：

- 项目总览：候选数量、活性比例、分布图和高潜力分子；
- 单分子预测：直接输入 SMILES，或使用 JSME 结构画板绘制后回填，并展示 pIC50、活性概率和分子属性；
- 批量虚拟筛选：导入 CSV、自动补算缺失预测、按概率过滤、分页、勾选和导出；
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

接口返回至少包含 `prediction` 和 `probability`，也可以补充 `confidence`、`mw`、`aromatic`、`hetero`、`rings`、`logp`、`uncertainty`。接口不可用时，页面会回退到本地演示估算，并在结果卡片和侧栏状态中明确标注。

批量预测接口使用 `POST /api/predict/batch`，请求体：

```json
{
  "molecules": [
    { "id": "CHEMBL123", "smiles": "CCOc1ccccc1" }
  ],
  "model": "Stacking",
  "threshold": 6.0
}
```

返回值可以是结果数组，也可以是 `{ "results": [...] }`。结果顺序应与请求中的分子顺序一致。

## 分子结构画板

单分子预测页会按需从 jsDelivr 加载固定版本的 JSME 结构编辑器。首次打开画板需要网络连接；如果组件加载失败，原有 SMILES 文本输入和预测功能仍可正常使用。
