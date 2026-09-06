# L2 / L3 Research

这个目录是订单簿与逐笔数据研究的入口。大体量原始数据仍放在项目标准数据目录，避免移动 parquet 后破坏现有脚本路径。

## Scope

- L2: 多档盘口深度、depth imbalance、microprice、OFI、短周期价格冲击。
- L3: 逐笔委托、逐笔成交、撤单/成交行为、隐藏参与者与状态空间推断。
- 面试表达材料保留在 `../../面试资料/`，这里主要放研究执行材料。
- L2/L3 研究方向路线图放在 `../../学习研究路线/l2_l3_research_roadmap.md`。

## Local Data

- 原始 A 股 L2/HF 数据: `../../data/raw/a_share_l2_hf/`
- 下载脚本: `../../scripts/download_a_share_l2_hf.py`
- 当前数据清单: [data_inventory.md](data_inventory.md)

## Working Folders

- `docs/`: 研究 memo、特征定义、实验设计。
- `notebooks/`: 探索性分析 notebook。
- `outputs/`: 小体量中间结果、截图、临时表。大体量产物继续放 `../../reports/` 或 `../../runs/`。

## Current Position

现在的数据足够做 L2/L3 原型和信号检测，不足以做正式稳健结论。优先目标是先用已下载的完整交易日跑通:

1. 数据读取与交易日切片。
2. L2/L3 基础特征: depth imbalance、microprice、OFI、成交主动性 proxy、撤单强度。
3. 短 horizon label 与防前视对齐。
4. 单票/多票样本外检验和成本敏感性。
