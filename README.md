# US Alpha Lab

一个美股量化因子研究项目。**核心不是工具，而是一套研究思路：怎样系统地找到别人拿不到的 alpha。**

> 仅供学习研究，不构成投资建议；历史表现不代表未来收益。真实交易前还需处理复权、幸存者偏差、交易成本、滑点、组合约束与样本外验证。

## 在线报告

- [报告入口页](https://yangminggulab.github.io/us-alpha-lab/)
- [研究报告](https://yangminggulab.github.io/us-alpha-lab/research_report.html)
- [方法说明](https://yangminggulab.github.io/us-alpha-lab/methodology.html)

（本地预览：[研究报告](https://htmlpreview.github.io/?https://github.com/yangminggulab/us-alpha-lab/blob/main/research_report.html) / [方法说明](https://htmlpreview.github.io/?https://github.com/yangminggulab/us-alpha-lab/blob/main/methodology.html)）

---

## 学习路线

- [ML 学习路线：随机森林 / LightGBM / XGBoost](docs/ml-learning.md) — 结合本项目练"熟练掌握"，含公式、代码位置、打勾清单与动手实验
- [补缺书单：从数学底座到前沿因子研究](docs/reading-gap-booklist.md) — 基于桌面网站源码仓库和本地 ML 论文仓库，按知识缺口筛权威书
- [文件地图与整理建议](docs/file-map.md) — 当前代码、数据、报告、runs 的主线与可清理项
- [学习 TODO](todo.md) — 逐项练习清单：两种树模型 + PyTorch（手写 MLP → 训练循环 → transformer）
- [量化笔记 dx-quant](dx-quant/) — LaTeX 学习笔记（基础知识 + 挖因子），独立仓库以 submodule 挂载

---

## 一、这项研究在做什么

### 我们到底在找什么

一句话答案：**找一个横截面信号 $s_{i,t}$，使正交化后的增量 IC 稳健为正。**

$$
\varepsilon_{i,t} = s_{i,t} - \sum_k \beta_k\, f_{k,i,t},\qquad
\mathrm{IC_{orth}} = \mathrm{corr}\big(\varepsilon,\ \text{future return}\big)
$$

$f_k$ 是公共因子池，残差 $\varepsilon$ 才是别人拿不到、属于你自己的部分。**研究的产出 = 增量 IC，不是绝对 IC。** 动量、低波 IC 永远不低，但所有人都在用，没有研究价值；只有 $\mathrm{IC_{orth}} \neq 0$ 的信号才值得研究。

### 研究逻辑链

```
数据（日线 OHLCV）
  → 公共因子（动量/波动/量价/位置）
  → 随机森林把所有因子"拧"成一个预测（样本外）
  → 四关 + 增量关裁决：哪个因子值得进一步研究
```

每一步都对应一个明确的问题：

1. **数据**：从 Massive 免费层拉日线。
2. **公共因子**：作为探针验证流水线、作为基线比较增量、作为底仓。它们本身不是 alpha。
3. **机器学习因子**：随机森林做**回归**（预测连续收益，MSE 损失），输出当作一个因子按横截面**排名**消费——回归训练、排名消费，这是标准的因子研究框架。
4. **裁决表**：信号关、单调关、净收益关、稳定关、增量关五关全过，才叫"可盈利候选"。

### 几条目前的关键认识

- **没有"每年都赚"的因子。** 年化夏普 1.0 的因子每 6 年也要亏一年。目标是"期望为正 + 可解释 + 知道何时失效"，不是"永远赚钱"。
- **公共因子是标尺，不是猎物。** 它们的价值在于定义"什么才算新 alpha"，而不是本身赚钱。
- **防前视是生死线。** walk-forward 滚动验证 + embargo + 特征滞后一天，三者缺一不可。
- **随机森林学的是公共因子的非线性组合。** 真正的增量往往不在这里，而在新数据、交互项、新预测目标。

### 当前研究状态

在免费 50 支美股、约 2 年日线上：**五关裁决没有因子全过。** 最有价值的一条证据是随机森林因子 `ml_prediction_5d` 的原始 IC 为 +0.019，正交化后变成 −0.011——说明它的信号基本是公共因子的重组，剥掉公共部分后没有增量。这验证了增量关存在的意义：**诚实地说出"手里还没有别人拿不到的因子"，比自欺欺人地看绝对 IC 重要得多。**

在 S&P 500 日线、约 2 年样本上，数据已经扩到 `data/raw/daily_bars_sp500.parquet`，并进入 **LightGBM / LambdaRank** 阶段。第一版 LightGBM 回归可跑通，但 `ml_prediction_5d` 的 IC 仍偏弱；后续优化重点已经从"预测收益绝对值"切到"预测横截面排序"。

---

## 二、研究路线（怎么做研究）

1. **造新信号。** 四个来源，按壁垒递增：公共数据上的新公式 → 价格路径/残差路径 → 非线性交互项 → 新预测目标（横截面排名、波动、跳跃、成交量）→ 新数据（财报、新闻、期权、资金流才是真正的护城河）。
2. **正交化验证。** 每个候选对公共因子池回归取残差，看 $\mathrm{IC_{orth}}$ 是否显著。残差 IC ≈ 0 说明只是公共因子的组合。
3. **五关裁决。** 信号 / 单调 / 净收益 / 稳定 / 增量，全过才进入下一轮。
4. **经济可解释 + 失效预案。** 能说清赚的是哪类钱（风险补偿 or 行为偏差）、知道什么情况下会亏，才配叫"你的 alpha"。

---

## 三、学习路线（怎么学）

从"能跑模型"到"知道为什么赚钱、怎么验证、怎么变组合"，树的枝干是学习阶段，枝干下挂对应书单（完整书单见 [docs/reading-gap-booklist.md](docs/reading-gap-booklist.md)）：

```
从"能跑模型"到"能验证、能解释、能变组合"
│
├─ 阶段一：数据与公共因子 —— 建立标尺
│   ├─ 概念：横截面 Spearman IC、ICIR、分位数收益、因子自相关
│   ├─ 代码：analysis.py、alpha_registry.py、feature_engineering.py
│   └─ 认知：公共因子是探针。动量、低波 IC 若为负，先怀疑数据或代码
│
├─ 阶段二：回测与多空组合 —— 预测力能否变成收益
│   ├─ 公式：年化毛利 ≈ IC × √breadth × σ_截面，盈利条件 = 毛利 > 换手成本
│   ├─ 代码：backtest.py、risk.py、benchmark.py
│   ├─ 认知：信号强不等于赚钱，扣费后的夏普才决定一切
│   └─ 书：Grinold & Kahn《Active Portfolio Management》—— 从预测到仓位
│
├─ 阶段三：机器学习因子 —— 回归不是分类
│   ├─ 概念：MSE 下最优预测 = 条件期望 E[y|x]；选股消费横截面排序，比较 rank/quantile/LambdaRank
│   ├─ 代码：modeling.py、validation.py、labels.py
│   ├─ 认知：回归是基线，排名学习是下一层；全部用 walk-forward + embargo + 特征滞后一天验证
│   └─ 书：★ López de Prado《Advances in Financial ML》—— 防泄露/多重检验/假回测
│
├─ 阶段四：增量 IC 与正交化 —— 找自己的 alpha
│   ├─ 概念：截面回归取残差、IC_orth、多重检验偏差
│   ├─ 代码：verdict.py 的增量关
│   ├─ 认知：ML 因子大概率是公共因子的重组，正交化会诚实说出来
│   └─ 书：Bali《Empirical Asset Pricing》—— 横截面因子的正统研究法
│
├─ 阶段五：可解释性与失效 —— 研究的护城河
│   ├─ 概念：利润来源两类（风险补偿 vs 行为偏差）；拥挤、行为演化、套利三个失效机制
│   ├─ 认知：统计可解释 ≠ 经济可解释
│   └─ 书：Cochrane《Asset Pricing》→ Ang《Asset Management》→ Ilmanen《Expected Returns》
│       —— 用资产定价语言回答"这个因子为什么该赚"
│
├─ 阶段六：模型进阶（已进入）
│   ├─ 随机森林 → LightGBM / LightGBM Ranker：对比而不是替换，同一套 walk-forward 比 OOS IC/ICIR/分层/成本后收益
│   ├─ 已新增价格路径与残差路径因子（alpha_residual_momentum_21d 等）
│   └─ 书：Tsay《Analysis of Financial Time Series》→ Campbell《Econometrics of Financial Markets》
│       —— 金融数据不是普通 IID 表格：非平稳、异方差、厚尾
│
└─ 延伸（非主线，未来再用）
    ├─ 研究员线：ESL → Statistical Learning with Sparsity → High-Dim Statistics
    │   → Causal Inference → Market Microstructure（更会找因子 + 验证纪律）
    ├─ 组合线：Qian《Quantitative Equity Portfolio Management》→ Boyd《Convex Optimization》
    ├─ 未来线：Probabilistic ML → Online Convex Optimization → RL → Stochastic Calculus
    └─ 明确不做：纯数学堆砌、深度学习优先、交易玄学书、重复搜广推入门
```

---

## 四、复现与运行（最小命令）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[dev]"
cp .env.example .env          # 填入 MASSIVE_API_KEY

alpha-lab fetch --tickers AAPL,MSFT,NVDA --start 2024-08-15 --end 2026-08-14
alpha-lab factors
alpha-lab ml-alpha --config configs/universe_free_50.yaml
alpha-lab html-report
```

S&P 500 + LightGBM 排名学习：

```bash
alpha-lab factors --config configs/universe_sp500.yaml
alpha-lab ml-alpha --config configs/universe_sp500.yaml \
  --model lightgbm_ranker \
  --label-transform quantile \
  --output-path data/processed/factors_sp500_lgbm_ranker.parquet
alpha-lab tune-lightgbm --config configs/universe_sp500.yaml \
  --output-path reports/lightgbm_tuning_sp500.csv
```

产物说明、免费层批量拉数脚本（`scripts/harvest_massive_free.py` 与 launchd 后台脚本）详见各脚本头部注释与 `docs/`。
