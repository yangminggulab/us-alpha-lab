# Architecture

这个项目先按“轻量 Qlib + 轻量 Alphalens + 轻量 VectorBT”的思路组织，不直接搬大框架。

## Project Tree By Purpose

```text
US Alpha Lab = Massive 免费数据 alpha 研究实验室
│
├── 0 项目说明与工程配置
│   │
│   ├── README.md
│   │   └── 项目入口说明、快速运行命令、核心产物说明
│   │
│   ├── docs/architecture.md
│   │   └── 架构、数据流、Alpha Cluster 错误码
│   │
│   ├── docs/quant-infra.md
│   │   └── 计算基础设施、benchmark、CUDA/GPU 加速路线
│   │
│   ├── pyproject.toml
│   │   └── Python 依赖、命令行入口、测试配置
│   │
│   ├── .gitignore
│   │   └── 忽略密钥、虚拟环境、数据、报告、本地缓存
│   │
│   ├── .env.example
│   │   └── Massive API Key 示例，不放真实密钥
│   │
│   └── .env
│       └── 本地真实 API Key，被 Git 忽略
│
├── 1 数据与实验配置层
│   │
│   ├── configs/universe.yaml
│   │   └── 小样本股票池：快速试跑和功能验证
│   │
│   ├── configs/universe_free_50.yaml
│   │   └── 免费版 50 股票池：当前主研究配置
│   │
│   ├── configs/universe_free_50_ml.yaml
│   │   └── 免费版 50 股票池 + ML 因子表：用于样本外预测因子评估
│   │
│   └── configs/experiments/free_alpha.yaml
│       └── 免费版实验配置：数据、标签、回测、模型参数
│
├── 2 数据获取与标准化层
│   │
│   ├── src/us_alpha_lab/config.py
│   │   └── 读取 YAML 配置，统一路径和参数
│   │
│   └── src/us_alpha_lab/massive_data.py
│       └── 从 Massive 拉 OHLCV，保存为 parquet
│
├── 3 Alpha 因子生成层
│   │
│   ├── src/us_alpha_lab/alpha_registry.py
│   │   └── alpha 注册表：名称、公式、来源、方向、lookback、说明
│   │
│   ├── src/us_alpha_lab/operators.py
│   │   └── 公式因子基础算子：rank、delay、delta、rolling_mean、ts_rank、rolling_corr
│   │
│   ├── src/us_alpha_lab/formula_factors.py
│   │   └── 公式候选因子库：跳空、均线偏离、低波动、流动性质量等
│   │
│   ├── src/us_alpha_lab/feature_engineering.py
│   │   └── 因子清洗、横截面 rank、横截面 zscore、winsorize
│   │
│   ├── src/us_alpha_lab/factors.py
│   │   └── 组合基础因子和公式因子，并维护 FACTOR_COLUMNS
│   │
│   ├── src/us_alpha_lab/factor_discovery.py
│   │   └── 候选因子快速筛选：覆盖率、横截面区分度、IC、ICIR
│   │
│   └── src/us_alpha_lab/labels.py
│       └── 生成未来收益标签，比如 future_return_5d
│
├── 4 因子评估与回测层
│   │
│   ├── src/us_alpha_lab/analysis.py
│   │   └── IC、ICIR、分位收益等信号层评估
│   │
│   ├── src/us_alpha_lab/backtest.py
│   │   └── 回测核心：BacktestConfig、分位权重、long/top-bottom/benchmark、绩效指标
│   │
│   └── src/us_alpha_lab/leaderboard.py
│       └── 批量合并 IC + 回测指标，输出因子排行榜
│
├── 5 风险与基准模块
│   │
│   ├── src/us_alpha_lab/risk.py
│   │   └── 收益风险指标：波动率、回撤、Calmar、downside、beta、相关性
│   │
│   └── src/us_alpha_lab/benchmark.py
│       └── 等权基准、超额收益、benchmark beta/correlation
│
├── 6 Alpha 检测集群层
│   │
│   ├── src/us_alpha_lab/diagnostics.py
│   │   └── 单个 alpha 体检：覆盖率、弱信号、负组合、高回撤、高换手等
│   │
│   └── src/us_alpha_lab/alpha_cluster.py
│       └── 本地 worker 化批量检测，每个 alpha 挨个测并输出错误码
│
├── 7 机器学习实验层
│   │
│   ├── src/us_alpha_lab/validation.py
│   │   └── walk-forward 时间切分和 embargo 防泄漏
│   │
│   └── src/us_alpha_lab/modeling.py
│       └── 用滞后因子预测未来收益，清洗异常值，输出样本外 ml_prediction_5d
│
├── 8 可视化与命令入口层
│   │
│   ├── src/us_alpha_lab/visualization.py
│   │   └── 输出 IC、分位收益、排行榜、收益曲线、回撤图
│   │
│   ├── src/us_alpha_lab/cli.py
│   │   └── alpha-lab 命令入口：fetch/factors/discover-factors/ml-alpha/report/backtest/leaderboard
│   │
│   └── src/us_alpha_lab/__init__.py
│       └── 包版本和基础包声明
│
├── 9 实验管理层
│   │
│   ├── src/us_alpha_lab/experiment.py
│   │   └── 创建 runs/ 实验目录，保存配置快照、排行榜、检测报告、图表和指标
│   │
│   └── runs/<timestamp>_<name>/
│       └── 每次实验的完整产物目录，被 Git 忽略
│
├── 10 测试层
│   │
│   ├── tests/test_operators.py
│   ├── tests/test_factors.py
│   ├── tests/test_analysis.py
│   ├── tests/test_backtest.py
│   ├── tests/test_leaderboard.py
│   ├── tests/test_diagnostics.py
│   └── tests/test_visualization.py
│       └── 保障每层核心逻辑能跑通，防止改 alpha 时破坏主流程
│
└── 11 本地产物层（不进 Git）
    │
    ├── data/raw/*.parquet
    │   └── Massive 下载的原始日线数据
    │
    ├── data/processed/*.parquet
    │   └── 因子表
    │
    ├── data/models/*.joblib
    │   └── 机器学习模型
    │
    ├── reports/**/*.csv / reports/**/*.png / reports/**/*.parquet
    │   └── IC 报告、排行榜、图表、回测明细、Alpha Cluster 诊断报告
    │
    ├── runs/<timestamp>_<name>/
    │   └── 实验快照：config、registry、leaderboard、cluster、backtest、charts、metrics
    │
    └── .venv/、build/、__pycache__/、*.egg-info/
        └── 本地环境、构建缓存、Python 缓存
```

## Data Flow

```text
1. alpha-lab fetch
   Massive API -> data/raw/daily_bars.parquet

2. alpha-lab factors
   raw OHLCV -> operators -> formula_factors -> feature_engineering -> data/processed/factors.parquet

3. alpha-lab discover-factors
   factors + forward returns -> reports/factor_discovery/factor_candidates.csv

4. alpha-lab report
   factors + forward returns -> IC table

5. alpha-lab charts
   factors + forward returns -> reports/*.png

6. alpha-lab backtest
   factor ranks -> top-minus-bottom quantile portfolio -> reports/backtest.parquet

7. alpha-lab train
   lagged factors -> return/rank/zscore labels -> model metrics + data/models/latest_model.joblib

8. alpha-lab ml-alpha
   lagged factors -> walk-forward ML prediction -> data/processed/*_ml.parquet

9. alpha-lab leaderboard
   IC report + factor backtests -> reports/leaderboard/factor_leaderboard.csv + charts

10. alpha-lab tune-lightgbm
   factors -> conservative LightGBM/LambdaRank/rank_xendcg search -> reports/lightgbm_tuning.csv

11. alpha-lab alpha-cluster
   candidate factors -> diagnostics -> reports/alpha_cluster/alpha_cluster_report.csv

12. alpha-lab run-experiment
   existing factors -> runs/<timestamp>_<name>/ complete experiment snapshot
```

## Design Rules

- 数据下载和研究分析分离：拉数据慢且受免费额度限制，后续分析尽量复用本地 parquet。
- 因子先透明后复杂：先用价格、成交量、VWAP 的可解释因子，再引入价格路径、残差路径和更多公式因子。
- 先信号评估，再组合评估：IC/Rank IC 判断预测性，分位回测判断能不能变成组合收益。
- 机器学习必须按时间切分：默认不随机打散，后续再加 walk-forward、purging 和 embargo。
- 报告和模型不进 Git：`data/`、`reports/`、`.env` 都是本地产物。

## Factor Discovery Module

因子发现模块现在分成四层：

- `operators.py`: 提供可复用基础算子，避免每个公式都手写 rolling 逻辑。
- `factors.py`: 生成基础因子、价格路径因子和市场残差路径因子。
- `formula_factors.py`: 集中放候选公式因子，包括隔夜跳空反转、日内强度、均线偏离、21 日高点距离、流动性质量等。
- `feature_engineering.py`: 统一清洗 `inf`，并生成横截面 rank / zscore 特征，给回测和机器学习共用。
- `factor_discovery.py`: 在完整回测前先输出候选报告，看覆盖率、横截面区分度、IC、ICIR 和 discovery score。

推荐节奏是：先 `alpha-lab factors` 生成候选，再 `alpha-lab discover-factors` 快速筛一遍，最后把可疑或靠前的因子送进 leaderboard 和 Alpha Cluster。

## Machine Learning Module

机器学习模块现在不是只训练一个模型文件，而是把模型预测本身作为一个可评估 alpha。

- `validation.py`: 生成 walk-forward 时间切分，并在训练窗口和测试窗口之间留出 embargo。
- `labels.py`: 同时支持未来收益原值、每日横截面 rank、z-score、quantile 和 top/bottom 分位标签。
- `modeling.py`: 用滞后后的因子预测未来 5 日收益或横截面排序标签；支持 `random_forest`、`lightgbm`、`lightgbm_ranker` 和 `rank_xendcg`，训练前把 `inf` 这类异常值转成缺失值，再交给中位数填补器。
- `alpha-lab ml-alpha`: 输出 `ml_prediction_5d` 到新的因子表，例如 `data/processed/factors_free_50_ml.parquet`。
- `alpha-lab tune-lightgbm`: 跑保守复杂度的 LightGBM / LambdaRank / rank_xendcg 搜索，覆盖 zscore、quantile、top_bottom 标签，按样本外横截面 IC 和 ICIR 排序。
- `configs/universe_free_50_ml.yaml`: 指向带 ML 因子的表，让排行榜、回测、Alpha Cluster 直接评估机器学习 alpha。

ML 因子只在样本外测试窗口有值，所以 Alpha Cluster 对这类因子使用“有效预测期”做覆盖率和横截面区分度检查。

## Backtest Module

回测现在作为独立重点模块维护，核心对象是：

- `BacktestConfig`: 回测配置，包括因子名、持有期、分位数、交易成本。
- `build_quantile_weights`: 把每日因子值转成 top/bottom 分位持仓权重。
- `run_quantile_backtest`: 生成 long-only、top-minus-bottom、等权 benchmark、long excess 四类收益。
- `BacktestResult`: 保存 `daily_returns`、`weights` 和 `metrics`。

当前回测仍是研究筛选级别，不是实盘撮合器。它的目标是快速判断 alpha 是否值得继续研究，而不是模拟所有交易细节。

## Risk And Benchmark Module

风险与基准是独立模块，不归属于回测模块。回测负责生成组合收益，风险与基准负责解释收益质量。

- `risk.py`: 通用风险指标，包括最大回撤、年化波动、Calmar、downside volatility、benchmark beta、benchmark correlation。
- `benchmark.py`: 当前先支持股票池等权基准，输出超额收益、benchmark beta 和相关性。

它会被两个模块调用：

- 回测模块：补充 benchmark、excess、beta、correlation 指标。
- Alpha Cluster：增加 `POOR_EXCESS_RETURN`、`HIGH_BENCHMARK_BETA`、`HIGH_BENCHMARK_CORRELATION` 等错误码。

## Experiment Management

`alpha-lab run-experiment` 会创建一个独立 run 目录。它默认不重新下载数据，而是复用配置里的 `factors_path`。

每个 run 包含：

- `config.yaml`: 本次实验使用的配置快照。
- `alpha_registry.csv`: 本次可用 alpha 的注册表快照。
- `ic_report.csv`: 因子 IC 表。
- `factor_leaderboard.csv`: IC + 回测排行榜。
- `alpha_cluster/alpha_cluster_report.csv`: alpha 错误诊断。
- `best_backtest/returns.parquet`: 最佳因子回测收益序列。
- `best_backtest/weights.parquet`: 最佳因子每日持仓权重。
- `charts/`: 图表。
- `metrics.json`: 机器可读的核心参数和结果摘要。

## Alpha Cluster

`alpha-lab alpha-cluster` 是本地版 alpha 检测集群。它不是先追求机器规模，而是先固定每个 alpha 必须经过的检查。

主要错误码：

- `MISSING_FACTOR`: 因子列不存在。
- `LOW_COVERAGE`: 有效值覆盖率过低。
- `LOW_ACTIVE_DAYS`: 可用于分位回测的交易日太少。
- `CONSTANT_CROSS_SECTION`: 横截面缺少区分度，无法稳定分组。
- `WEAK_SIGNAL`: IC 和 ICIR 都太弱。
- `NEGATIVE_PORTFOLIO`: 分位多空组合收益或 IR 为负。
- `HIGH_DRAWDOWN`: 回撤过高。
- `HIGH_TURNOVER`: 换手过高，可能被交易成本吃掉。
- `IC_BACKTEST_MISMATCH`: IC 方向和组合结果不一致。

后续如果迁到真正的集群，这层可以替换成任务队列；单个 alpha 的诊断函数不需要大改。
