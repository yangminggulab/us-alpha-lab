# US Alpha Lab

一个美股量化研究毛坯房：用 Massive 拉 OHLCV 数据，做基础 alpha 因子，生成未来收益标签，再用机器学习做第一版预测实验。

> 仅用于学习和研究，不构成投资建议。真实交易前还需要严谨处理复权、幸存者偏差、交易成本、滑点、组合约束和样本外验证。

## 在线报告

- [报告入口页](https://yangminggulab.github.io/us-alpha-lab/)
- [研究报告](https://yangminggulab.github.io/us-alpha-lab/research_report.html)
- [方法说明](https://yangminggulab.github.io/us-alpha-lab/methodology.html)

如果链接显示 404，需要在 GitHub 仓库 `Settings -> Pages` 里把发布源设为 `gh-pages` 分支的 `/(root)`。
Pages 启用前，也可以用临时预览：
[研究报告预览](https://htmlpreview.github.io/?https://github.com/yangminggulab/us-alpha-lab/blob/main/research_report.html) /
[方法说明预览](https://htmlpreview.github.io/?https://github.com/yangminggulab/us-alpha-lab/blob/main/methodology.html)。

## 1. 准备环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
cp .env.example .env
```

然后把 `.env` 里的 `MASSIVE_API_KEY` 换成你的 Massive key。Massive 官方 Python 客户端支持通过 `MASSIVE_API_KEY` 环境变量读取 key。

## 2. 跑通第一条链路

```bash
alpha-lab fetch --tickers AAPL,MSFT,NVDA --start 2024-08-15 --end 2026-08-14
alpha-lab factors
alpha-lab discover-factors --config configs/universe_free_50.yaml
alpha-lab train
alpha-lab ml-alpha --config configs/universe_free_50.yaml --output-path data/processed/factors_free_50_ml.parquet
alpha-lab report
alpha-lab charts
alpha-lab backtest
alpha-lab leaderboard --config configs/universe_free_50_ml.yaml
alpha-lab alpha-cluster --config configs/universe_free_50_ml.yaml
alpha-lab registry
alpha-lab run-experiment
```

生成文件：

- `data/raw/daily_bars.parquet`: Massive 下载的日线 OHLCV
- `data/processed/factors.parquet`: 因子表
- `data/processed/factors_free_50_ml.parquet`: 带样本外机器学习预测的因子表
- `data/models/latest_model.joblib`: 第一版机器学习模型
- `reports/*.png`: 因子 IC、分位收益和累计多空收益图
- `reports/backtest.parquet`: 分位多空回测结果
- `reports/factor_discovery/factor_candidates.csv`: 候选因子覆盖率、IC、ICIR 快速筛选表
- `reports/leaderboard/factor_leaderboard.csv`: 因子 IC + 回测排行榜
- `reports/alpha_cluster/alpha_cluster_report.csv`: alpha 错误诊断报告
- `runs/<timestamp>_<name>/`: 一次完整实验的配置、报告、图表和指标快照

## 3. 项目结构

```text
US Alpha Lab
│
├── 项目说明与工程配置
│   ├── README.md
│   ├── docs/architecture.md
│   ├── docs/quant-infra.md
│   ├── pyproject.toml
│   ├── .gitignore
│   ├── .env.example
│   └── .env                 # 本地密钥，不进 Git
│
├── 数据与实验配置
│   ├── configs/universe.yaml
│   ├── configs/universe_free_50.yaml
│   ├── configs/universe_free_50_ml.yaml
│   └── configs/experiments/free_alpha.yaml
│
├── 数据获取与标准化
│   ├── src/us_alpha_lab/config.py
│   └── src/us_alpha_lab/massive_data.py
│
├── Alpha 因子生成
│   ├── src/us_alpha_lab/alpha_registry.py
│   ├── src/us_alpha_lab/operators.py
│   ├── src/us_alpha_lab/formula_factors.py
│   ├── src/us_alpha_lab/feature_engineering.py
│   ├── src/us_alpha_lab/factors.py
│   ├── src/us_alpha_lab/factor_discovery.py
│   └── src/us_alpha_lab/labels.py
│
├── 因子评估与回测
│   ├── src/us_alpha_lab/analysis.py
│   ├── src/us_alpha_lab/backtest.py      # 回测核心：配置、权重、基准、绩效
│   └── src/us_alpha_lab/leaderboard.py
│
├── 风险与基准
│   ├── src/us_alpha_lab/risk.py
│   └── src/us_alpha_lab/benchmark.py
│
├── Alpha 检测集群
│   ├── src/us_alpha_lab/diagnostics.py
│   └── src/us_alpha_lab/alpha_cluster.py
│
├── 机器学习实验
│   ├── src/us_alpha_lab/validation.py
│   └── src/us_alpha_lab/modeling.py     # walk-forward 训练，输出 ml_prediction_5d
│
├── 可视化与命令入口
│   ├── src/us_alpha_lab/visualization.py
│   ├── src/us_alpha_lab/cli.py
│   └── src/us_alpha_lab/__init__.py
│
├── 测试
│   └── tests/test_*.py
│
├── 实验管理
│   └── runs/<timestamp>_<name>/
│
└── 本地产物
    ├── data/
    ├── reports/
    ├── .venv/
    └── build/
```

## 4. 框架数据流

```text
Massive API
  -> data/raw/daily_bars.parquet
  -> alpha-lab factors
  -> data/processed/factors.parquet
  -> alpha-lab discover-factors
  -> reports/factor_discovery/factor_candidates.csv
  -> alpha-lab ml-alpha
  -> data/processed/factors_free_50_ml.parquet
  -> alpha-lab report / charts / backtest / train
  -> reports/ + data/models/
```

当前设计是四层：

- 数据层：`massive_data.py` 只负责下载和保存原始 OHLCV。
- 因子层：`operators.py` 提供通用算子，`formula_factors.py` 管公式因子，`feature_engineering.py` 做清洗和横截面特征。
- 研究层：`analysis.py` 看 IC 和分位收益，`backtest.py` 看组合结果。
- 学习层：`modeling.py` 用因子预测未来收益，后面再加 walk-forward / purged CV。

更完整的框架说明见 [docs/architecture.md](docs/architecture.md)。
计算基础设施、CUDA/GPU 加速路线见 [docs/quant-infra.md](docs/quant-infra.md)。

## 5. 第一版研究假设

当前因子偏教学版，方便你先理解完整流程：

- 短期反转：`reversal_1d`
- 中期动量：`momentum_21d`
- 量能变化：`volume_z_21d`
- 波动率：`volatility_21d`
- 日内位置：`close_to_high`, `close_to_low`
- 公式候选：`alpha_ma_gap_21d`, `alpha_price_to_21d_high`, `alpha_liquidity_quality_21d`

默认标签是未来 5 个交易日收益 `future_return_5d`。模型训练时按日期切分训练集和测试集，避免随机打散导致时间泄漏。

Massive 当前 Stocks Basic 免费层更适合先做最近 2 年日线研究；如果请求更早历史，接口可能只返回免费层允许的最近区间。
默认配置会在每个 ticker 请求之间暂停一下，避免撞到免费版 `5 API Calls / Minute` 的限制。

### 免费层批量拉取脚本

如果想把免费层可用的日线数据尽量完整拉下来，可以用独立脚本按 ticker 续跑下载：

```bash
python scripts/harvest_massive_free.py \
  --config configs/universe_free_50.yaml \
  --rolling-free-window \
  --end previous-weekday \
  --output data/raw/daily_bars_free_50.parquet
```

脚本默认按 Massive Stocks Basic 免费层 `5 API Calls / Minute` 限速，并把每个 ticker
先写入 `data/raw/massive_free_tier_shards/`，中断后再次运行会跳过已覆盖完整日期区间的 shard。
正式跑之前可先检查计划：

```bash
python scripts/harvest_massive_free.py --dry-run --rolling-free-window
```

如果要先用 Massive 免费层的 reference data 发现 active 美股 universe，再批量抓取：

```bash
python scripts/harvest_massive_free.py \
  --discover-active \
  --rolling-free-window \
  --end previous-weekday \
  --output data/raw/daily_bars_massive_active_free.parquet
```

这会额外保存 `data/raw/massive_active_tickers.csv`。全 active universe 在免费层速率下会跑很久，
可先加 `--max-tickers 100` 做小批量验证。

## 6. 下一步可以加什么

- 加入 Massive Flat Files，批量下载更大的历史数据
- 接入公司基本面、新闻、财报事件
- 做 cross-sectional IC、分层回测和交易成本
- 做 walk-forward 验证
- 从 scikit-learn 扩展到 LightGBM / XGBoost
- 建立 benchmark 和 GPU smoke test，再逐步验证 RAPIDS/cuDF、Polars GPU、Dask-CUDA
