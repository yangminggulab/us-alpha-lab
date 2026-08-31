# 文件地图与整理建议

更新日期：2026-08-29

这个文档记录当前项目的文件脉络，区分“主干代码”“权威数据产物”“历史实验产物”和“可清理候选”。项目里的大多数数据、报告、日志都已被 `.gitignore` 排除，不会进入 Git；它们是本地研究产物。

## 一、主干脉络

```text
configs/*.yaml
  -> src/us_alpha_lab/config.py
  -> alpha-lab CLI
  -> data/raw/*.parquet
  -> data/processed/*.parquet
  -> reports/ 或 runs/
```

核心运行链路：

```text
alpha-lab fetch / scripts/harvest_massive_free.py
  -> raw OHLCV
alpha-lab factors
  -> 基础因子 + 公式因子 + 路径/残差因子 + 横截面 rank/zscore
alpha-lab ml-alpha
  -> random_forest / lightgbm / lightgbm_ranker 样本外预测
alpha-lab report / discover-factors / leaderboard / alpha-cluster
  -> IC、分层收益、回测、增量裁决
```

## 二、源码目录

`src/us_alpha_lab/` 是唯一业务代码主目录。

| 文件 | 职责 |
| --- | --- |
| `cli.py` | `alpha-lab` 命令入口，串起数据、因子、模型、报告、回测 |
| `config.py` | 读取研究配置 |
| `massive_data.py` | Massive 日线下载 |
| `factors.py` | 基础 OHLCV 因子、路径因子、残差路径因子 |
| `formula_factors.py` | 公式化 alpha 候选 |
| `operators.py` | rolling、rank、delay、corr 等公式算子 |
| `feature_engineering.py` | 横截面 rank/zscore、winsorize、异常值清洗 |
| `labels.py` | 未来收益标签，以及 rank/zscore/quantile 标签变换 |
| `modeling.py` | RandomForest、LightGBM、LightGBM Ranker、调参 |
| `validation.py` | walk-forward 切分和 embargo |
| `analysis.py` | IC、ICIR、分位收益 |
| `backtest.py` | top-minus-bottom 分位组合回测 |
| `benchmark.py` / `risk.py` | 基准和风险指标 |
| `leaderboard.py` | 合并 IC 与回测指标，形成排行榜 |
| `diagnostics.py` / `verdict.py` / `alpha_cluster.py` | 因子覆盖、稳定性、增量关、裁决 |
| `visualization.py` / `report.py` / `methodology.py` | 图表、HTML 报告、方法页 |
| `alpha_registry.py` | 因子注册表，是因子是否进入评估主流程的开关 |

目前源码没有明显重复模块；职责边界是清楚的。

## 三、配置文件

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `configs/universe.yaml` | 保留 | 6 支股票最小 smoke test 配置 |
| `configs/universe_free_50.yaml` | 保留 | 50 支免费层基线配置 |
| `configs/universe_free_50_ml.yaml` | 历史兼容 | 指向 `factors_free_50_ml.parquet`，用于旧 ML 报告/回测 |
| `configs/universe_sp500.yaml` | 当前主配置 | 当前 SP500 研究主线，raw/factors/model 都指向 SP500 产物 |
| `configs/experiments/free_alpha.yaml` | 可后续更新 | 旧 experiment 配置，仍指向 6 支股票基线数据 |

建议：后续新实验尽量走 `configs/universe_sp500.yaml`，不要再扩散新的“临时配置名”。如果要保留实验参数，用 `runs/<timestamp>_<name>/config.yaml` 留快照。

## 四、当前权威数据产物

| 文件 | 状态 | 规模 |
| --- | --- | --- |
| `data/raw/daily_bars_sp500.parquet` | 当前主 raw 数据 | 248,226 行，502 支，2024-08-28 至 2026-08-28 |
| `data/processed/factors_sp500.parquet` | 当前主因子表 | 248,226 行，100 列，含新增路径/残差因子 |
| `data/models/sp500_model.joblib` | 当前主模型文件 | 最近一次 SP500 模型训练产物 |
| `reports/lightgbm_tuning_sp500.csv` | 当前调参结果 | LightGBM / LambdaRank / rank_xendcg 保守网格搜索结果 |

这四个是目前最容易继续往下推进的主线文件。

## 五、历史/对照数据产物

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `data/raw/daily_bars.parquet` | smoke test | 6 支股票，小样本 |
| `data/processed/factors.parquet` | smoke test | 6 支股票因子表 |
| `data/raw/daily_bars_free_50.parquet` | 旧基线 | 50 支股票原始行情 |
| `data/processed/factors_free_50.parquet` | 旧基线 | 50 支股票因子表 |
| `data/processed/factors_free_50_ml.parquet` | 旧基线 | 50 支股票随机森林 ML alpha |
| `data/processed/factors_sp500_lgbm.parquet` | 历史对照 | 新增路径因子之前生成，只有 76 列 |
| `data/processed/factors_sp500_lgbm_zscore.parquet` | 实验对照 | LightGBM + zscore label |
| `data/processed/factors_sp500_lgbm_ranker.parquet` | 实验对照 | LightGBM Ranker + quantile label |
| `data/processed/factors_sp500_rank_xendcg_top_bottom.parquet` | 新实验候选 | rank_xendcg + top/bottom label，尚需重新生成 |

注意：`factors_sp500_lgbm.parquet` 比当前 `factors_sp500.parquet` 少列，容易误用。继续研究时应优先从 `factors_sp500.parquet` 重新生成 ML 输出。

## 六、下载分片和日志

| 路径 | 状态 | 说明 |
| --- | --- | --- |
| `data/raw/massive_sp500_shards/` | 保留 | SP500 成功分片 502 个，`failures.csv` 记录 `PCAR` 限流失败 |
| `data/raw/massive_free_500_shards/` | 可归档或删除 | “active 500” 下载中断残留，只有 9 个分片 |
| `logs/massive_free_harvest*.log` | 可归档 | 下载历史日志 |
| `runs/massive_free_harvest/` | 可归档 | launchd/PID/latest log 状态文件，当前任务不在运行 |

建议：如果近期只走 SP500 主线，`massive_free_500_shards/` 可以移动到归档目录或删除；不要和 `massive_sp500_shards/` 混用。

## 七、报告与实验产物

`reports/` 是散装报告区，`runs/` 是完整实验快照区。

当前有多组历史命名：

- `free_50_*`
- `*_feature_v2*`
- `*_ml`
- `risk_benchmark_*`
- `backtest_module_check`

这些不是代码重复，而是历史实验产物重复。建议后续统一规则：

```text
runs/YYYYMMDD_HHMMSS_<experiment_name>/
  config.yaml
  ic_report.csv
  factor_leaderboard.csv
  alpha_registry.csv
  metrics.json
  charts/
  leaderboard_backtests/
  alpha_cluster/
```

`reports/` 只放最新或手工指定的轻量输出，完整实验都进 `runs/`。

## 八、根目录文件

| 文件/目录 | 状态 | 说明 |
| --- | --- | --- |
| `README.md` | 保留 | 项目总说明和运行入口 |
| `index.html` / `research_report.html` / `methodology.html` | 发布产物 | GitHub Pages 或本地展示用，可由命令重新生成 |
| `docs/` | 保留 | 架构、计算基础设施、学习路线、文件地图 |
| `scripts/` | 保留 | 批量下载和状态脚本 |
| `tests/` | 保留 | 当前 43 个测试 |
| `.venv/` | 本地环境 | 已忽略，占空间最大但不进 Git |
| `build/` / `src/us_alpha_lab.egg-info/` | 可清理 | Python 打包/安装残留，可重新生成 |
| `.DS_Store` / `__pycache__/` / `.pytest_cache/` / `.ruff_cache/` | 可清理 | 系统/测试/工具缓存 |
| `dx-quant/` | Git submodule | 独立 LaTeX 量化笔记仓库，remote 为 `git@github.com:yangminggulab/dx-quant.git`，主仓库只记录指针 |
| `todo.md` / `docs/ml-learning.md` | 未跟踪资料 | 看起来是学习路线/TODO，是否纳入 Git 需要单独决定 |
| `凯读对冲基金面试准备.md` / `截屏*.png` | 个人资料 | 已被 `.gitignore` 忽略，不属于项目主线 |

## 九、混乱点结论

当前没有严重的源码重复，真正容易混淆的是这些：

1. `data/processed/factors_sp500*.parquet` 有多版实验表，其中 `factors_sp500.parquet` 才是当前主表。
2. `data/raw/massive_free_500_shards/` 是未完成的 active-500 下载残留，不应当再被误认为 SP500。
3. `reports/` 里历史实验目录较多，建议后续统一用 `runs/` 保存完整快照。
4. `dx-quant/` 已作为 Git submodule 集成，只记录仓库指针，不混入主项目导入、测试或打包。
5. 根目录 HTML 是发布产物，源码在 `src/us_alpha_lab/report.py` 和 `src/us_alpha_lab/methodology.py`。

## 十、建议的下一步整理动作

不改变研究结果的安全整理顺序：

1. 保留 `data/raw/daily_bars_sp500.parquet`、`data/processed/factors_sp500.parquet`、`reports/lightgbm_tuning_sp500.csv` 作为当前主线。
2. 把 `factors_sp500_lgbm*.parquet` 统一视为实验对照；需要新实验时从 `factors_sp500.parquet` 重新生成。
3. 把 `data/raw/massive_free_500_shards/`、旧下载日志、旧散装 reports 归档，或者确认不需要后删除。
4. 清理 `.DS_Store`、`__pycache__`、`.pytest_cache`、`.ruff_cache`、`build/`、`src/us_alpha_lab.egg-info/` 这类可再生缓存。
5. 保持 `dx-quant/` 作为 submodule；更新笔记时在子模块内提交，再回到主仓库提交新的 submodule 指针。
