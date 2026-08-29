# Quant Infra

这个文件记录 US Alpha Lab 后续做“大一点的因子库 / 更长历史 / 更多股票”时的计算基础设施路线。目标不是一开始就把项目改成重型平台，而是先知道哪些地方会慢、应该怎样测、什么时候值得上 CUDA/GPU。

> 更新日期：2026-08-29。GPU 生态变化很快，安装命令和版本兼容以官方文档为准。

## 1. 当前项目的计算热点

当前框架主要用 `pandas`、`numpy`、`scikit-learn` 和 `lightgbm`：

- 因子生成：`src/us_alpha_lab/factors.py` 里大量 `groupby("ticker")`、`rolling`、`pct_change`。
- 公式因子：`src/us_alpha_lab/operators.py` 里有 `rolling_mean`、`rolling_std`、`rolling_max`、`ts_rank`、`rolling_corr`、`decay_linear`。
- 横截面处理：`src/us_alpha_lab/feature_engineering.py` 和 `factors.py` 里按 `date` 做 rank、z-score、winsorize。
- 回测评估：`analysis.py`、`backtest.py`、`leaderboard.py` 按日期循环算 IC、分位组合和收益曲线。
- 机器学习：`modeling.py` 里 walk-forward 反复训练 `RandomForestRegressor(n_jobs=-1)`、`LGBMRegressor` 或 `LGBMRanker`。

一般瓶颈顺序大概率是：

1. 数据量变大后的 parquet 读写和排序。
2. per-ticker rolling/correlation。
3. per-date 横截面 rank、quantile、z-score。
4. walk-forward 多折训练。
5. 大量 alpha 候选的重复计算。

## 2. 加速原则

先做基准，再换引擎。GPU 不是魔法按钮，尤其是小数据时 CPU 到 GPU 的数据搬运会抵消收益。

推荐先固定一个 benchmark：

```bash
alpha-lab factors
alpha-lab discover-factors --config configs/universe_free_50.yaml
alpha-lab ml-alpha --config configs/universe_free_50.yaml
alpha-lab tune-lightgbm --config configs/universe_sp500.yaml --max-trials 24
alpha-lab leaderboard --config configs/universe_free_50_ml.yaml
```

每次记录：

- 输入规模：ticker 数、交易日数、行数、alpha 数量。
- 运行时间：fetch / factors / discovery / ml-alpha / leaderboard。
- 峰值内存：本机可用 `time` 或 Python profiler，云机器可看监控面板。
- 产物一致性：随机抽样比较 CPU 和新引擎输出，允许浮点误差，但不允许排序、日期泄漏、分组边界错位。

## 3. CPU 侧先做好的事

即使未来上 CUDA，也应该先把 CPU 路线整理干净：

- 数据保存为 parquet，并尽量只读需要的列。
- `ticker`、`date` 排序只做一次，后续函数默认输入已排序。
- 价格、收益、因子可以考虑 `float32`，金额或需要高精度的地方继续用 `float64`。
- 对多个因子共享的中间量做缓存，比如 `ret_1d`、`volume_mean_21d`、`close_rank`。
- 减少 Python 层 `for date, group in data.groupby("date")` 循环，优先用向量化 groupby/transform。
- walk-forward 训练天然可并行：每个 fold 独立，后面可把 fold 级别拆成 joblib / Ray / Dask task。

CPU 阶段适合先加一个 `alpha-lab benchmark` 命令，把几个核心步骤的耗时输出到 `reports/benchmarks/*.json`，这样后面换 GPU 才知道有没有真的变快。

## 4. CUDA / GPU 路线图

### 阶段 A：低侵入验证 cuDF pandas 加速

RAPIDS 的 `cudf.pandas` 可以在 pandas API 层做 GPU 加速，支持的操作跑 GPU，不支持的操作自动回退到 pandas。它适合做第一轮可行性验证，因为项目当前主要就是 pandas 代码。

适用场景：

- 想快速判断 `groupby`、`rolling`、`read_parquet` 这类操作能不能从 GPU 获益。
- 不想马上重写 `factors.py` / `operators.py`。

注意：

- 小数据不一定更快。
- 回退到 CPU 的操作越多，收益越小。
- 必须检查 CPU/GPU 输出一致性，特别是 rolling rank、rolling corr、缺失值和排序。

实验方式：

```bash
# 在有 NVIDIA GPU 的 Linux / WSL2 / 云机器上
python -m cudf.pandas path/to/benchmark_script.py
```

如果后面要把 CLI 也纳入验证，可以新增一个很薄的 benchmark 脚本，调用 `us_alpha_lab` 里的现有函数，而不是马上改业务代码。

### 阶段 B：显式 cuDF 因子引擎

如果阶段 A 有明显收益，再考虑把因子计算拆成后端：

```text
pandas backend: 当前稳定实现
cudf backend: GPU dataframe 实现
```

建议的接口形状：

```python
def add_alpha_factors(frame, backend: str = "pandas"):
    ...
```

或者更干净一点：

```text
src/us_alpha_lab/factor_engine/pandas_engine.py
src/us_alpha_lab/factor_engine/cudf_engine.py
```

优先迁移的算子：

- `delay` / `delta`
- rolling mean / std / max
- cross-sectional rank
- per-ticker pct_change
- `volume_z_21d`

谨慎迁移的算子：

- `rolling_corr`：要确认 cuDF 的分组 rolling/corr 能不能完全表达当前逻辑。
- `ts_rank`：不同实现对 tie、NaN、窗口不足的处理可能不同。
- `decay_linear`：当前用了 rolling apply + Python lambda，这类自定义函数通常是 GPU 加速最容易失效的地方，可能需要改成卷积/矩阵化或自定义 kernel。

### 阶段 C：Polars Lazy + GPU engine

Polars 的 GPU 支持基于 RAPIDS cuDF，走 Lazy API。它适合未来把“批量筛很多 alpha 候选”的流程改成列式 query plan。

适用场景：

- 数据规模上到几千万行或更高。
- 因子表达式能写成 Polars expression。
- 需要查询优化、投影下推、延迟执行。

不适合作为当前第一步，因为项目已经是 pandas 结构，而且 Polars GPU 支持仍在快速发展阶段。

### 阶段 D：CuPy / CUDA Python / 自定义 kernel

只有在 dataframe 引擎无法覆盖瓶颈时，才考虑写更底层的 CUDA kernel。

候选场景：

- 大规模矩阵形式的因子批量计算：`date x ticker x factor`。
- 可转成数组运算的 rolling window。
- 同一个窗口权重反复复用，比如 `decay_linear`。
- 极重的相关系数、协方差、风险模型计算。

可选工具：

- CuPy：NumPy 风格 GPU 数组，适合数组运算和与 CUDA 库互操作。
- CUDA Python / cuda-python：更底层地访问 CUDA runtime、bindings 和并行算法。
- Numba-CUDA：可以写 Python CUDA kernel，但官方文档已有维护模式提示，长期路线要关注 `numba-cuda-mlir`。

原则：能用成熟库就不要手写 kernel。手写 CUDA 的维护成本高，debug 难，收益必须由 benchmark 证明。

### 阶段 E：Dask-CUDA / 多 GPU

当单 GPU 显存不够，或需要把 universe 扩到全美股、多年分钟线，再考虑 Dask-CUDA。

适用场景：

- 多 GPU 单机。
- 多节点 GPU 集群。
- 每个 ticker / 每段日期 / 每组 alpha 可以拆成独立 partition。

拆分建议：

- 按 ticker 分区：适合 per-ticker rolling。
- 按日期分区：适合横截面 rank，但 rolling 需要处理分区边界。
- 按 alpha 分区：适合 Alpha Cluster 和候选因子大批量评估。

## 5. 推荐环境

本地 macOS 适合开发、测试、读报告；CUDA 适合放到 NVIDIA GPU 环境里跑。

最低建议：

- OS：Linux，或 Windows 11 + WSL2。
- GPU：NVIDIA Volta 或更新架构，compute capability 7.0+。
- CUDA：CUDA 12 或 CUDA 13，和驱动版本匹配。
- 内存：系统内存最好大于 GPU 显存，RAPIDS 文档建议约 2:1 的系统内存/GPU 显存比例。
- 存储：NVMe SSD，parquet 读写会明显受磁盘影响。

安装策略：

- 稳定研究环境优先用 conda/mamba 或 Docker。
- pip 安装 RAPIDS 时必须选和 CUDA 匹配的 wheel，比如 `-cu12` 或 `-cu13`。
- 不要把 GPU 依赖直接塞进默认 `pyproject.toml` 依赖；建议未来加 optional extra，例如 `gpu-cu12`、`gpu-cu13`，或单独提供 `environment-gpu.yml`。

未来可以新增：

```text
envs/environment-cpu.yml
envs/environment-gpu-cu12.yml
envs/environment-gpu-cu13.yml
```

## 6. 项目落地顺序

### 第一步：加 benchmark

新增一个命令或脚本，固定输入、输出耗时和结果摘要：

```text
alpha-lab benchmark --config configs/universe_free_50.yaml
```

输出：

```text
reports/benchmarks/<timestamp>.json
reports/benchmarks/<timestamp>.csv
```

指标：

- 数据行数、ticker 数、日期数。
- 每个阶段耗时。
- 内存峰值。
- top 10 最慢函数。

### 第二步：整理因子算子边界

把 `operators.py` 视为未来 backend 切换的核心边界。先保证每个算子都有测试：

- index 顺序不变。
- group 边界不穿透。
- NaN 和 inf 行为稳定。
- CPU/GPU 误差容忍用 `np.testing.assert_allclose`。

### 第三步：做 cuDF pandas smoke test

在 GPU 机器上跑同一份小数据：

```text
CPU pandas output
GPU cudf.pandas output
```

检查：

- 因子列是否齐全。
- 每列非空数量是否一致。
- 每列均值/标准差是否接近。
- 抽样 1000 行逐值比较。

### 第四步：只迁移收益最大的 2-3 个算子

优先考虑：

- `rolling_mean`
- `rolling_std`
- `cs_rank`
- `rolling_corr`

不要一口气重写全项目。每迁移一个算子，都要保留 pandas 实现作为 oracle。

### 第五步：训练侧再考虑 GPU

当前模型支持 `RandomForestRegressor`、`LGBMRegressor` 和 `LGBMRanker`，都已经用 `n_jobs=-1` 吃 CPU 多核。后续如果训练变慢，可以比较：

- CPU RandomForest baseline。
- LightGBM CPU vs LightGBM GPU。
- XGBoost GPU hist。
- RAPIDS cuML 随机森林或梯度提升。

训练侧要特别小心时间序列泄漏：加速不能改变 walk-forward、embargo、feature lag。

## 7. 结果一致性检查清单

每次换计算引擎，都检查：

- `ticker,date` 主键是否唯一。
- 排序是否仍为 `ticker,date` 或设计指定顺序。
- rolling 窗口没有跨 ticker。
- 横截面 rank 没有跨 date。
- label 仍然只使用未来 close，不影响 feature。
- feature lag 仍然生效。
- NaN、inf、窗口不足、停牌/缺失数据处理一致。
- 相同输入下 leaderboard 排名大体稳定。

## 8. 什么时候不要上 GPU

- 只有几十只股票、两年日线，CPU 已经秒级或分钟级。
- 主要时间花在 Massive API 限流和网络下载。
- 代码还在频繁改研究逻辑，过早引入 GPU 会拖慢迭代。
- bottleneck 是 Python 循环或 I/O，尚未做向量化和缓存。
- 没有稳定 benchmark，无法证明收益。

## 9. 参考资料

- NVIDIA RAPIDS Installation Guide: https://docs.rapids.ai/install/
- NVIDIA cuDF pandas accelerator: https://docs.nvidia.com/cudf/26.10/cudf_pandas/
- Polars GPU Support: https://docs.pola.rs/user-guide/gpu-support/
- NVIDIA CUDA Python: https://nvidia.github.io/cuda-python/latest/
- Numba-CUDA Installation: https://nvidia.github.io/numba-cuda/user/installation.html
- Dask-CUDA Documentation: https://docs.rapids.ai/api/dask-cuda/stable/
