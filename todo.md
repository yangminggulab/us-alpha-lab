# ML + PyTorch 学习 TODO

> 结合 us_alpha_lab 项目练习，目标是 JD 的"熟练掌握" = **不看资料能讲原理 + 上手能改参数看出效果**。
> 教学说明看 [docs/ml-learning.md](docs/ml-learning.md)，代码在 `src/us_alpha_lab/modeling.py`。
> 每项打勾标准：不看资料能口头讲清"为什么"，并且动手跑过、能看出效果。

## 一、机器学习：随机森林 / LightGBM / XGBoost

### A. 随机森林（代码在 [modeling.py:78-92](src/us_alpha_lab/modeling.py:78)）

- [ ] 加 `oob_score=True`，打印 OOB R²，讲清"为什么不需要单独验证集"
- [ ] 打印 `feature_importances_`，对到因子名，讲"怎么用它筛因子"
- [ ] 改 `min_samples_leaf` = 5 / 20 / 100，观察 MAE 变化，讲"叶子越少越容易过拟合"
- [ ] 不看资料讲出：随机在哪（bootstrap + 特征子集）、Gini 公式、误差 = 偏差² + 方差

### B. LightGBM（代码在 [modeling.py:48-190](src/us_alpha_lab/modeling.py:48)）

- [ ] `learning_rate` 0.03 → 0.3，观察验证集先降后升，讲"和 `n_estimators` 此消彼长"
- [ ] `num_leaves` 8 vs 512，观察过拟合，讲"leaf-wise 为什么容易过拟合"
- [ ] 对比 `lightgbm + zscore`、`lightgbm_ranker + quantile`、`rank_xendcg + top_bottom`，记录 OOS IC / ICIR
- [ ] 讲清 `bagging_freq=1` 为什么要和 `subsample` 一起开，否则行采样不会按预期生效
- [ ] 讲清 `top_bottom` 标签为什么丢掉中间分位：中间股票未来收益噪声最大，先学头尾差异
- [ ] 不看资料讲出：每轮拟合残差（负梯度）、level-wise vs leaf-wise、`reg_alpha`/`reg_lambda` 在罚什么

### C. XGBoost（扩展，目前项目未接入）

- [ ] 在 [modeling.py:29](src/us_alpha_lab/modeling.py:29) `normalize_model_name` 加 `xgboost` 分支，接入同一 walk-forward
- [ ] 跑 RF / LGBM / XGB 三模型对比，记录 MAE / R² / 方向准确率表
- [ ] 不看资料讲出 XGBoost 与 LGBM 差异：level-wise、二阶梯度、显式正则 $\Omega(f) = \gamma T + \tfrac12\lambda\|\mathbf{w}\|^2$

### D. 量化落地（教程没有、面试必问）

- [ ] 讲清防前视：[modeling.py](src/us_alpha_lab/modeling.py) `shift(feature_lag)` 和 walk-forward `embargo`
- [ ] 讲清为什么树模型不需要 `StandardScaler`（对比 MLP 需要）
- [ ] 讲清缺失值为什么用 median 不用 mean（[modeling.py](src/us_alpha_lab/modeling.py)）

### E. 遗忘机制（市场非平稳）

- [ ] 在 `ml-alpha` 和 `tune-lightgbm` 加 `--time-decay-half-life` 参数，支持 `none / 126 / 252 / 504`
- [ ] 实现指数时间衰减样本权重：`weight = 0.5 ** (age_days / half_life)`，传给 LightGBM 的 `sample_weight`
- [ ] 对比 `train_size` = 126 / 252 / 504，判断硬滚动窗口和软时间衰减谁更稳
- [ ] 加简单 regime 特征：`market_return_21d`、`market_volatility_21d`，分别看高波/低波、上涨/下跌 regime 下 IC
- [ ] 记录结论：旧样本是帮忙还是污染；如果 504 + decay 优于 252，说明老样本有用但需要降权
- [ ] 不看资料讲出：市场非平稳、样本半衰期、硬遗忘 vs 软遗忘、regime 条件复用

## 二、量化研究：价格行为与交易员行为因子

目标：把“技术分析看到的图形”翻译成“交易员行为假设”，再写成可验证、可回测、可解释的因子。

### A. 技术分析语言翻译

- [ ] 把突破、支撑、阻力、放量、缩量、跳空、假突破分别写成一句交易员行为假设
- [ ] 给每个形态标注可能的参与者：趋势资金、止损盘、价值资金、做市商、被动资金、散户追涨杀跌
- [ ] 判断每个形态更可能预测收益方向、波动率、成交量，还是只预测风险
- [ ] 不看资料讲出：画线本身不是 alpha，背后的订单、流动性和行为偏差才可能是 alpha

### B. 可计算因子实现

- [ ] 突破：实现 `alpha_breakout_60d`，例如 `close / rolling_high_60 - 1`
- [ ] 支撑/超跌：实现 `alpha_distance_to_low_120d` 或 rolling low reversal
- [ ] 假突破：实现突破后 1-3 日回落的 `breakout_failure` 候选
- [ ] 缩量整理：组合 `range_compression`、`volatility_contraction`、`volume_contraction`
- [ ] 放量确认：实现 `return_5d * volume_z_21d` 或 volume shock confirmation
- [ ] 跳空压力：扩展 `alpha_gap_pressure_21d`，区分 gap continuation 和 intraday reversal

### C. K 线数学结构

- [ ] 单根 K 线几何：把 `(open, high, low, close)` 转成实体、上影线、下影线、振幅、收盘位置、实体占比、跳空
- [ ] 形态区域化：用无量纲比例定义 doji、hammer、long upper shadow、engulfing 等，不用主观画图
- [ ] 多根 K 线统计识别：读 Lo, Mamaysky, Wang《Foundations of Technical Analysis》，理解非参数形态识别和条件收益分布检验
- [ ] 蜡烛图实证检验：读 Caginalp & Laurent《The predictive power of price patterns》，看 OHLC candlestick pattern 如何做样本外统计检验
- [ ] 路径结构：学习 path signature / shapelets，把一段 K 线路径编码成顺序敏感的特征
- [ ] 事件时间：学习 directional-change / intrinsic time，用价格移动事件替代固定时间 K 线
- [ ] 实现候选因子：`alpha_candle_body_ratio`、`alpha_close_location_value`、`alpha_shadow_pressure`、`alpha_directional_change_count`
- [ ] 不看资料讲出：K 线不是图片，而是 OHLC 约束四元组、路径片段和事件序列

### D. 微观结构学习

- [ ] 读 Hasbrouck《Empirical Market Microstructure》：bid-ask spread、order flow、price impact、liquidity、informed trading
- [ ] 读 O'Hara《Market Microstructure Theory》：做市商库存、信息不对称、战略交易者、价格形成
- [ ] 读 Kyle model / Glosten-Milgrom model：理解知情交易者、做市商、order flow 和 adverse selection 如何共同决定价格
- [ ] 读 [Probability of Informed Trading / PIN model](https://acaciafund.org/markets/research/easley-1996-probability-informed-trading-pin/)：理解如何从买卖单到达率估计信息交易概率
- [ ] 把书里的概念对应到日线可观测 proxy：价差不可见时用振幅、成交额、换手、跳空、日内位置做代理
- [ ] 不看资料讲出：为什么同一个突破，在高流动性和低流动性股票里含义不同

### E. 隐藏参与者与博弈结构（进阶）

- [ ] 读 [Hidden Participation and the Timing of Price Discovery](https://www.mdpi.com/2227-7390/14/16/3019)：重点看隐藏参与者是否在场、Bayesian posterior、Kyle-type price discovery
- [ ] 读 [A Hidden Markov Process Approach to Information-Based Trading](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2021557)：重点看用 HMM 从 order flow 推断信息状态和交易动机
- [ ] 读 [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668)：重点看盘口空间结构 + 时间依赖如何被 CNN/LSTM 编码
- [ ] 读 [Deep limit order book forecasting: a microstructural guide](https://discovery.ucl.ac.uk/id/eprint/10218102/)：重点看为什么高预测准确率不等于可交易信号
- [ ] 读 [ABIDES-MARL: Optimal Execution with Endogenous Liquidity](https://academ.us/article/2511.02016/)：重点看多智能体执行、内生流动性和战略互动
- [ ] 读 [Multi-Agent Reinforcement Learning for Market Making](https://doi.org/10.1145/3768292.3770388)：重点看不同做市 agent 的竞争、适应和 interaction-level metrics
- [ ] 读 [MAGAT: multi-agent game-theoretic adversarial trading](https://www.nature.com/articles/s41598-026-60518-6)：只作为仿真/鲁棒训练参考，不直接当作实盘证据
- [ ] 把研究问题写成链路：可观测市场数据 → 推断隐藏参与者类型 → 推断目标/约束 → 预测下一步动作 → 预测价格冲击
- [ ] 定义隐藏状态集合：`机构买入`、`机构卖出`、`做市库存调整`、`套利修复`、`被迫卖出`、`噪声交易`
- [ ] 用日线 proxy 先做低频版 latent-state 特征：成交量冲击、连续小幅趋势、跳空延续、振幅扩张、收盘位置、反转失败
- [x] 实现 HMM / Bayesian filtering 原型：估计 `P(Z_t | X_1:t)`，输出每只股票每日最可能的参与者状态
- [x] 把参与者状态转成候选因子：机构吸筹概率、被迫卖出后反弹概率、做市库存压力、套利修复压力
- [x] 把 Top latent participant 状态概率接入 `validate-experiment`：当前 best 为 `alpha_latent_arbitrage_repair_prob`，正交 IC 0.0141，但五关未达标
- [ ] 做策略互动 memo：机构拆单 vs 做市商 adverse selection、止损盘 vs 反向流动性提供者、趋势资金 vs 套利资金
- [ ] 记录边界：日线只能看到行为痕迹，不能证明真实身份；要用正交 IC、样本外和事件复盘约束解释
- [ ] 不看资料讲出：这不是“看 K 线猜心理”，而是 observable data → latent agent inference → strategic action prediction

### F. 验证与 memo

- [ ] 每个 price-action 因子跑样本外 IC / ICIR / positive IC rate
- [ ] 跑分位收益和成本后 top-minus-bottom 回测
- [ ] 做市场/行业/size/beta/momentum 中性化，判断是否只是旧因子的变体
- [ ] 做 regime 分层：高波/低波、上涨/下跌、放量/缩量环境
- [ ] 每个因子写一页 memo：表象、交易员行为、可计算定义、样本外结果、失效机制

## 三、前沿论文转实验

原则：论文只读和我们当前主线有关的部分。每篇至少落成一个实验、一个指标或一条研究纪律，否则先不深入。

### A. Kronos：金融 K 线基础模型

- [ ] 读 [Kronos: A Foundation Model for the Language of Financial Markets](https://neurips.cc/virtual/2025/130441)，重点看 K 线 tokenizer、预训练任务、RankIC 评估
- [x] 把 OHLCV 路径改写成 token/离散状态：涨跌幅桶、振幅桶、成交量冲击桶、收盘位置桶
- [x] 实现一组轻量版“序列 K 线因子”：最近 20/60 日 token n-gram 频率、路径相似度、状态转移概率
- [x] 和现有 `range_compression`、`gap_pressure`、`intraday_quality` 做正交化，判断是否有增量 IC
- [x] 记录结论：K 线基础模型的价值在我们这里是“路径表征”，不是直接相信模型输出
- [x] 升级为 K 线 token 自动发现器：批量枚举状态、转移、组合 token，输出正交 IC Top 候选
- [x] 抽出统一实验检验模块：任意实验 `alpha_*` 输出都可接入 IC、正交 IC、五关裁决和扣费回测
- [ ] 把 Top K 线 pattern 候选并入五关裁决和扣费组合回测，判断是否能从“增量信息”走到“可交易候选”

### B. OFR Benchmark：金融预测公平比较

- [ ] 读 [Do Deep Learning Methods Improve Financial Forecasts?](https://www.financialresearch.gov/the-ofr-blog/2026/08/25/deep-learning-methods-improve-financial-forecasts/)，重点看固定数据、固定切分、统一基准的比较方式
- [ ] 给 `ml-alpha` 输出增加统一 benchmark 表：historical mean / Ridge / RF / LightGBM / ranker 的 OOS MAE、R²、IC、ICIR
- [ ] 所有模型使用同一份 `train/valid/test` walk-forward 切分，避免“换模型也换数据”的假优势
- [ ] 增加 naive baseline：预测 0、历史均值、行业/市场暴露基线，确认模型是否真的超过简单基准
- [ ] 记录结论：深度学习只在有稳定重复结构时值得上；短期收益预测先尊重强基线

### C. Practitioner Pipeline：横截面收益预测流程

- [ ] 读 [Cross-Sectional Return Prediction Using Machine Learning: A Practitioner Pipeline](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6675380)，重点看 Ridge/XGBoost/LightGBM、walk-forward、IC、成本后 Sharpe、Deflated Sharpe Ratio
- [ ] 接入 `Ridge` / `ElasticNet` 作为强线性基线，和 RF / LightGBM / ranker 同表比较
- [ ] 接入 XGBoost 后，用同一套 walk-forward 输出 RF / LGBM / XGB / Ridge 对比表
- [ ] 在裁决表加入 Deflated Sharpe Ratio 或至少 Probabilistic Sharpe Ratio，减少短样本高 Sharpe 误判
- [ ] 输出每次实验的“可复现卡片”：数据版本、universe、日期区间、模型参数、成本假设、是否中性化

### D. LLM / Agent 论文：只服务研究流程

- [ ] 读 [LLMs Get Lost In Multi-Turn Conversation](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)，把启发落到研究记录：每轮实验必须有固定问题、固定配置、固定裁决标准
- [ ] 读 [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)，作为“不急着做 RL 交易”的反例提醒
- [ ] 如果以后做自动挖因子 Agent，先让 Agent 生成候选定义和 memo，再由现有五关裁决自动否决或通过
- [ ] 不做直接 RL 下单，除非已经有稳定 alpha、真实成本模型、组合约束和可验证 reward

## 四、跨资产扩展：期货、股指期货与货币

原则：先把股票日线多因子框架做硬，再扩资产；扩资产时先 ETF/股指代理，再股指期货，最后普通期货和 FX。

### A. 暂缓引入的条件

- [ ] SP500 主线完成：数据稳定、因子库、LightGBM/ranker、遗忘机制、价格行为因子、行业/市场中性、成本后回测
- [ ] 至少写出 3 篇因子 memo：假设、数据、定义、样本外 IC、分位收益、成本后收益、失效机制
- [ ] 明确当前股票项目的问题是“需要对冲/宏观 regime”，而不是“股票框架还没验证清楚”
- [ ] 不看资料讲出：为什么期货/FX 会引入合约切换、展期、交易时间、保证金、报价 convention 等新复杂度

### B. 第一步：ETF / 指数代理

- [ ] 加入 SPY、QQQ、IWM 和 sector ETF，作为市场、风格、行业、risk-on/risk-off proxy
- [ ] 做市场 beta hedge：用 SPY/QQQ 回归股票因子收益，观察残差 alpha
- [ ] 加 regime 特征：指数 21/63 日收益、指数波动、市场宽度、sector dispersion
- [ ] 对比裸股票多空 vs ETF hedge 后组合的收益、波动、回撤

### C. 第二步：股指期货

- [ ] 学 continuous futures construction：主力合约、换月、back-adjustment、roll return
- [ ] 加 ES / NQ / RTY，先做股指 hedge 和 basis 研究，不急着做独立期货 alpha
- [ ] 研究 basis / 贴水升水 / 到期日效应 / 套保盘压力
- [ ] 回测加入保证金、乘数、交易时间、展期成本和滑点

### D. 第三步：宏观期货与 FX

- [ ] 商品/利率/外汇先从 trend、carry、value 三类经典因子开始
- [ ] 学 cross-asset volatility targeting，把不同资产波动缩放到可比较尺度
- [ ] 加 FX pairs 时先搞清楚报价方向、利差/carry、24 小时交易和宏观事件日
- [ ] 形成第二条研究线：cross-asset trend/carry/value + risk parity / crisis alpha

## 五、PyTorch：从 0 到熟练掌握

路线：手写 MLP（地基）→ 训练循环（核心）→ 手搓 transformer（封顶）。

### A. 手写 MLP（作为第三模型，替换/并列 RandomForest）

- [ ] 用 `nn.Module` 写 MLP：`Linear → ReLU → Linear`
- [ ] 手写训练循环五行 `forward → loss → zero_grad → backward → step`，不看资料讲清每行在干嘛
- [ ] 讲清为什么 MLP 需要 `StandardScaler` 而树不需要（梯度对尺度敏感）
- [ ] 跑 RF / LGBM / MLP 同一 walk-forward 对比，记录 MAE / R²，得出"表格小数据上树更好"的结论

### B. 训练细节（熟练掌握的分水岭）

- [ ] 过拟合：train loss 降、val 不降 → 用 dropout / L2 / 早停 / 减小模型 修
- [ ] 欠拟合：两者都高 → 增大模型 / 降正则 / 加特征
- [ ] loss 变 NaN → 学习率太大 / 梯度爆炸 → `clip_grad_norm_`、打印梯度范数排查
- [ ] 学习率：固定 vs 余弦退火 vs warmup，画曲线对比
- [ ] optimizer：SGD vs Adam 区别，为什么 Adam 不用手调学习率那么多
- [ ] `model.train()` vs `.eval()`、`torch.no_grad()`、`.to(device)` 何时用、为什么

### C. 手搓小型 Transformer（封顶作业）

- [ ] 写 self-attention：$Q = XW_Q,\ K = XW_K,\ V = XW_V$，$\text{softmax}(QK^\top/\sqrt{d})V$
- [ ] Multi-head：多组并行 + 拼接
- [ ] 残差 + LayerNorm + MLP 块（复用 A 的 MLP）
- [ ] 位置编码（sinusoidal）
- [ ] 用同一数据集跑一遍，记录结果
- [ ] 不看资料讲出：为什么 transformer 在横截面小数据上大概率不如树（数据规模万级、信噪比低）

### D. PyTorch 工程化（面试加分）

- [ ] 用 `torch.save` / `load` 存模型，接入 `joblib` 同款 dump/load 流程
- [ ] 用 GPU（`.to("cuda")`）或 MPS，对比训练速度
- [ ] 写一个 `EarlyStopping` 类（monitor val loss、patience、恢复 best weights）
- [ ] 用 TensorBoard 或 wandb 记录训练曲线

## 达标标准

把上面每一项带 `[ ]` 的点，不看资料口头讲一遍；能讲满 = 熟练，讲不出 = 该补。顺序建议按 A → B → C → D 走，每节一次会话量。

## 进度

- 机器学习：RF 未开始 / LightGBM 已接入 rank_xendcg/top_bottom，遗忘机制待实现 / XGBoost 未接入
- 价格行为因子：已形成学习路线，隐藏参与者低频 HMM 原型已实现；待实现 breakout / support / failure / volume shock / K 线数学结构候选
- 前沿论文转实验：Kronos 轻量 K 线路径 token 因子 MVP 已实现，K 线 pattern 自动发现器已接入主报告，统一实验检验模块已抽出；待读 Kronos / OFR benchmark / practitioner pipeline；待实现 Ridge 基线、DSR/PSR
- 跨资产扩展：先不引入期货/FX；股票主线成熟后先 ETF/股指代理，再股指期货，最后宏观期货和货币
- PyTorch：未开始
