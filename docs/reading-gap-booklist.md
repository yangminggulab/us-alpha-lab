# 补缺书单：从数学底座到前沿因子研究

更新日期：2026-08-29

依据：

- 网站源码仓库：`/Users/liubike/Desktop/yangminggu.github.io/books.json`
- 网站笔记清单：`/Users/liubike/Desktop/yangminggu.github.io/notes_manifest.json`
- 本地 ML 论文仓库：`/Users/liubike/Desktop/机器学习论文/dx-ML/书单.tex`
- 已读论文目录：`/Users/liubike/Desktop/机器学习论文/已经读完的书/attention is all you need.pdf`

已有主线包括 Real Analysis、Probability Theory、Mathematical Statistics、Functional Analysis、Complex Analysis、Topology、Abstract Algebra、PDE、Numerical ODEs、ML、Python Pandas、SQL、AP；本地 `dx-ML/书单.tex` 还已经覆盖 A/B Test、CTR 预估、推荐系统、Learning to Rank、Wide & Deep、DeepFM、DIN、BST、DLRM、LambdaRank/LambdaMART 等搜广推论文路线。整体判断：你的数学底座和 ML/推荐论文入口已经有了，下一步不该继续堆纯数学或重复推荐 Transformer/CTR 入门，而要补“金融问题怎样被建模、因子怎样被发现、怎样验证不是数据挖掘、怎样变成组合”的知识。

筛选标准：

1. 权威：优先出版社、作者主页、大学课程或领域经典教材。
2. 补缺：不重复你已有的数学/基础 ML，而补资产定价、金融计量、组合、微观结构、金融 ML 验证、高维统计、因果和在线学习。
3. 可落地：每本都对应本项目里能做的一件事。

## P0：先读，最直接补当前项目短板

| 书 | 权威来源 | 它补的知识 | 为什么正好补你现在没学过的 |
| --- | --- | --- | --- |
| Bali, Engle, Murray, *Empirical Asset Pricing: The Cross Section of Stock Returns* | [Wiley](https://uat.store.wiley.com/en-us/empirical-asset-pricing-the-cross-section-of-stock-returns-p-9781118589663) | 横截面收益、组合排序检验、Fama-MacBeth 回归、beta、size、value、momentum、liquidity、idiosyncratic volatility 等经典异象 | 这是“怎么找股票因子”的正统研究手册。你现在已经能算 IC 和 LightGBM 排序，但缺一套学术标准流程：单变量/双变量排序、横截面回归、控制变量、解释因子溢价。 |
| Andrew Ang, *Asset Management: A Systematic Approach to Factor Investing* | [Oxford Academic](https://academic.oup.com/book/3342) | factor risk premium、资产配置、低风险异象、跨资产因子、factor investing、委托投资 | 它补“因子不是技术指标，而是坏时期风险暴露”的投资框架。适合把你项目里的 alpha 从统计信号升级成风险溢价/行为偏差/约束套利的解释。 |
| John H. Cochrane, *Asset Pricing* | [作者主页](https://www.johnhcochrane.com/asset-pricing) | 随机贴现因子、风险溢价、预期收益、折现率、因子定价的经济含义 | 你现在能算 IC、跑 LightGBM，但还缺“为什么这个因子应该赚钱”的资产定价语言。读完后，每个 alpha 都要能说清是风险补偿、行为偏差，还是纯数据噪声。 |
| Campbell, Lo, MacKinlay, *The Econometrics of Financial Markets* | [JSTOR / Princeton University Press](https://www.jstor.org/stable/j.ctt7skm5) | 收益可预测性、随机游走检验、事件研究、CAPM/APT、微观结构、ARCH/非线性金融数据 | 你已有数理统计，但缺“金融市场数据该怎么做统计检验”。它把统计工具换成金融问题，能补 IC 显著性、异常收益、样本外检验的语境。 |
| Ruey S. Tsay, *Analysis of Financial Time Series* | [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470644560) / [作者主页](https://faculty.chicagobooth.edu/ruey-s-tsay/research/analysis-of-financial-time-series-3rd-edition) | ARMA/GARCH、厚尾、波动率、非线性时间序列、多资产收益、状态空间、PCA/因子模型 | 你有 Probability/Stats，但项目里真正碰到的是金融时间序列：非平稳、异方差、厚尾、横截面相关。它补“金融数据不是普通 IID 表格”的部分。 |
| Grinold & Kahn, *Active Portfolio Management* | [McGraw Hill](https://www.mheducation.com/highered/mhp/product/active-portfolio-management-pb.html) | alpha、benchmark、残差收益、信息比率、breadth、风险模型、组合构建、交易成本、绩效归因 | 你现在在“找预测因子”，但还缺“预测分数怎样变成仓位和收益”。这本补从 `ml_prediction_5d` 到 top-minus-bottom、风险约束、换手和成本后的完整链路。 |
| Marcos López de Prado, *Advances in Financial Machine Learning* | [Wiley](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/finance-investments-13fi/finance-investments-special-topics-13fiz/advances-in-financial-machine-learning-978-1-119-48208-6) | 金融标签、样本权重、金融交叉验证、特征重要性、超参调优、回测假阳性、fractional differentiation、微观结构特征 | 这是当前 LightGBM/Ranker 项目最该直接吸收的书。它补的是“金融 ML 不等于 Kaggle”：防泄露、防过拟合、防多重检验、防漂亮回测。 |
| Antti Ilmanen, *Expected Returns* | [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118467190) | 各资产类别长期收益、carry、value、momentum、防御、流动性、宏观风险、预期收益估计 | 它补“不同收益来源的地图”。读完后，你不会只盯股票日线 OHLCV，而能把因子放进更大的 expected return 框架里比较。 |
| Qian, Hua, Sorensen, *Quantitative Equity Portfolio Management* | [Routledge / Chapman & Hall](https://www.routledge.com/Quantitative-Equity-Portfolio-Management-Modern-Techniques-and-Applications/author/p/book/9781584885580) | 量化权益投资流程、alpha 模型、风险模型、交易成本、组合优化、业绩归因 | 它比 APM 更贴近股票多因子工程。补的是“研究员信号如何进入 PM 的组合生产线”，适合对应当前 `alpha_registry`、leaderboard、backtest。 |

## P1：第二批，补“前沿找因子”的研究能力

| 书 | 权威来源 | 它补的知识 | 为什么正好补你现在没学过的 |
| --- | --- | --- | --- |
| Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* | [作者主页](https://hastie.su.domains/ElemStatLearn/main.html) / [Springer](https://link.springer.com/book/10.1007/978-0-387-84858-7) | 模型评估、正则化、树、boosting、随机森林、SVM、无监督、高维问题 | 你已有 `dx-ML`，但当前项目正在用 RandomForest/LightGBM。ESL 补的是这些模型背后的统一统计学习框架，尤其是 bias-variance、model assessment 和 ensemble。 |
| Hastie, Tibshirani, Wainwright, *Statistical Learning with Sparsity* | [作者主页](https://hastie.su.domains/StatLearnSparsity/) | Lasso、稀疏建模、变量选择、广义线性模型稀疏化、结构化稀疏 | 前沿因子不是“造 1000 个特征然后全喂给模型”。这本补如何在大量候选因子里选少数稳定信号，适合接到 `alpha_registry` 和因子裁决流程。 |
| Martin J. Wainwright, *High-Dimensional Statistics* | [Cambridge](https://www.cambridge.org/core/books/highdimensional-statistics/8A91ECEEC38F46DAB53E9FF8757C7A4E) | concentration、uniform law、随机矩阵、协方差估计、高维 PCA、稀疏模型、非渐近误差界 | 量化最大的坑是 p 大、样本短、噪声强。你已有概率统计，但缺高维条件下“为什么样本内好看不代表真有 alpha”的理论。 |
| Joel Hasbrouck, *Empirical Market Microstructure* | [Oxford Academic](https://academic.oup.com/book/52241) | 交易机制、订单流、信息交易、库存、限价订单簿、交易成本、执行策略 | 你目前主要用日线 OHLCV。真正能扩因子的下一层是成交、流动性、订单流、冲击成本。它补微观结构数据该看什么，以及为什么这些变量可能有经济含义。 |
| Miguel Hernán & James Robins, *Causal Inference: What If* | [作者主页](https://miguelhernan.org/whatifbook) / [Routledge](https://www.routledge.com/Causal-Inference-What-If/Hernan-Robins/p/book/9781420076165) | 因果效应、DAG、混杂、选择偏差、IPW、g-formula、target trial emulation | 因子研究很容易把相关性讲成因果故事。这本补“什么情况下你有资格说 X 导致未来收益”，也能帮你识别幸存者偏差、选择偏差和后验叙事。 |
| Maureen O'Hara, *Market Microstructure Theory* | [Wiley](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/accounting-13ac/corporate-finance-13ac3/market-microstructure-theory-978-0-631-20761-0) | 做市商库存模型、信息不对称、战略交易者、价格形成、流动性与市场稳定 | Hasbrouck 偏实证，这本偏机制。它补“订单流/价差/冲击为什么会和收益有关”的经济模型，适合做微观结构 alpha 的解释层。 |
| Boyd & Vandenberghe, *Convex Optimization* | [Stanford 作者页](https://web.stanford.edu/~boyd/cvxbook/) / [Cambridge](https://www.cambridge.org/highereducation/books/convexoptimization/17D2FAA54F641A2F62C7CCD01DFA97C4) | 凸集、凸函数、对偶、KKT、QP、约束优化、数值解法 | 因子研究最后一定落到组合优化：权重、风险、行业中性、换手、成本、杠杆约束。你有分析基础，但还缺“把约束写成可解优化问题”的工程语言。 |

## P2：第三批，补更前沿或相邻方向

| 书 | 权威来源 | 它补的知识 | 为什么正好补你现在没学过的 |
| --- | --- | --- | --- |
| Kevin P. Murphy, *Probabilistic Machine Learning: Advanced Topics* | [MIT Press](https://mitpress.mit.edu/9780262048439/probabilistic-machine-learning/) | 深度生成模型、图模型、贝叶斯推断、变分推断、分布迁移、因果、决策不确定性 | 如果你要从树模型走向更前沿的“表征学习/不确定性/分布变化”，这本比普通深度学习书更适合量化，因为市场问题核心是噪声、后验和不确定决策。 |
| Elad Hazan, *Introduction to Online Convex Optimization* | [MIT Press](https://mitpress.mit.edu/9780262046985/introduction-to-online-convex-optimization/) | regret、expert advice、online learning、在线组合选择、非平稳环境中的学习 | 市场不是静态训练集。它补“模型每期更新、环境漂移、策略后悔值”的语言，适合以后做动态因子权重和在线组合。 |
| Sutton & Barto, *Reinforcement Learning: An Introduction* | [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) | MDP、bandit、TD learning、off-policy、policy gradient、actor-critic | 不建议现在用 RL 找日线 alpha，但它补交易执行、仓位调整、探索-利用、动态决策。读它是为了以后理解 execution/policy，不是为了立刻替代 LightGBM。 |
| Nocedal & Wright, *Numerical Optimization* | [Springer](https://link.springer.com/book/10.1007/b98874) | line search、trust region、Newton/quasi-Newton、约束优化、SQP、interior point | Boyd 告诉你什么问题是凸的；这本告诉你优化器到底怎么跑、为什么不收敛。适合后续调组合优化器、约束优化和深度模型训练。 |
| Steven Shreve, *Stochastic Calculus for Finance II* | [Springer](https://link.springer.com/book/9780387401010) | Brownian motion、Ito calculus、martingale、risk-neutral pricing、连续时间模型、跳扩散、利率模型 | 你有概率论，但如果未来切到期权、波动率、衍生品或风险中性定价，这本补连续时间金融的正统入口。对当前股票日线因子不是最急，但能打开另一个赛道。 |

阅读顺序见 [README「三、学习路线」](../README.md) 的树状图——树按阶段把 P0/P1/P2 的书串起来了，本文件只保留每本书的清单与理由。

## 不优先推荐的方向

1. 暂时不继续堆纯数学分析类书：你已有 Real Analysis、Functional Analysis、Topology、PDE/ODE，边际收益低。
2. 暂时不把深度学习放第一位：当前数据是日线横截面，信噪比低、样本短，树模型和金融验证框架更重要。
3. 暂时不从交易玄学/畅销量化书入手：你现在缺的是研究框架和验证纪律，不是灵感故事。
4. 暂时不重复搜广推入门论文：本地 `dx-ML/书单.tex` 已经覆盖 YouTube DNN、Wide & Deep、DeepFM、DIN、BST、DLRM、LambdaRank 等；除非目标切回数据科学/推荐算法岗位，否则量化主线优先级更高。
