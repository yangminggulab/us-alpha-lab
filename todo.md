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

### B. LightGBM（代码在 [modeling.py:48-76](src/us_alpha_lab/modeling.py:48)）

- [ ] `learning_rate` 0.03 → 0.3，观察验证集先降后升，讲"和 `n_estimators` 此消彼长"
- [ ] `num_leaves` 8 vs 512，观察过拟合，讲"leaf-wise 为什么容易过拟合"
- [ ] 不看资料讲出：每轮拟合残差（负梯度）、level-wise vs leaf-wise、`reg_alpha`/`reg_lambda` 在罚什么

### C. XGBoost（扩展，目前项目未接入）

- [ ] 在 [modeling.py:29](src/us_alpha_lab/modeling.py:29) `normalize_model_name` 加 `xgboost` 分支，接入同一 walk-forward
- [ ] 跑 RF / LGBM / XGB 三模型对比，记录 MAE / R² / 方向准确率表
- [ ] 不看资料讲出 XGBoost 与 LGBM 差异：level-wise、二阶梯度、显式正则 $\Omega(f) = \gamma T + \tfrac12\lambda\|\mathbf{w}\|^2$

### D. 量化落地（教程没有、面试必问）

- [ ] 讲清防前视：[modeling.py:108](src/us_alpha_lab/modeling.py:108) `shift(feature_lag)` 和 walk-forward `embargo`
- [ ] 讲清为什么树模型不需要 `StandardScaler`（对比 MLP 需要）
- [ ] 讲清缺失值为什么用 median 不用 mean（[modeling.py:57](src/us_alpha_lab/modeling.py:57)）

## 二、PyTorch：从 0 到熟练掌握

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

- 机器学习：RF 未开始 / LightGBM 未开始 / XGBoost 未接入
- PyTorch：未开始
