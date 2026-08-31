# ML 学习路线：随机森林 / LightGBM / XGBoost

> 目标：JD 上那句"熟练掌握机器学习 / 随机森林 / LightGBM"，翻译成两句话——
> **不看资料能讲出原理**，**上手能改参数并看出效果**。
> 本文件不是代码手册，是学习目标和练习路线。代码实现以 `src/us_alpha_lab/modeling.py` 为准。

三个模型跑的是**同一套量化管线**（标签 → 防前视 → walk-forward 评估），所以先建立共同框架，再逐个拆模型。

## 共用管线（量化特有的，教程学不到）

```
原始因子 → future_return_5d 标签 → shift(feature_lag) 防前视 → walk-forward 分折 → MAE / R² / 方向准确率
```

对应代码：

- 标签：[labels.py](labels.py) `add_forward_return_label` → 用未来 5 日收益当预测目标
- 防前视：[modeling.py:108](modeling.py:108) 特征 `shift(feature_lag)` —— 只用昨天的因子预测明天
- 缺失值：[modeling.py:57](modeling.py:57) `SimpleImputer(strategy="median")` —— 为什么 median 不用 mean（对离群值稳）
- 评估：[modeling.py:191](modeling.py:191) walk-forward 带 `embargo`，不泄露相邻时点

**这四件事是"量化 ML"和"Kaggle demo"的分水岭。** 面试官问"你处理 lookahead bias 没"，答"代码里 shift 了一天"，比背定义强一百倍。

---

## 一、随机森林

### 数学公式

单棵决策树：按特征分裂，使子节点纯度最高。分类用 **Gini**，回归用 **MSE**：

$$\text{Gini} = 1 - \sum_k p_k^2$$

森林 = 很多棵树，每棵用**有放回抽样（bootstrap）+ 随机选特征子集**训练，预测取平均：

$$\hat{y} = \frac{1}{B} \sum_{b=1}^B f_b(x)$$

### 项目里的位置

[modeling.py:84](modeling.py:84)

```python
RandomForestRegressor(n_estimators=300, min_samples_leaf=20)
```

注意：RF 分支没有 `StandardScaler`——树模型对单调变换不敏感，不需要。

### 学习目标（面试能讲出的点）

- [ ] "随机"在哪：**样本随机（bootstrap）+ 特征随机（每次分裂随机选子集）**，这降低树之间的相关性，相关性越低、平均越稳
- [ ] Gini 怎么算、分裂怎么选特征（回归用 MSE）
- [ ] **OOB 误差**：每棵树没用到的样本（袋外）自然当验证集，所以不需要单独划验证集
- [ ] `min_samples_leaf` 是什么、调大为什么防过拟合
- [ ] 为什么"很多相关树平均"不如"不相关树平均"（误差 = 偏差² + 方差，随机性降的是方差）
- [ ] 特征重要性怎么算、怎么用它筛因子

### 动手实验

1. 给 `RandomForestRegressor` 加 `oob_score=True`，打印 OOB R² —— 练"为什么不用验证集"
2. 训完打印 `feature_importances_`，对到因子名上 —— 练"怎么用树筛因子"，直接是面试考点
3. 改 `min_samples_leaf` = 5 / 20 / 100，看 MAE 怎么变 —— 练"叶子越少越容易过拟合"

---

## 二、LightGBM

### 数学公式

Boosting：每一轮**拟合上一轮的残差**（负梯度），把"错误"逐步补上：

$$F_{m+1}(x) = F_m(x) + \eta \cdot h_m(x),\qquad h_m \approx -\nabla_F \mathcal{L}(F_m)$$

- $\eta$ = `learning_rate`：每步走多小
- 总步数 = `n_estimators`

LightGBM 与 RF 的区别：RF 是**并行**独立树平均；GBDT 是**串行**，后一棵树学前一棵的错。

LightGBM 特有：**leaf-wise 生长**（每次分裂全局最优叶）+ **直方图近似**（连续值分桶加速）+ **GOSS/EFB**（采样和特征合并加速）。

### 项目里的位置

[modeling.py:60](modeling.py:60)

```python
LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=31,
              max_depth=6, min_child_samples=100,
              subsample=0.8, bagging_freq=1, colsample_bytree=0.8,
              min_gain_to_split=0.01, reg_alpha=0.1, reg_lambda=5.0)
```

### 学习目标（面试能讲出的点）

- [ ] 每一轮迭代在拟合什么（负梯度 ≈ 残差），和 RF 的本质区别
- [ ] `learning_rate` 和 `n_estimators` **此消彼长**：学习率小 → 要更多轮 → 换 `early_stopping` 找最佳轮数
- [ ] 树多了为什么过拟合（模型复杂度 ↑ → 方差 ↑）
- [ ] **leaf-wise vs level-wise**：leaf-wise 拟合更快但更容易过拟合，所以靠 `num_leaves` + `min_child_samples` 约束
- [ ] `num_leaves=31` ≈ 满二叉树 depth 5，改大 = 模型变复杂
- [ ] `bagging_freq=1` 为什么要和 `subsample` 一起开，否则行采样不会按预期生效
- [ ] `rank_xendcg` 与 `lambdarank` 都是排序目标，适合直接优化每日横截面排序
- [ ] `top_bottom` 标签为什么丢掉中间分位：中间股票未来收益噪声最大，先让模型学头尾差异
- [ ] `reg_alpha` / `reg_lambda` 正则项在约束什么（L1/L2 罚权重）
- [ ] 为什么 LGBM 也**不需要** `StandardScaler`（对比 PyTorch/MLP 需要）

### 动手实验

1. 把 `learning_rate` 改 0.3（同时不动 `n_estimators`），看验证集先降后升 —— 练"此消彼长"
2. `num_leaves` = 8 vs 512，看谁过拟合 —— 练"leaf-wise 为什么容易过拟合"
3. 对比 `lightgbm_ranker + quantile` 和 `rank_xendcg + top_bottom`，看 OOS IC / ICIR —— 练"预测收益 vs 预测排序"
4. 把 `n_estimators` 减到 100、`learning_rate` 提到 0.3，对比 R² —— 练"调参 trade-off"

---

## 三、XGBoost（扩展）

### 数学公式

和 LightGBM 同为 GBDT 家族，区别在**目标函数加正则项**，每轮贪心找分裂：

$$\mathcal{L} = \sum_i \ell(y_i, \hat{y}_i) + \sum_k \Omega(f_k),\qquad \Omega(f) = \gamma T + \frac12 \lambda \|\mathbf{w}\|^2$$

- $T$ = 叶节点数（$\gamma$ 罚它）
- $\mathbf{w}$ = 叶权重（$\lambda$ 罚它）

XGBoost 与 LightGBM 关键差异：**level-wise**（按层生长）+ 二阶梯度 + 显式正则。LightGBM 是它的加速进化版。

### 项目里的位置

**目前还没用**。`modeling.py` 只支持 `random_forest` 和 `lightgbm`（[modeling.py:29](modeling.py:29) `normalize_model_name`）。XGBoost 是加分扩展。

### 学习目标（面试能讲出的点）

- [ ] 正则项 $\Omega(f)$ 在罚什么（叶子数和叶权重）—— XGBoost 名字里的"extreme"来自这里
- [ ] level-wise vs leaf-wise 的生长差异（XGBoost 是 level-wise）
- [ ] 和 LightGBM 对比：速度、精度、调参习惯的差别
- [ ] 二阶导（Newton）近似 vs LightGBM 一阶近似，为什么 XGBoost 收敛步数更多

### 动手实验

在 `normalize_model_name` 加一个 `"xgboost"` 分支 + `XGBRegressor`，跑同一 walk-forward，对比三模型 MAE / R²。

---

## 三模型对比总表（面试一页纸）

| 维度 | 随机森林 | LightGBM | XGBoost |
|---|---|---|---|
| 建模方式 | 并行独立树平均 | 串行拟合残差 | 串行拟合残差（二阶） |
| 生长方式 | level-wise | leaf-wise | level-wise |
| 正则 | 靠 `min_samples_leaf` 等 | `reg_alpha`/`reg_lambda` | 显式 $\Omega(f)$ |
| 对缺失值 | 扛得住 | 扛得住 | 显式支持 |
| 标准化 | 不需要 | 不需要 | 不需要 |
| 项目现状 | 已跑通 | 已跑通 | 未接入 |

共同结论：**横截面收益预测是中小规模表格数据，树模型是主场，不需要 MLP/transformer。** 这个结论本身就是面试题。

---

## 自测方法（达标标准）

不看资料，把每个模型"学习目标"里打 `[ ]` 的点口头讲一遍。能讲满 = 熟练；讲不出 = 该补。

优先级建议：

1. **RF 全打满**（加 `oob_score` + 特征重要性，约 20 分钟）
2. **LightGBM 全打满**（改 `learning_rate`/`num_leaves` 看效果）
3. **XGBoost 接入对比**（加分项）
4. **手写 PyTorch MLP 作为第三模型**（另起学习路线，练 `forward→loss→zero_grad→backward→step`）
