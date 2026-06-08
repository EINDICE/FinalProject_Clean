# 硕士生课程小组大作业：基于中证1000的多因子选股策略开发

---

## 一、作业概述

本作业要求各小组完成一个**完整的多因子选股策略闭环**：从因子开发，经过特征工程与算子计算、因子筛选、Barra风格暴露分析，最终构建组合因子并生成行业与风格中性的持仓回测曲线。
重点是因子开发的研究过程而不是单独衡量结果，因为因子可能来自多个方面，人脑学习总结，参考中英文研报，机器学习，大模型获取等多个方面，因此需要在因子开发过程中体现出对因子开发过程的理解和思考。
使用人脑学习总结的，报告需要覆盖思考过程；参考中英文研报需要提交参考文献，机器学习需要提交代码和运行结果；大模型获取需要提交相应skill。

**股票池**：中证1000成分股
**回测区间**：20170101 -- 20251231
**目标**：产出 10 个达标因子，构建组合因子，实现相对中证1000的超额收益

---

## 二、数据环境

### 2.1 已提供的数据矩阵

数据路径通过 `data\config.json` 中的 `S1000` 配置项获取。已加载的日频矩阵存入字典 `dt`，包含但不限于：

| 类别      | 字段                                                                                                                   | 说明                               |
| --------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 价格      | `open`, `high`, `low`, `close`, `pre_close`, `vwap`                                                        | 日频OHLCV+均价                     |
| 复权价格  | `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_factor`                                                 | 前复权价与因子                     |
| 量        | `vol`, `adj_vol`, `amount`                                                                                       | 成交量(手)、复权量、成交额(千元)   |
| 收益      | `totalRet`                                                                                                           | 日收益率（已屏蔽abs>0.2的异常值）  |
| 市值      | `circ_mv`, `total_mv`, `float_share`, `total_share`                                                            | 流通市值、总市值、流通股、总股本   |
| 行业      | `hy`, `BarraIndLabelD1`                                                                                            | 中信行业编码、Barra行业标签        |
| 资金流    | `buy_lg_vol`, `sell_lg_vol`, `buy_elg_vol`, `sell_elg_vol`, `buy_sm_vol`, `sell_sm_vol`, `net_mf_amount` | 大单/超大单/小单买卖量、主力净流入 |
| Barra风格 | `Beta`, `BTP`, `Size`, `Growth`, `Leverage`, `Liquidity`, `Momentum`, `ResVol`, `NLS`, `EY`        | 10个Barra风格因子暴露              |
| Barra收益 | `BarraStyleRet`, `Style3Ret`                                                                                       | 风格因子收益(折叠面板)             |
| 特质收益  | `idioRet`, `IdioRec`, `IdioRetD1`                                                                                | Barra回归残差                      |

### 2.2 中间数据构造

|  参考文件进行中间数据构造

### 2.3 指数成分股掩码

```python
idxWgt = pd.read_pickle(config['paths']['S1000'] + 'idxWgt.pkl')
univ_mask = idxWgt == 0  # True=非成分股，需屏蔽
listed = (dt['vol'].cumsum() > 0).shift(20)  # 上市满20个交易日
```

---

## 三、算子函数库

### 3.1 已提供的算子

算子定义在 `feature.py` 中常用算子如下，可以依靠需要自己构筑函数和算子：

**横截面算子**（每日对全体股票做截面运算）：

| 算子         | 函数                               | 说明           |
| ------------ | ---------------------------------- | -------------- |
| 截面排名     | `pn_Rank(X)`                     | 百分位排名     |
| 截面标准化   | `pn_Stand(X)` / `Normalize(X)` | Z-score标准化  |
| 秩→逆正态   | `pn_TransNorm(X)`                | 截面正态化变换 |
| 行业标准化   | `pn_GroupNorm(X, GrpLabel)`      | 行业内Z-score  |
| 行业中性     | `pn_GroupNeutral(X, GrpLabel)`   | 行业内去均值   |
| 截面回归残差 | `pn_CrossFit(X, GrpLabel)`       | 截面回归取残差 |
| 截面截断     | `pn_Cut(X, low, high)`           | 截断异常值     |
| 截面缩放     | `pn_Scale(X)`                    | 标准化         |

**时序算子**（对每只股票做时间序列运算）：

| 算子       | 函数                          | 说明             |
| ---------- | ----------------------------- | ---------------- |
| 时序均值   | `ts_Mean(X, N)`             | N日滚动均值      |
| 时序标准差 | `ts_Stdev(X, N)`            | N日滚动标准差    |
| 时序求和   | `ts_Sum(X, N)`              | N日滚动求和      |
| 时序排名   | `ts_Rank(X, N)`             | 过去N日百分位    |
| 时序延迟   | `ts_Delay(X, N)`            | 滞后N期          |
| 时序差分   | `ts_Delta(X, N)`            | X - shift(X, N)  |
| 时序变化率 | `ts_ChgRate(X, N)`          | X/shift(X,N) - 1 |
| 线性衰减   | `ts_Decay(X, N)`            | 线性加权移动平均 |
| 指数衰减   | `ts_DecayExp(X, N)`         | 指数半衰期加权   |
| 时序相关   | `ts_Corr(X, Y, N)`          | N日滚动相关系数  |
| 时序协方差 | `ts_Cov(X, Y, N)`           | N日滚动协方差    |
| 时序偏度   | `ts_Skewness(X, N)`         | N日滚动偏度      |
| 时序最大值 | `ts_Max(X, N)`              | N日滚动最大      |
| 时序最小值 | `ts_Min(X, N)`              | N日滚动最小      |
| 回归拟合度 | `ts_RegressionFit(Y, X, N)` | N日滚动R²       |

**通用函数**：

| 函数        | 说明     |
| ----------- | -------- |
| `Log(X)`  | 自然对数 |
| `Abs(X)`  | 绝对值   |
| `Sign(X)` | 符号函数 |
| `Sqrt(X)` | 平方根   |

### 3.2 中性化处理方法

量价指标常用的中性化方法：

| 中性思路     | 代表公式                                         | 消除什么      |
| ------------ | ------------------------------------------------ | ------------- |
| 价格比率     | `safe_div(Close, VWAP) - 1`                    | 价格绝对水平  |
| 价差/基准价  | `safe_div(High-Low, PreClose)`                 | 价格绝对水平  |
| 波动率标准化 | `safe_div(ts_Mean(Ret,20), ts_Stdev(Ret,20))`  | 波动率差异    |
| 成交量/股本  | `safe_div(AdjVolume, AShares)`                 | 流通盘大小    |
| 对数变换     | `Log(AdjVolume)`                               | 量级差异      |
| 回归残差     | `CS_REGRESSION(close, vol, out_type=0)`        | 风格暴露      |
| 1-R²特质度  | `Sqrt(1 - ts_RegressionFit(Ret, StyleRet, N))` | 可解释部分    |
| 复权因子     | `F_AdjPx(price, adj_factor)`                   | 除权除息      |
| 条件过滤     | `IfThen(0.04 - Abs(Ret), 0, -1)`               | 涨跌停/极端值 |
| 横截面排名   | `pn_Rank(X)`                                   | 分布偏斜      |

---

## 四、因子资料来源

各小组需从以下资料中选取并复现因子：

### 4.1 提供的因子资料库，或者自行寻找资料

### 4.2 其他机器学习或者深度学习方法生成因子

---

## 五、作业任务分解

### 第一阶段：特征工程与算子实现（权重 25%）

**要求**：

1. 阅读至少 5 篇知乎「实战因子365」或券商研报，理解因子逻辑
2. 基于提供的 `feature.py` 算子库，实现因子的中间数据处理
3. 若现有算子不足以复现研报因子，需自行补充算子函数并添加到 `feature.py`
4. 每个因子必须写成标准化的公式表达式，仅使用 `dt[...]` 和 `feature.py` 中的算子

**提交物**：

- 因子公式清单（markdown表格，列明：因子名、来源、公式、状态ready/proxy/no）
- 补充的算子函数代码（如有）

**评价标准**：

- 因子逻辑是否忠实于原文
- 中性化处理是否合理
- 补充算子的实现正确性

---

### 第二阶段：因子计算与筛选（权重 30%）

**要求**：

1. 编写批量因子计算脚本，计算至少 40 个候选因子
2. 对每个因子进行标准化处理（`pn_TransNorm` 或 `pn_Rank`）
3. 计算因子的多空收益指标，筛选出 **10个达标因子**：
   - 因子收益年化 > 10%（`|ar| > 0.1`）
   - 因子收益夏普 > 2（`|sr| > 2`）
   - 因子间相关系数 < 0.3（两两 `|corr| < 0.3`）
4. 若因子间相关性超标，需通过正交化或替换因子解决

**关键约定**：

```python
# 未来收益计算（避免前视偏差）
fut_ret = factor_port.shift(2) * dt['totalRet']

# 成分股掩码
factor[univ_mask] = np.nan
factor[~listed] = np.nan

# 多空权重
long_side, short_side = get_ls_post(factor)

# 指标计算
sr = long_ret.mean() / long_ret.std() * np.sqrt(252)  # 年化夏普
ar = (1 + long_ret.mean())**252 - 1                    # 年化收益
ic = factor.corrwith(dt['totalRet'].shift(-2), axis=1) # IC
```

**提交物**：

- 批量计算脚本
- 10个达标因子的公式、sr、ar、ic_mean、ic_ir 汇总表
- 因子间相关系数矩阵

**评价标准**：

- 是否有10个因子同时满足三项指标
- 相关性控制是否达标
- 代码运行是否无错

---

### 第三阶段：Barra 风格暴露分析（权重 20%）

**要求**：

1. 利用已有的 Barra CNE5 模型（10个风格因子 + 30个行业因子），分析每个达标因子的风格暴露
2. 对每个因子，计算其与 10 个 Barra 风格因子的截面相关系数
3. 识别因子的主要风格暴露（|corr| > 0.4 的风格因子），判断因子是否为纯 Alpha 还是被风格驱动
4. 分析因子的行业暴露：计算因子在各行业分组的均值差异

**Barra 风格因子列表**：

| 因子      | 含义       | 计算方式                                |
| --------- | ---------- | --------------------------------------- |
| Size      | 市值       | ln(总市值)                              |
| Beta      | 贝塔       | 252日WLS回归β                          |
| Momentum  | 动量       | 504日相对强度(滞后21天)                 |
| ResVol    | 残差波动   | 0.74×DASTD + 0.16×CMRA + 0.10×HSIGMA |
| NLS       | 非线性市值 | Size³ 对 Size 正交残差                 |
| BTP       | 账面市值比 | 净资产/总市值                           |
| Liquidity | 流动性     | 0.35×STOM + 0.35×STOQ + 0.30×STOA    |
| EY        | 盈利收益率 | 0.66×CETOP + 0.34×ETOP                |
| Growth    | 增长       | 季频增长率TransNorm等权                 |
| Leverage  | 杠杆       | 0.38×MLEV + 0.35×DTOA + 0.27×BLEV    |

**分析框架示例**：

```python
# 因子与Barra风格的截面相关
for style in STYLES:
    corr = factor_stack.corrwith(style_exposure_stack)
  
# 因子Barra回归：factor = α + Σ β_k × Style_k + Σ γ_j × Industry_j + ε
# 检验 α 是否显著（纯Alpha部分）
```

**提交物**：

- 10个因子 × 10个风格的暴露矩阵（热力图）
- 每个因子主要暴露风格的说明
- 因子纯Alpha比例分析（1 - R² of Barra回归）

**评价标准**：

- 暴露分析的完整性
- 对因子Alpha/风格属性的判断合理性
- 是否发现并处理了风格高度重叠的因子

---

### 第四阶段：组合因子构建与回测（权重 25%）

**要求**：

1. 将 10 个达标因子合成组合因子，合成方法可选：
   - **等权合成**：`combo = mean(factor_i)`
   - **IC加权**：`combo = Σ(ic_i / Σ|ic_i|) × factor_i`
   - **IC_IR加权**：`combo = Σ(ic_ir_i / Σ|ic_ir_i|) × factor_i`
   - **最优化加权**：最大化组合因子Sharpe的目标函数

   ```

   ```
2. 基于中性化后的组合因子构建持仓：

- 采用barra风格和行业因子风险暴露控制的办法

3. 计算持仓回测曲线，并与中证1000指数对比：
   - 组合净值曲线
   - 超额收益曲线
   - 年化超额收益、超额Sharpe、最大回撤

**关键实现**：

```python
# 行业+风格中性持仓构建
combo = IC_IR_weighted(factors)           # IC_IR加权
combo = barra_neutralize(combo)           # Barra中性化
long_w, short_w = get_ls_post(combo)      # 多空权重

# 回测
port_ret = (long_w * dt['totalRet']).sum(axis=1) - (short_w * dt['totalRet']).sum(axis=1)
cum_ret = (1 + port_ret).cumprod()

# 相对中证1000
bench_ret = dt['ARet']  # 或从指数行情获取
excess_ret = port_ret - bench_ret
excess_cum = (1 + excess_ret).cumprod()
```

**提交物**：

- 组合因子构建代码
- 持仓回测净值曲线图（绝对+超额）
- 回测指标汇总表
- 行业/风格中性化前后对比

**评价标准**：

- 组合因子是否实现超额收益
- 中性化是否有效（风格暴露是否降低）
- 回测逻辑是否避免前视偏差
- 超额收益的稳定性（月度胜率）

---

## 六、提交格式

需提交以下文件：

```
group_XX/
├── 01_feature_engineering/
│   ├── factor_formulas.md          # 因子公式清单
│   └── custom_operators.py         # 补充算子代码
├── 02_factor_calculation/
│   ├── batch_compute.py            # 批量计算脚本
│   ├── factor_results.md           # 10个达标因子汇总表
│   └── corr_matrix.png             # 因子相关矩阵热力图
├── 03_portfolio_backtest/
│   ├── combo_factor.py             # 组合因子构建代码
│   ├── backtest.py                 # 回测脚本
│   ├── equity_curve.png            # 净值曲线图
│   └── backtest_report.md          # 回测报告
└── README.md                       # 小组信息、分工、总结
```

---

## 七、评分标准

| 阶段      | 权重 | 优秀                                                 | 良好           | 及格             |
| --------- | ---- | ---------------------------------------------------- | -------------- | ---------------- |
| 特征工程  | 30%  | 10因子逻辑忠实、中性化充分、补2+算子                 | 10因子逻辑正确 | 10+因子可运行    |
| 因子筛选  | 40%  | 10因子全部达标(sr>2, ar>0.1, corr<0.3)               | 10因子部分达标 | 10因子仅一项达标 |
| Barra分析 | 20%  | 暴露分析完整、Alpha/风格归因清晰、有中性化前后对比   | 暴露分析完整   | 有基本暴露计算   |
| 组合回测  | 10%  | 超额Sharpe>2、中性化有效、月度胜率>60%、净值曲线平滑 | 有正超额收益   | 组合可运行       |

**加分项**：

- 因子创新性（非简单照搬研报公式）
- 补充的算子函数通用性强
- 使用最优化加权优于简单加权
- 组合因子超额收益回撤小
- 报告撰写规范、图表清晰

---

## 八、时间安排

| 阶段 | 周次 | 里程碑 |

1. 6月11日、12日进行开题报告
2. 6月25日提交总报告

---

## 九、参考资源

---

## 十、注意事项

1. **禁止前视偏差**：未来收益统一使用 `shift(2)`，风格暴露使用 `shift(1)`
2. **成分股掩码必须应用**：所有因子计算和回测都必须在 `idxWgt > 0` 且 `listed == True` 的范围内
3. **异常值处理**：`abs(totalRet) > 0.2` 的收益已屏蔽，因子本身也需做极值处理（`pn_Cut` 或 Winsorize）
4. **算子返回类型**：所有算子必须返回与输入同形状的 DataFrame，不得返回 Series 或标量
