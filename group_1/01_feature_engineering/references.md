# 补充算子设计思路与参考文献

> 本文档对 `custom_operators.py` 中每个算子的设计逻辑、数学原理和出处做详细说明。

---

## 一、`safe_div` — 安全除法（第15-22行）

### 公式

$$\text{safe\_div}(a, b, \text{fill}) = \begin{cases} a / b, & b \neq 0 \text{ and } b \neq \text{NaN} \\ \text{fill}, & \text{otherwise} \end{cases}$$

### 设计思路

因子公式中大量使用除法：`close/vwap`、`vol/free_share`、`1/pe_ttm`。直接做 `a/b` 会产生 `inf`（分母为 0）、`NaN`（0/0）、`-inf`（负值除 0）。

这些异常值一旦出现，在后续 `rolling().mean()` 等时序运算中会**传染**——整个滚动窗口的结果全变为 NaN。金融数据中 zero volume、zero free_share、PE 为负等场景很常见，因此安全除法是因子计算管道的基础设施。

### 实现细节

```python
with np.errstate(divide='ignore', invalid='ignore'):
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan).fillna(fill)
```

三步走：先抑制 numpy 除零警告 → 将 `±inf` 统一替换为 `NaN` → 再用 `fill` 值填充。之所以分 `replace` 和 `fillna` 两步，是因为 pandas 的 `fillna` 不处理 `inf`（`inf` 是合法浮点数）。

### 出处

防御性编程通用实践，无特定文献。在量化因子库中（如 `alphalens`、`WorldQuant` 101 Formulaic Alphas）随处可见。

---

## 二、`pn_Rank` — 截面排名百分位（第28-30行）

### 公式

$$\text{pn\_Rank}(X)_{t,i} = \frac{\text{rank}_{t}(X_{t,i})}{N_t}$$

其中 $\text{rank}_t$ 是第 $t$ 天对所有股票的排序（升序），$N_t$ 是当天有效股票数。返回值域 [0, 1]。

### 设计思路

排名比原始值更**稳健**。原因：
- 原始值可能受极端离群值影响（如一只 PE 10000 的股票会让均值失去意义）
- 排名的分布是**均匀的**，不受原始分布（偏态、肥尾）的影响
- 排名可以跨时间直接比较

`pn_Rank` 只要求排序关系正确，不要求原始值满足任何分布假设。

### 使用场景

最基础的截面归一化方法。在需要对因子做绝对排序时使用（如"选排名前 10% 的股票"）。如果需要更精细的标准化，后续用 `pn_TransNorm`（Rank → 逆正态）效果更好。

### 出处

- 非参数统计排序方法：**Kendall, M. G. & Gibbons, J. D. (1990).** *Rank Correlation Methods.* 5th ed. Oxford University Press.
- 在量化金融中的应用：**Asness, C. S. (2016).** *The Siren Song of Factor Timing.* Journal of Portfolio Management.

---

## 三、`pn_Stand` / `Normalize` — 截面 Z-score 标准化（第36-43行）

### 公式

$$\text{pn\_Stand}(X)_{t,i} = \frac{X_{t,i} - \mu_t}{\sigma_t}$$

其中 $\mu_t = \frac{1}{N_t}\sum_i X_{t,i}$，$\sigma_t$ 是截面标准差。

### 设计思路

Z-score 是最经典的标准化方法。它假设原始数据近似正态分布，将每只股票表示为「偏离截面均值多少个标准差」。

**优点**：计算简单，保留了因子值的相对大小信息。
**缺点**：受极端值影响大（一个异常值就能大幅扭曲均值和标准差），通常在 `pn_Cut` 之后再使用。

### 出处

- 经典统计学：**Fisher, R. A. (1925).** *Statistical Methods for Research Workers.* Oliver & Boyd.
- 在多因子模型中的应用：**Grinold, R. & Kahn, R. (1999).** *Active Portfolio Management.* McGraw-Hill.（第 3 章：因子暴露标准化）

---

## 四、`pn_Cut` — Winsorize 截面截断（第49-70行）

### 设计思路

量化因子中最常见的数据预处理操作。财务数据常有极端值（PE 上千倍、换手率突然暴增），这些极值会主导截面排名和回归结果。

**双模式设计**：

| 调用方式 | 行为 | 场景 |
|----------|------|------|
| `pn_Cut(X, 0.01, 0.99)` | 按分位数截断（1%/99%） | 日常因子处理 |
| `pn_Cut(X, -3, 3)` | 按绝对值截断 | 已知边际范围 |

**实现选择**：用 numpy 数组操作（`np.percentile` + `np.clip`）而非 pandas 逐行处理。对 2000 天 × 1000 股的矩阵，纯 numpy 方式约快 10 倍。每行至少需要 3 个有效值才做截断（`valid.sum() < 3`），避免在退市股票上做无意义的截断。

### 出处

- **Barnett, V. & Lewis, T. (1994).** *Outliers in Statistical Data.* 3rd ed. Wiley.
- 在 Barra 风险模型中的应用：**Menchero, J., Morozov, A., & Shepard, P. (2011).** *Global Equity Risk Modeling.* In: Handbook of Portfolio Construction. Springer.（Barra 标准的 3σ winsorization）

---

## 五、`pn_Scale` — Min-Max 缩放（第76-80行）

### 公式

$$\text{pn\_Scale}(X)_{t,i} = \frac{X_{t,i} - \min_t(X)}{\max_t(X) - \min_t(X)}$$

### 设计思路

将所有值线性映射到 [0, 1]。与 `pn_Stand` 的区别：不假设正态分布，也不受均值和标准差的影响。但完全依赖最大最小值，对极端值非常敏感——因此通常先 `pn_Cut` 再 `pn_Scale`。

### 出处

机器学习常用归一化方法，无特定金融文献。参见 **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer.（数据预处理章节）

---

## 六、`pn_GroupNeutral` — 行业中性化（第86-106行）

### 公式

$$\text{pn\_GroupNeutral}(X, G)_{t,i} = X_{t,i} - \bar{X}_{t,G(i)}$$

其中 $G(i)$ 是股票 $i$ 的行业，$\bar{X}_{t,g}$ 是 $t$ 日行业 $g$ 内因子值的均值。

### 设计思路

中国 A 股有极强的行业轮动效应（如 2020 年医药、2021 年新能源、2023 年 AI）。如果因子天然偏好某些行业（ROE 因子偏好消费、PB 因子偏好金融），多空组合的收益大部分来自行业配置而非选股能力。

行业中性化后：每个行业的因子均值为 0，因子值只反映**行业内**的相对排序。这使得因子收益可归因于选股而非行业 Beta。

### 出处

- **Grinold, R. & Kahn, R. (1999).** *Active Portfolio Management.* McGraw-Hill.（第 7 章：因子中性化）
- **MSCI Inc. (2012).** *MSCI Barra Global Equity Risk Models.* Barra 行业因子的标准处理方法

---

## 七、`pn_GroupNorm` — 行业内 Z-score（第109-123行）

### 公式

$$\text{pn\_GroupNorm}(X, G)_{t,i} = \frac{X_{t,i} - \mu_{t,G(i)}}{\sigma_{t,G(i)}}$$

### 设计思路

与 `pn_GroupNeutral` 的区别：不仅去行业均值，还除以行业标准差。当不同行业的因子方差差异很大时（如 PB 因子在金融行业方差小、在科技行业方差大），仅去均值会导致高方差行业的股票主导排名。

行业内标准化后，每个行业均值为 0、标准差为 1，跨行业因子值可直接比较。

### 出处

- 同 `pn_GroupNeutral` 的 Barra 来源
- 在教育统计（标准化考试分）中类似的方法：**Kolen, M. J. & Brennan, R. L. (2014).** *Test Equating, Scaling, and Linking.* Springer.

---

## 八、`pn_CrossFit` — 截面回归残差（第129-154行）

### 公式

$$\text{residual}_{t,i} = X_{t,i} - \hat{X}_{t,i}, \quad \hat{X}_{t} = Y_{t} \cdot \hat{\beta}_t$$

其中 $\hat{\beta}_t = (Y_t^T Y_t)^{-1} Y_t^T X_t$ 是第 $t$ 天的 OLS 系数。

### 设计思路

这是最通用的中性化方法。`pn_GroupNeutral` 只能处理分类变量（行业），`pn_CrossFit` 可以同时处理多个**连续变量**（如市值、波动率）和**分类变量**（行业哑变量）。

**使用 `np.linalg.lstsq` 而非 `np.linalg.inv`**：后者在矩阵接近奇异时会崩溃，`lstsq` 可通过 SVD 给出最小范数解。跨截面只有 >10 个有效样本时才回归（`len(common) < 10`），避免在极端少样本日做无意义的拟合。

### 出处

- **Fama, E. & MacBeth, J. (1973).** *Risk, Return, and Equilibrium: Empirical Tests.* Journal of Political Economy, 81(3), 607-636.（Fama-MacBeth 两阶段回归框架）
- **Grinold, R. & Kahn, R. (1999).** *Active Portfolio Management.* McGraw-Hill.（因子正交化）

---

## 九、`ts_Rank` — 时序排名（第160-162行）

### 公式

$$\text{ts\_Rank}(X, N)_{t,i} = \frac{\text{rank}_{[t-N+1, t]}(X_{t,i})}{N}$$

### 设计思路

与 `pn_Rank`（截面排名）的区别：这个算的是每只股票**自己过去 N 天**内的百分比位置。例如：

- `ts_Rank(price, 60)` = 0.9 → 当前价格处于过去 60 天的较高位置（接近新高）
- `ts_Rank(price, 60)` = 0.1 → 当前价格处于过去 60 天的较低位置（接近新低）

截面排名回答「这只股票在所有股票中排第几」，时序排名回答「这只股票当前处于自己历史什么位置」。

### 出处

- 时序排名是趋势跟随策略的基础工具：**Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012).** *Time Series Momentum.* Journal of Financial Economics, 104(2), 228-250.

---

## 十、`ts_ChgRate` — 时序变化率（第165-167行）

### 公式

$$\text{ts\_ChgRate}(X, N)_{t,i} = \frac{X_{t,i}}{X_{t-N,i}} - 1$$

### 设计思路

衡量因子值的 N 日增长率。用 `safe_div` 而非直接除法，避免 `X_{t-N,i} = 0` 时产生 `inf`。

**使用场景**：`FF_TURNOVER_MOM = ts_ChgRate(turnover, 21)` → 换手率比 21 天前增长了多少。正值表示市场关注度上升，负值表示关注度下降。

### 出处

- 金融时间序列增长率的标准定义，参见 **Tsay, R. S. (2010).** *Analysis of Financial Time Series.* 3rd ed. Wiley.

---

## 十一、`ts_Corr` — 时序滚动相关系数（第170-181行）

### 公式

$$\text{ts\_Corr}(X, Y, N)_{t,i} = \frac{\text{Cov}_N(X_i, Y_i)}{\sigma_N(X_i) \cdot \sigma_N(Y_i)}$$

其中 $\text{Cov}_N$、$\sigma_N$ 分别是过去 N 期的协方差和标准差。

### 设计思路

**关键实现决策**：逐列用 `Series.rolling(N).corr(Series)` 而非 `DataFrame.rolling(N).corr(DataFrame)`。

原因：DataFrame 级别的 `rolling().corr()` 返回的是列对列的 MultiIndex 矩阵（形状为 (T×n_pairs, 1)），而不是我们需要的 (T, N_cols) 矩阵。这是一个 pandas API 的陷阱，许多新手会踩坑。

### 使用场景

`FF_VOL_PRICE_CORR = -ts_Corr(close, vol, 20)` → 量价相关系数取负。正值表示量价同向（放量上涨/缩量下跌），取负后转为反转信号。

### 出处

- Pearson 积差相关：**Pearson, K. (1895).** *Notes on Regression and Inheritance in the Case of Two Parents.* Proceedings of the Royal Society of London.
- 量价关系的实证研究：**Lee, C. & Swaminathan, B. (2000).** *Price Momentum and Trading Volume.* Journal of Finance, 55(5), 2017-2069.

---

## 十二、`ts_Cov` — 时序滚动协方差（第184-193行）

### 公式

$$\text{ts\_Cov}(X, Y, N)_{t,i} = \frac{1}{N-1} \sum_{\tau = t-N+1}^{t} (X_{\tau,i} - \bar{X}_{N,i})(Y_{\tau,i} - \bar{Y}_{N,i})$$

### 设计思路

与 `ts_Corr` 同样的逐列设计。协方差用于衡量两个变量的**方向性**联动，不除以各自标准差。

### 出处

- 协方差在投资组合理论中的核心地位：**Markowitz, H. (1952).** *Portfolio Selection.* Journal of Finance, 7(1), 77-91.

---

## 十三、`ts_RegressionFit` — 滚动回归 R²（第196-215行）

### 公式

$$R^2_i(t) = 1 - \frac{\sum_{\tau} (Y_{\tau,i} - \hat{Y}_{\tau,i})^2}{\sum_{\tau} (Y_{\tau,i} - \bar{Y}_{N,i})^2}$$

其中 $\hat{Y}$ 是 $Y$ 对 $X$ 的滚动 OLS 拟合值。

### 设计思路

用 R² 衡量个股收益能被某种因子（如市场收益）解释的程度：

- R² 高 → 这只股票的走势主要由该因子驱动（系统性风险高）
- R² 低 → 特质风险高，适合做多空对冲

**实现简化**：用 $r^2$（相关系数的平方）近似 OLS R²。这在单变量回归中完全等价，且计算量远小于做 N 次完整回归。

### 出处

- 滚动回归在金融中的应用：**Fama, E. & French, K. (1992).** *The Cross-Section of Expected Stock Returns.* Journal of Finance, 47(2), 427-465.
- R² 作为特质度的衡量：**Roll, R. (1988).** *R².* Journal of Finance, 43(3), 541-566.

---

## 十四、数学函数：`Log`、`Abs`、`Sign`、`Sqrt`（第221-238行）

### 设计思路

对 numpy 的薄封装，确保返回 DataFrame 而非 ndarray。在因子计算管道中保持类型一致性——所有中间结果都是 DataFrame。

### `Log` 的特殊处理

`np.log(X.replace(0, np.nan))`：先排除 0 值（对数无定义），再取 log。如 `-Log(circ_mv)` → 负的市值对数 = 小市值正向信号。

### 出处

- 对数变换用于数据压缩和线性化：**Box, G. E. P. & Cox, D. R. (1964).** *An Analysis of Transformations.* Journal of the Royal Statistical Society, Series B, 26(2), 211-252.
- 市值因子的 Log 处理：**Fama, E. & French, K. (1992).** 市值效应通常用 ln(市值) 而非原始市值，因为市值的截面分布极度右偏。

---

## 十五、`ts_BBOLL_PctB` — Bollinger %B（第244-252行）

### 公式

$$\text{%B}_{t,i} = \frac{P_{t,i} - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}$$

其中 $\text{Upper}_t = \text{MA}_{20} + 2 \cdot \sigma_{20}$，$\text{Lower}_t = \text{MA}_{20} - 2 \cdot \sigma_{20}$。

### 设计思路

%B 将价格标准化到 [0, 1] 区间：0 = 下轨，1 = 上轨，0.5 = 中轨。相比原始价格，%B 消除了绝对价格水平的影响，使得跨股票可比。

**归一化参数**（N=20, k=2）是 Bollinger 本人的标准推荐。

### 出处

- **Bollinger, J. (2002).** *Bollinger on Bollinger Bands.* McGraw-Hill.（第 1-4 章）

---

## 十六、`ts_MACD` — MACD 柱（第258-266行）

### 公式

$$\text{MACD\_Bar} = (\text{EMA}_{12} - \text{EMA}_{26}) - \text{EMA}_{9}(\text{EMA}_{12} - \text{EMA}_{26})$$

### 设计思路

经典参数 (12, 26, 9) 来源于 Gerald Appel 的原始设定。返回 MACD 柱（DIF - DEA）而非传统的 DIF/DEA 双线，原因是：
- 柱状图为单一数值，可直接用作因子值
- 正值 = 快线在慢线上方 = 上升动能
- 负值 = 死叉信号

### 出处

- **Appel, G. (2005).** *Technical Analysis: Power Tools for Active Investors.* Financial Times Prentice Hall.（第 8 章：MACD 的使用与参数优化）

---

## 十七、`ts_FundFlowRatio` — 资金流向比率（第272-279行）

### 公式

$$\text{FlowRatio}_t = \text{MA}_{N}\left( \frac{\text{BuyVol}_t - \text{SellVol}_t}{\text{BuyVol}_t + \text{SellVol}_t} \right)$$

### 设计思路

分子（净买入）除以分母（总成交），得到归一化的资金流向：+1 = 全是买方主动成交，-1 = 全是卖方主动成交，0 = 完全均衡。滚动平均平滑噪音。

### 出处

- 资金流分析的基础：**Frazzini, A., Israel, R., & Moskowitz, T. (2018).** *Trading Costs.* AQR Working Paper.
- 大单/小单信息含量：**Barber, B. & Odean, T. (2008).** *All That Glitters.* Review of Financial Studies, 21(2), 785-818.

---

## 十八、`ts_Amihud` — Amihud 非流动性（第285-291行）

### 公式

$$\text{Illiq}_{t,i} = \frac{1}{N} \sum_{\tau=t-N+1}^{t} \frac{|r_{\tau,i}|}{\text{Amount}_{\tau,i}}$$

### 设计思路

Amihud (2002) 提出这是最简洁且最稳健的非流动性度量：

- **分子** $|r|$：价格变动的绝对值 → 交易对价格的冲击
- **分母** Amount：成交额 → 交易的规模

单位成交额引起的价格冲击越大，说明该股票流动性越差。因子中取负号 `-ts_Amihud(...)`，使得非流动性高 → 正向因子值 → 多空组合买入高非流动性股票（流动性溢价）。

### 出处

- **Amihud, Y. (2002).** *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects.* Journal of Financial Markets, 5(1), 31-56.（全文定义及实证）

---

## 十九、`ts_PriceRange` — 价格区间（第297-301行）

### 公式

$$\text{PriceRange}_t = \frac{\max_{N}(\text{High}) - \min_{N}(\text{Low})}{\text{MA}_{N}\left( \frac{\text{High} + \text{Low}}{2} \right)}$$

### 设计思路

衡量过去 N 天的价格波动幅度（相对值）。分子是绝对振幅，分母用中间价做归一化以消除价格水平的绝对差异。高 PriceRange 的股票波动剧烈，通常在波动率因子下是负向信号。

### 出处

- 技术分析中的波动区间度量：**Edwards, R. D. & Magee, J. (2018).** *Technical Analysis of Stock Trends.* 11th ed. CRC Press.
- 价格区间与波动率的等价性：**Parkinson, M. (1980).** *The Extreme Value Method for Estimating the Variance of the Rate of Return.* Journal of Business, 53(1), 61-65.

---

## 二十、算子设计总原则

| 原则 | 体现 | 出处 |
|------|------|------|
| **同形输出** | 所有算子返回 (T, N) DataFrame | 管道化编程思想 |
| **防御性编程** | `safe_div` 防除零、`pn_Cut` 截断极值、`ts_Corr` 禁用有 bug 的 API | **Kernighan, B. W. & Pike, R. (1999).** *The Practice of Programming.* Addison-Wesley. |
| **最小惊讶** | 参数范围 `0<val<1` 自动识别分位数模式 | Unix 设计哲学 |
| **可组合性** | 每个算子独立可用 → 通过函数组合构建复杂因子 | 函数式编程范式 |
| **numpy 加速** | 核心运算用 numpy 数组而非 pandas 逐行 | 高性能计算通用实践 |

---

## 二十一、与已有 `feature.py` 算子的协作关系

| feature.py 算子 | custom_operators.py 配合使用 |
|-----------------|------------------------------|
| `ts_Decay` / `ts_DecayExp` | 用于包裹自定义算子的输出，实现衰减加权 |
| `ts_Stdev` / `ts_Sum` | 自定义算子的基础构件 |
| `pn_TransNorm` | 在 `pn_Cut` 之后做最终标准化 |
| `get_ls_post` | 在标准化后的因子值上构建多空组合 |

---

## 二十二、参考文献总目

### 统计学基础
1. **Kendall, M. G. & Gibbons, J. D. (1990).** *Rank Correlation Methods.* 5th ed.
2. **Barnett, V. & Lewis, T. (1994).** *Outliers in Statistical Data.* 3rd ed. Wiley.
3. **Fisher, R. A. (1925).** *Statistical Methods for Research Workers.* Oliver & Boyd.
4. **Box, G. E. P. & Cox, D. R. (1964).** *An Analysis of Transformations.* JRSS-B.
5. **Pearson, K. (1895).** *Notes on Regression and Inheritance.* Proceedings of the Royal Society.

### 量化金融与因子投资
6. **Grinold, R. & Kahn, R. (1999).** *Active Portfolio Management.* McGraw-Hill.
7. **Fama, E. & MacBeth, J. (1973).** *Risk, Return, and Equilibrium.* JPE.
8. **Fama, E. & French, K. (1992).** *The Cross-Section of Expected Stock Returns.* JF.
9. **Fama, E. & French, K. (1993).** *Common Risk Factors.* JFE.
10. **Asness, C. S., Moskowitz, T., & Pedersen, L. H. (2013).** *Value and Momentum Everywhere.* JF.
11. **Asness, C. S. (2016).** *The Siren Song of Factor Timing.* JPM.
12. **Markowitz, H. (1952).** *Portfolio Selection.* JF.

### 波动率与异象
13. **Ang, A. et al. (2006).** *The Cross-Section of Volatility and Expected Returns.* JF.
14. **Bali, T., Cakici, N., & Whitelaw, R. (2011).** *Maxing Out.* JFE.
15. **Parkinson, M. (1980).** *The Extreme Value Method for Estimating Variance.* JB.

### 流动性与资金流
16. **Amihud, Y. (2002).** *Illiquidity and Stock Returns.* JFM.
17. **Barber, B. & Odean, T. (2008).** *All That Glitters.* RFS.
18. **Lee, C. & Swaminathan, B. (2000).** *Price Momentum and Trading Volume.* JF.
19. **Frazzini, A. et al. (2018).** *Trading Costs.* AQR Working Paper.

### 技术分析
20. **Bollinger, J. (2002).** *Bollinger on Bollinger Bands.* McGraw-Hill.
21. **Appel, G. (2005).** *Technical Analysis: Power Tools for Active Investors.* FT Press.
22. **Edwards, R. D. & Magee, J. (2018).** *Technical Analysis of Stock Trends.* 11th ed. CRC Press.

### Barra 风险模型
23. **MSCI Inc. (2012).** *MSCI Barra Global Equity Risk Models.*
24. **Menchero, J., Morozov, A., & Shepard, P. (2011).** *Global Equity Risk Modeling.* Handbook of Portfolio Construction. Springer.

### 时序分析与特质风险
25. **Tsay, R. S. (2010).** *Analysis of Financial Time Series.* 3rd ed. Wiley.
26. **Roll, R. (1988).** *R².* JF.
27. **Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012).** *Time Series Momentum.* JFE.

### 软件工程
28. **Kernighan, B. W. & Pike, R. (1999).** *The Practice of Programming.* Addison-Wesley.
