# 因子公式清单

> 因子来源：人脑总结/知乎研报/券商研报  
> 状态: ready=已实现  proxy=近似实现  no=暂未实现

## 一、动量类 (Momentum)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 1 | `FF_MOM_1M` | 经典动量 | `ts_DecayExp(dt['totalRet'], 21)` | 过去1月超额收益的指数衰减加权（半衰期~10天） | ready |
| 2 | `FF_MOM_3M` | 经典动量 | `ts_DecayExp(dt['totalRet'], 63)` | 过去3月超额收益的指数衰减加权 | ready |
| 3 | `FF_MOM_6M_SKIP` | Jegadeesh-Titman | `ts_DecayExp(ts_Delay(dt['totalRet'], 21), 126)` | 过去6月收益（跳过最近1月） | ready |
| 4 | `FF_RSTR_LIKE` | Barra RSTR | `ts_Sum(dt['totalRet'].shift(21), 252)` | 近似504日相对强度（skip 21） | ready |
| 5 | `FF_ABNORM_VOL` | 研报 | `safe_div(dt['vol'], ts_Mean(dt['vol'], 20)) - 1` | 放量异动：成交量/20日均量-1 | ready |
| 6 | `FF_VOL_PRICE_CORR` | 量价共振 | `-ts_Corr(dt['close'], dt['vol'], 20)` | 量价相关系数（取负：量价背离为正信号） | ready |
| 7 | `FF_TURNOVER_MOM` | 换手动量 | `ts_ChgRate(dt['turnover_rate'], 21)` | 换手率21日变化率 | ready |

## 二、反转类 (Reversal)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 8 | `FF_REV_5D` | 短期反转 | `-ts_Sum(dt['totalRet'], 5)` | 过去5日累计收益（取负：反转） | ready |
| 9 | `FF_REV_OVERNIGHT` | 隔夜反转 | `-dt['overnightRet']` | 隔夜收益反转 | ready |
| 10 | `FF_REV_GAP` | 跳空反转 | `-safe_div(dt['open'] - ts_Delay(dt['close'], 1), ts_Delay(dt['close'], 1))` | 开盘跳空幅度反转 | ready |
| 11 | `FF_LONG_REV` | 长期反转 | `-ts_Sum(dt['totalRet'], 252)` | 过去252日累计收益（长期反转） | ready |

## 三、波动率类 (Volatility)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 12 | `FF_VOL_20D` | 低波异象 | `-ts_Stdev(dt['totalRet'], 20)` | 过去20日波动率（取负：低波正收益） | ready |
| 13 | `FF_VOL_60D` | 低波异象 | `-ts_Stdev(dt['totalRet'], 60)` | 60日波动率 | ready |
| 14 | `FF_MAX_RET` | 最大日收益 | `-ts_Max(dt['totalRet'], 20)` | 过去20日最大日收益（彩票效应负相关） | ready |
| 15 | `FF_MIN_RET` | 最小日收益 | `ts_Min(dt['totalRet'], 20)` | 过去20日最小日收益 | ready |
| 16 | `FF_SKEW` | 偏度 | `-ts_Skewness(dt['totalRet'], 60)` | 过去60日收益偏度（正偏度→彩票型） | ready |
| 17 | `FF_DOWNSIDE_VOL` | 下行波动 | `-ts_Stdev(dt['totalRet'].clip(upper=0), 20)` | 下行波动率（只计负收益） | ready |
| 18 | `FF_VOL_OF_VOL` | 波动率波动 | `-ts_Stdev(ts_Stdev(dt['totalRet'], 5), 20)` | 波动率的波动率 | ready |

## 四、流动性类 (Liquidity)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 19 | `FF_TURNOVER` | 换手率 | `-dt['turnover_rate']` | 换手率（取负：低换手正收益） | ready |
| 20 | `FF_AMIHUD` | Amihud | `-ts_Amihud(dt['totalRet'], dt['amount'], 20)` | Amihud非流动性 | ready |
| 21 | `FF_VOLUME_RATIO` | 量比 | `safe_div(dt['vol'], ts_Mean(dt['vol'], 5))` | 当日量 / 5日均量 | ready |
| 22 | `FF_FREEFLOAT_TO` | 自由流通换手 | `-safe_div(dt['vol'], dt['free_share'], 0)` | 成交量/自由流通股（取负） | ready |
| 23 | `FF_AMOUNT_STD` | 成交额波动 | `ts_Stdev(dt['amount'], 20)` | 20日成交额标准差 | ready |
| 24 | `FF_LIQ_DECAY` | 流动性衰减 | `ts_Decay(-dt['turnover_rate'], 20)` | 换手率20日衰减（取负） | ready |

## 五、资金流类 (Money Flow)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 25 | `FF_NET_MF` | 主力净流入 | `safe_div(dt['net_mf_amount'], dt['amount'] * 1000)` | 主力净流入占比（amount原单位为千元） | ready |
| 26 | `FF_LARGE_BUY` | 大单买入 | `safe_div(dt['buy_lg_vol'] + dt['buy_elg_vol'], dt['vol'], 0)` | 大单+超大单买入占比 | ready |
| 27 | `FF_LARGE_SELL` | 大单卖出 | `-safe_div(dt['sell_lg_vol'] + dt['sell_elg_vol'], dt['vol'], 0)` | 大单卖出占比（取负） | ready |
| 28 | `FF_MF_DECAY` | 资金流衰减 | `ts_Decay(safe_div(dt['net_mf_amount'], dt['amount'] * 1000), 20)` | 主力资金流20日线性衰减 | ready |

## 六、技术类 (Technical)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 29 | `FF_VWAP_DEV` | VWAP偏离 | `-safe_div(dt['close'] - dt['vwap'], dt['vwap'])` | 收盘价偏离VWAP（超买反转） | ready |
| 30 | `FF_BOLL_PCTB` | Boll %B | `ts_BBOLL_PctB(dt['close'], 20, 2)` | Bollinger %B（超卖为正） | ready |
| 31 | `FF_MACD` | MACD柱 | `-ts_MACD(dt['close'], 12, 26, 9)` | MACD柱（取负：死叉→正信号） | ready |
| 32 | `FF_RSI` | RSI | `ts_Sum(dt['totalRet'].clip(lower=0), 14) / (ts_Sum(dt['totalRet'].clip(lower=0), 14) + ts_Sum((-dt['totalRet']).clip(lower=0), 14))` | 14日RSI，1-RSI 为超买反转信号 | ready |
| 33 | `FF_PRICE_POS` | 价格位置 | `safe_div(dt['close'] - ts_Min(dt['close'], 60), ts_Max(dt['close'], 60) - ts_Min(dt['close'], 60))` | 60日价格位置（0~1） | ready |
| 34 | `FF_HL_RATIO` | 振幅比 | `-safe_div(dt['high'] - dt['low'], dt['pre_close'])` | 日内振幅（高波→负相关） | ready |
| 35 | `FF_OPEN_RET` | 开盘效应 | `safe_div(dt['open'] - ts_Delay(dt['close'], 1), ts_Delay(dt['close'], 1))` | 开盘跳空率 | ready |
| 36 | `FF_CLOSE_POS` | 收盘位置 | `safe_div(dt['close'] - dt['low'], dt['high'] - dt['low'])` | K线实体位置（0~1） | ready |

## 七、基本面类 (Fundamental)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 37 | `FF_SIZE` | 市值因子 | `-Log(dt['circ_mv'])` | 流通市值对数（小市值效应） | ready |
| 38 | `FF_PB` | 市净率 | `-Log(dt['pb'])` | PB对数（价值效应） | ready |
| 39 | `FF_EP` | 盈利收益率 | `safe_div(1, dt['pe_ttm'], 0)` | 1/PE（PE_TTM的倒数） | ready |
| 40 | `FF_DIV` | 股息率 | `dt['dv_ttm']` | TTM股息率 | ready |
| 41 | `FF_PS` | 市销率 | `-Log(dt['ps_ttm'])` | PS对数（取负） | ready |
| 42 | `FF_TO_F` | 自由流通换手率 | `-dt['turnover_rate_f']` | 自由流通换手率（取负） | ready |

## 八、特质类 (Idiosyncratic)

| # | 因子名 | 来源 | 公式 | 说明 | 状态 |
|---|--------|------|------|------|------|
| 43 | `FF_IDIO_MOM` | 特质动量 | `-ts_Sum(dt['totalRet'] - dt['totalRet'].mean(axis=1, skipna=True).values.reshape(-1,1) if isinstance(dt['totalRet'], pd.DataFrame) else 0, 20)`  | 特质收益（去市场均值）20日累计 | ready |
| 44 | `FF_PCT_CHG` | 涨跌幅 | `-dt['pct_chg']` | 当日涨跌幅反转 | ready |
| 45 | `FF_EXRET_5D` | 超额收益 | `ts_Sum(dt['exRet'], 5)` | 5日超额收益累计 | ready |

## 九、补充算子说明

以下为需要补充的算子（定义在 `custom_operators.py`）：

| 算子名 | 功能 | 使用因子 |
|--------|------|----------|
| `safe_div` | 安全除法（避免除零） | 多个因子 |
| `pn_Rank` | 截面排名百分比 | 通用 |
| `pn_Stand / Normalize` | 截面Z-score | 通用 |
| `pn_Cut` | 截面截断 | 通用 |
| `pn_Scale` | 截面缩放 | 通用 |
| `pn_GroupNeutral` | 行业中性化 | Barra分析 |
| `pn_GroupNorm` | 行业内标准化 | Barra分析 |
| `pn_CrossFit` | 截面回归残差 | 中性化 |
| `ts_Rank` | 时序排名 | — |
| `ts_ChgRate` | 时序变化率 | FF_TURNOVER_MOM |
| `ts_Corr` | 时序相关系数 | FF_VOL_PRICE_CORR |
| `ts_Cov` | 时序协方差 | — |
| `ts_RegressionFit` | 时序回归R² | — |
| `Log / Abs / Sign / Sqrt` | 数学函数 | 多个因子 |
| `ts_BBOLL_PctB` | Bollinger %B | FF_BOLL_PCTB |
| `ts_MACD` | MACD | FF_MACD |
| `ts_FundFlowRatio` | 资金流向比率 | — |
| `ts_Amihud` | Amihud非流动性 | FF_AMIHUD |
| `ts_PriceRange` | 价格区间 | — |

## 十、中性化处理方案

所有因子在计算后均需做以下处理：
1. **成分股掩码**: `factor[univ_mask] = np.nan` （非中证1000成分股）
2. **上市天数过滤**: `factor[~listed] = np.nan` （上市不足20日）
3. **极值截断**: `pn_Cut(factor, 0.01, 0.99)` （1%/99%分位数）
4. **标准化**: `pn_TransNorm(factor)` （截面正态化→均值0、标准差1）
5. **行业中性（部分因子）**: `pn_GroupNeutral(factor, hy)`
