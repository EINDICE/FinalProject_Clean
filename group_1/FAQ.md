# Q&A：`data/` 目录下 `.pkl` 文件分类

## Q: `data/` 目录下的 `.pkl` 文件都有哪些不同分类？

A: `data/` 目录下的 `.pkl` 文件共 **106 个**，按子目录和用途分为四大类：

---

## 一、`matrix/` — 日频行情数据（约 75 个文件，每个约 30MB）

最大的一类，每一只股票的行情数据按 "字段 × 日期 × 股票" 的矩阵形式存储。

### 1.1 价格数据（9 个）

| 文件 | 含义 | 说明 |
|------|------|------|
| `open.pkl` | 开盘价 | 原始未复权 |
| `high.pkl` | 最高价 | |
| `low.pkl` | 最低价 | |
| `close.pkl` | 收盘价 | |
| `pre_close.pkl` | 前收盘价 | 用于计算涨跌幅/隔夜收益 |
| `adj_open.pkl` | 复权开盘价 | 后复权 |
| `adj_high.pkl` | 复权最高价 | |
| `adj_low.pkl` | 复权最低价 | |
| `adj_close.pkl` | 复权收盘价 | 计算收益的基准价 |

### 1.2 复权因子

| 文件 | 含义 |
|------|------|
| `adj_factor.pkl` | 复权因子，原始价 → 复权价：`复权价 = 原始价 × adj_factor` |

### 1.3 成交与流动性（6 个）

| 文件 | 含义 |
|------|------|
| `vol.pkl` | 成交量（股） |
| `amount.pkl` | 成交额（元） |
| `turnover_rate.pkl` | 换手率（流通股口径） |
| `turnover_rate_f.pkl` | 换手率（自由流通股口径） |
| `adj_vol.pkl` | 复权成交量 |
| `change.pkl` | 涨跌额 |

### 1.4 收益率（3 个）

| 文件 | 含义 |
|------|------|
| `totalRet.pkl` | **总收益**（含分红），回测中作为下一期收益 |
| `overnightRet.pkl` | 隔夜收益（`open / pre_close - 1`），衡量隔夜跳空 |
| `pct_chg.pkl` | 涨跌幅（%） |

### 1.5 估值与市值（8 个）

| 文件 | 含义 |
|------|------|
| `circ_mv.pkl` | 流通市值 |
| `total_mv.pkl` | 总市值 |
| `free_share.pkl` | 自由流通股本 |
| `float_share.pkl` | 流通股本 |
| `total_share.pkl` | 总股本 |
| `pb.pkl` | 市净率 PB |
| `pe_ttm.pkl` | 市盈率 PE(TTM) |
| `ps_ttm.pkl` | 市销率 PS(TTM) |
| `dv_ttm.pkl` | 股息率（TTM） |

### 1.6 行业

| 文件 | 含义 |
|------|------|
| `hy.pkl` | 行业分类（中信一级行业代码），**中性化必需** |

### 1.7 资金流数据（约 24 个）

按**订单大小**将每笔成交分为四类，分别记录买卖方向的成交量、成交额、VWAP：

| 前缀 | 含义 |
|------|------|
| `sm` | 小单（Small） |
| `md` | 中单（Medium） |
| `lg` | 大单（Large） |
| `elg` | 特大单（Extra Large） |

每条资金流记录 3 个维度的文件：

| 后缀 | 含义 |
|------|------|
| `_amount` | 成交额 |
| `_vol` | 成交量 |
| `_vwap` | 成交量加权均价 |

| 文件示例 | 含义 |
|----------|------|
| `buy_lg_amount.pkl` | 大单主动买入成交额 |
| `sell_elg_vol.pkl` | 特大单主动卖出成交量 |
| `net_mf_amount.pkl` | **净主动买入额**（`buy - sell` 汇总） |
| `net_mf_vol.pkl` | **净主动买入量** |

| `*_vwap_gap` 类 | 含义 |
|-----------------|------|
| `sm_vwap_gap.pkl` | 小单 VWAP 偏离度 |
| `md_vwap_gap.pkl` | 中单 VWAP 偏离度 |
| `lg_vwap_gap.pkl` | 大单 VWAP 偏离度 |
| `elg_vwap_gap.pkl` | 特大单 VWAP 偏离度 |

> `vwap_gap` 的经济含义：主动买入 VWAP 相对于全市场 VWAP 的偏离，正值表示买方愿意付出溢价，信号看涨。

---

## 二、`finMatrix/` — 年报/季报财务数据（约 43 个文件，约 30MB 每个）

按照证监会披露的三大报表提取，部分字段有 Q1 (1季度) / Y1 (1年) 和 QoQ (环比) / YoY (同比) 两个增量口径。

### 2.1 利润表（P&L）

| 文件 | 含义 |
|------|------|
| `revenue.pkl` | 营业收入 |
| `total_profit.pkl` | 利润总额 |
| `oper_cost.pkl` | 营业成本 |
| `fin_exp.pkl` | 财务费用 |
| `rd_exp.pkl` | 研发费用 |
| `basic_eps.pkl` | 基本每股收益 |
| `eps_annual.pkl` | 年度 EPS |
| `rev_per_share_annual.pkl` | 年度每股营收 |
| `ebit.pkl` | 息税前利润 |
| `ebitda.pkl` | 税息折旧摊销前利润 |

### 2.2 资产负债表

| 文件 | 含义 |
|------|------|
| `total_assets.pkl` | 总资产 |
| `total_liab.pkl` | 总负债 |
| `total_cur_liab.pkl` | 流动负债 |
| `NetAsset.pkl` | 净资产（所有者权益） |
| `fix_assets.pkl` | 固定资产 |
| `lt_borr.pkl` | 长期借款 |
| `st_borr.pkl` | 短期借款 |

### 2.3 现金流量表

| 文件 | 含义 |
|------|------|
| `n_cashflow_act.pkl` | 经营活动现金流净额 |
| `n_cashflow_inv_act.pkl` | 投资活动现金流净额 |

### 2.4 增长率和 TTM 口径

| 后缀 | 含义 |
|------|------|
| `Q1` | 最近一个季度 |
| `Y1` | 最近一年 |
| `QoQ` | 同比（Quarter over Quarter） |
| `YoY` | 环比（Year over Year） |
| `TTM` | 滚动12个月 |

| 示例 | 含义 |
|------|------|
| `RevenueIncYoY.pkl` | 营收同比增速 |
| `NetProfitTTMQ1.pkl` | 净利润 TTM（单季度滚动） |
| `TotalAssetY1.pkl` | 总资产（最近一年） |

### 2.5 其他

| 文件 | 含义 |
|------|------|
| `c_fr_sale_sg.pkl` | 销售现金流增速 |
| `rf_aligned.pkl` | 无风险利率（对齐到交易日） |

---

## 三、`finTTM/` — TTM 口径三大报表（3 个文件）

| 文件 | 含义 | 大小 |
|------|------|------|
| `balance_ttm.pkl` | 资产负债表（TTM） | 21.6 MB |
| `income_ttm.pkl` | 利润表（TTM） | 9.8 MB |
| `cash_ttm.pkl` | 现金流量表（TTM） | 7.9 MB |

> 与 `finMatrix` 的区别：`finMatrix` 是**每个财报截止日**的原始值（如 2023Q1、2023Q2…），`finTTM` 是按**每个交易日**填充的 TTM 值，更方便与日频行情矩阵对齐运算。

---

## 四、根目录文件

| 文件 | 含义 | 大小 |
|------|------|------|
| `idxWgt.pkl` | **中证1000成分股权重** | 33.9 MB |
| `idxWgt.csv` | 同上（CSV备份） | 82.4 MB |
| `config.json` | 数据路径配置 | 797 B |

> `idxWgt` 的形状是 (T, N)，T 是交易日、N 是股票代码列。值为该股票在指数中的权重，0 表示非成分股。代码中用 `idxWgt == 0` 作为 `univ_mask` 过滤非成分股。

---

## 五、关于 Barra 数据

`data/` 下**没有** Barra 风格暴露的 `.pkl` 文件。Barra 数据是通过 `barra/` 目录下的脚本**实时计算**出来的：

```text
barra/v1_barra_style.py   →  计算 10 个 Barra 风格因子
barra/v2_barra_stats.py   →  统计分析
barra/tool.py             →  Barra 工具函数
```

`combo_factor.py` 中加载 Barra 的代码（`path1000_barra`）指向的是运行时产出目录，而非 `data/` 原始数据。

---

## 六、总览

```text
data/
├── idxWgt.pkl              ← 中证1000成分股权重
├── config.json             ← 路径配置
├── matrix/   (75个文件)    ← 日频行情数据 ★因子计算主要数据源
│   ├── 价格: open/close/high/low + adj_*
│   ├── 成交: vol/amount/turnover_rate
│   ├── 收益: totalRet/overnightRet/pct_chg
│   ├── 估值: circ_mv/pb/pe_ttm/ps_ttm/dv_ttm
│   ├── 行业: hy
│   └── 资金流: buy_*/sell_*/net_mf_* (按大小单拆分)
├── finMatrix/ (43个文件)   ← 年报/季报财务数据
│   ├── 利润表: revenue/ebit/eps/rd_exp
│   ├── 资产负债表: total_assets/NetAsset/lt_borr
│   ├── 现金流量表: n_cashflow_act/inv_act
│   └── 增长率/Q1/Y1/QoQ/YoY 增量口径
└── finTTM/    (3个文件)    ← TTM口径三大报表
    ├── balance_ttm
    ├── income_ttm
    └── cash_ttm
```

总共 **106 个 `.pkl` 文件**，数据粒度覆盖 2016–2025 年约 10 年日频、中证1000 成分股范围（约 1000 只股票）。
