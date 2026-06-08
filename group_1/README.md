# Group Assignment — 中证1000多因子选股策略开发

> **课程**: 量化投资概论
> **项目**: 基于中证1000的多因子选股策略开发
> **股票池**: 中证1000成分股
> **回测区间**: 20xx-01-01 至 20xx-12-31（待定）

---

## 小组信息

| 姓名 | 学号 | 分工               |
| ---- | ---- | ------------------ |
| —   | —   | 因子开发与公式整理 |
| —   | —   | 算子实现与批量计算 |
| —   | —   | Barra暴露分析      |
| —   | —   | 组合构建与回测     |
| —   | —   | 报告撰写           |

---

## 项目结构

```
Group_Assignment/
├── README.md                          # 小组信息、分工、总结
├── 01_feature_engineering/            # 第一阶段：特征工程
│   ├── factor_formulas.md            # 45个候选因子公式清单
│   └── custom_operators.py           # 补充算子函数（19个算子）
├── 02_factor_calculation/            # 第二阶段：因子计算与筛选
│   ├── batch_compute.py              # 批量因子计算与筛选脚本
│   ├── factor_results.md             # 10个达标因子汇总表
│   └── corr_matrix.png               # 因子相关矩阵热力图
├── 03_portfolio_backtest/            # 第三+四阶段：组合回测
│   ├── combo_factor.py               # Barra暴露分析 + 组合因子构建
│   ├── backtest.py                   # 回测脚本
│   ├── equity_curve.png              # 净值曲线图
│   ├── barra_exposure_heatmap.png    # Barra风格暴露热力图
│   ├── neutralize_comparison.png     # 中性化前后对比
│   └── backtest_report.md            # 回测报告
└── data/
    └── group1/                        # 运行生成的中间数据
        ├── selected_factors.pkl      # 10个达标因子的标准化矩阵
        ├── factor_rets.pkl           # 因子收益序列
        ├── selected_names.pkl        # 因子名称列表
        ├── combo_factor.pkl          # 中性化后组合因子
        ├── combo_raw.pkl             # 中性化前组合因子
        └── combo_full.pkl            # 完整数据包
```

---

## 运行指南

### 环境要求

```bash
conda env create -f ../../environment.yml
conda activate py311
```

依赖包：`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `cvxpy`, `optuna`, `numba`

### 执行顺序

```bash
cd "你的路径/FinalProject_Clean"

# 1. 批量计算因子并筛选（第二阶段）
python group_1/02_factor_calculation/batch_compute.py

# 2. Barra暴露分析 + 组合因子构建（第三阶段 + 第四阶段准备）
python group_1/03_portfolio_backtest/combo_factor.py

# 3. 回测（第四阶段）
python group_1/03_portfolio_backtest/backtest.py
```

### 输出说明

| 输出文件                       | 说明                                                                |
| ------------------------------ | ------------------------------------------------------------------- |
| `corr_matrix.png`            | 全量候选因子 + 10个达标因子的相关系数热力图                         |
| `barra_exposure_heatmap.png` | 10个因子 × 10个Barra风格的暴露热力图 + 纯Alpha比例                 |
| `equity_curve.png`           | 6面板图：净值对比、超额收益、中性化对比、换手率、多空分解、分组收益 |
| `neutralize_comparison.png`  | 中性化前后风格暴露时间序列对比                                      |
| `backtest_report.md`         | 完整回测报告                                                        |

---

## 策略设计思路

### 因子来源

1. **人脑总结**（核心）：基于量化投资课程知识 + 学术文献经典因子的理解
2. **经典文献参考**：
   - Jegadeesh & Titman (1993) — 动量效应
   - Fama & French (1993) — 三因子模型
   - Ang et al. (2006) — 低波动异象
   - Amihud (2002) — 非流动性溢价
3. **实战经验**：知乎「实战因子365」、VWAP偏离、资金流向等

### 因子分类

| 类别     | 数量 | 核心逻辑               |
| -------- | ---- | ---------------------- |
| 动量类   | 7    | 趋势持续 + 量价配合    |
| 反转类   | 4    | 短期过度反应→均值回归 |
| 波动率类 | 7    | 低波异象 + 彩票型规避  |
| 流动性类 | 6    | 非流动性溢价           |
| 资金流类 | 4    | 主力资金净流入         |
| 技术类   | 8    | VWAP/Boll/MACD/RSI     |
| 基本面类 | 6    | 市值/PB/PE/股息        |
| 特质类   | 3    | 特质动量 + 涨跌幅      |

### 因子筛选标准

- **年化收益** |AR| > 0.1（多空组合年化>10%）
- **年化夏普** |SR| > 2（高收益-风险比）
- **因子间相关** |corr| < 0.3（低重叠度，保证维度覆盖）

### 组合构建

- **加权方式**: IC_IR 加权（利用历史IC信息比率分配权重）
- **风险控制**: Barra CNE5 10风格因子 + 30行业因子截面回归中性化
- **交易成本**: 单边0.1%（含印花税0.05% + 佣金0.03% + 滑点0.02%）

---

## 补充算子

项目中补充了 19 个通用算子函数（`custom_operators.py`），覆盖：

- 安全除法/比率算子
- 截面排名/标准化/截断/缩放/中性化
- 时序排名/变化率/相关/协方差/回归拟合
- Bollinger %B / MACD 技术指标
- Amihud 非流动性
- 资金流向比率

---

## 总结

本项目完成了一个完整的多因子选股策略开发闭环：

1. **因子开发**：基于学术文献和实战经验，设计了45个覆盖多维度、多逻辑的候选因子
2. **因子筛选**：通过严格的年化收益、夏普比率、相关性三重标准筛选出10个达标因子
3. **暴露分析**：通过Barra CNE5模型分析了每个因子的风格暴露和纯Alpha比例
4. **组合回测**：IC_IR加权合成 + 行业风格中性化，在回测区间实现了相对中证1000的稳健超额收益

项目的可扩展性强：可随时添加新的因子上到 `batch_compute.py`，自动参与筛选和组合优化。

---

*最后更新: 2026年6月 赵晨光*

