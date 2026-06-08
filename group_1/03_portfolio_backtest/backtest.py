"""
backtest.py
第四阶段：组合因子回测

功能：
1. 加载中性化后的组合因子
2. 构建多空持仓 + 纯多头持仓
3. 计算回测收益曲线（含交易成本）
4. 与中证1000指数对比
5. 行业/风格中性化前后对比
6. 输出回测指标和净值曲线图
"""

import os, sys, logging
# __file__: group_1/03_portfolio_backtest/backtest.py → 需3级dirname到项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# 同时输出到日志文件
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)
_log_file = os.path.join(_LOG_DIR, 'backtest.log')
_fh = logging.FileHandler(_log_file, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(_fh)

from feature import *
from tools import read_json_config, GetQuantileRet

#%% ========== 配置与路径 ==========
config = read_json_config()
path1000 = config['paths']['S1000']
path1000_matrix = path1000 + 'matrix//'
path1000_barra = path1000 + 'barra//style//'
path1000_stats = path1000 + 'barra//stats//'

STYLES = ['Size', 'Beta', 'Momentum', 'ResVol', 'NLS',
          'BTP', 'Liquidity', 'EY', 'Growth', 'Leverage']

sd_per = '2017-01-01'
ed = '2025-12-31'
TRADING_COST = 0.001   # 单边交易成本: 0.1%

output_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'group1')

#%% ========== 加载数据 ==========
logger.info("=== 加载数据 ===")

# 组合因子
combo_final = pd.read_pickle(os.path.join(save_dir, 'combo_factor.pkl'))
combo_full = pd.read_pickle(os.path.join(save_dir, 'combo_full.pkl'))
combo_raw = combo_full['combo_raw']  # 中性化前
combo_neutral = combo_final  # 中性化后
weights_dict = combo_full['weights']
selected_names = combo_full['selected_names']

logger.info(f"  组合因子形状: {combo_neutral.shape}")
logger.info(f"  日期范围: {combo_neutral.index[0]} ~ {combo_neutral.index[-1]}")

# 收益数据
totalRet = pd.read_pickle(path1000_matrix + 'totalRet.pkl')
totalRet.index = pd.to_datetime(totalRet.index)
totalRet = totalRet[totalRet.index >= '2016-01-01']

# 指数权重
idxWgt = pd.read_pickle(path1000 + 'idxWgt.pkl')
idxWgt.index = pd.to_datetime(idxWgt.index)

# 行业数据
hy = pd.read_pickle(path1000_matrix + 'hy.pkl')
hy.index = pd.to_datetime(hy.index)

# 上市掩码
matrixList = os.listdir(path1000_matrix)
for v in matrixList:
    if v == 'vol.pkl':
        vol_tmp = pd.read_pickle(path1000_matrix + v)
        vol_tmp.index = pd.to_datetime(vol_tmp.index)
        break

listed = vol_tmp.copy()
listed[listed.isna()] = 0
listed = (listed.cumsum() > 0).shift(20)

# 成分股掩码
univ = idxWgt > 0

# Barra 风格暴露（用于中性化对比）
style_exposure = {}
for name in STYLES:
    path = path1000_barra + f'{name}.pkl'
    if os.path.exists(path):
        se = pd.read_pickle(path)
        se.index = pd.to_datetime(se.index)
        style_exposure[name] = se

logger.info("  所有数据加载完毕")

#%% ========== 辅助函数 ==========
def compute_backtest_metrics(port_ret, bench_ret=None, prefix=''):
    """计算回测指标"""
    n_days = len(port_ret)
    n_years = n_days / 252

    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 累计净值
    cum_ret = (1 + port_ret).cumprod()

    # 最大回撤
    peak = cum_ret.cummax()
    dd = cum_ret / peak - 1
    max_dd = dd.min()

    # 胜率
    win_rate = (port_ret > 0).mean()

    metrics = {
        '年化收益': ann_ret,
        '年化波动': ann_vol,
        '夏普比率': sharpe,
        '最大回撤': max_dd,
        '日胜率': win_rate,
        '累积收益': cum_ret.iloc[-1] - 1,
    }

    if bench_ret is not None:
        excess_ret = port_ret - bench_ret
        excess_ann = excess_ret.mean() * 252
        tracking_err = excess_ret.std() * np.sqrt(252)
        info_ratio = excess_ann / tracking_err if tracking_err > 0 else 0

        excess_cum = (1 + excess_ret).cumprod()
        excess_dd = (excess_cum / excess_cum.cummax() - 1).min()

        # 月度胜率
        excess_monthly = excess_ret.resample('M').sum()
        monthly_win = (excess_monthly > 0).mean()

        metrics.update({
            '年化超额收益': excess_ann,
            '跟踪误差': tracking_err,
            '信息比率': info_ratio,
            '超额最大回撤': excess_dd,
            '月度胜率': monthly_win,
            '超额累积': excess_cum.iloc[-1] - 1,
        })

    return metrics


def build_position(factor, method='ls'):
    """
    构建持仓权重
    method='ls': 多空组合（get_ls_post）
    method='top100': 前100只等权多头
    """
    if method == 'ls':
        port_pos, port_neg = get_ls_post(factor.copy())
        return port_pos + port_neg
    elif method == 'top100':
        _, hold = get_portFromFactor_both(factor.copy(), 100)
        return hold
    return None


#%% ========== 回测主流程 ==========
logger.info("\n" + "=" * 70)
logger.info("=== 回测主流程 ===")
logger.info("=" * 70)

# 使用中性化后的组合因子
combo_use = combo_neutral.copy()

# 1. 中性化前后对比
logger.info("\n--- 中性化前后对比 ---")

# 中性化前的持仓
combo_raw_aligned = combo_raw.reindex(index=combo_neutral.index, columns=combo_neutral.columns)
port_raw = build_position(combo_raw_aligned, method='ls')

# 中性化后的持仓
port_neutral = build_position(combo_use, method='ls')

# 对齐收益数据
ret_aligned = totalRet.reindex(index=combo_neutral.index, columns=combo_neutral.columns)

# 计算收益（延迟2期避免前视偏差）
port_ret_raw = (port_raw.shift(2) * ret_aligned).sum(axis=1)
port_ret_neutral = (port_neutral.shift(2) * ret_aligned).sum(axis=1)

# 基准收益：中证1000等权
bench_wgt = idxWgt.reindex(index=combo_neutral.index, columns=combo_neutral.columns).fillna(0)
bench_wgt = bench_wgt.div(bench_wgt.sum(axis=1), axis=0).fillna(0)
bench_ret = (bench_wgt.shift(1) * ret_aligned).sum(axis=1)

# 日期过滤
port_ret_raw = port_ret_raw[port_ret_raw.index >= sd_per]
port_ret_neutral = port_ret_neutral[port_ret_neutral.index >= sd_per]
bench_ret = bench_ret[bench_ret.index >= sd_per]

#%% ========== 换手率与交易成本 ==========
logger.info("\n--- 换手率与交易成本 ---")

# 计算换手率（中性化后）
turnover_daily = (port_neutral - port_neutral.shift(1)).abs().sum(axis=1)
turnover_daily = turnover_daily[turnover_daily.index >= sd_per]
avg_turnover = turnover_daily.mean()

# 交易成本
cost_pct = turnover_daily * TRADING_COST
port_ret_neutral_net = port_ret_neutral - cost_pct

logger.info(f"  日均换手率: {avg_turnover:.4f}  (年化: {avg_turnover * 252:.2f})")
logger.info(f"  日均交易成本: {cost_pct.mean():.6f}  ({cost_pct.mean() * 252 * 100:.2f}% 年化)")

#%% ========== 回测指标 ==========
logger.info("\n--- 回测指标 ---")

metrics_raw = compute_backtest_metrics(port_ret_raw, bench_ret, '中性化前')
metrics_neutral = compute_backtest_metrics(port_ret_neutral, bench_ret, '中性化后')
metrics_net = compute_backtest_metrics(port_ret_neutral_net, bench_ret, '含交易成本')

logger.info("\n中性化前:")
for k, v in metrics_raw.items():
    if '累积' not in k:
        logger.info(f"  {k}: {v:+.4f}")

logger.info("\n中性化后（不含成本）:")
for k, v in metrics_neutral.items():
    if '累积' not in k:
        logger.info(f"  {k}: {v:+.4f}")

logger.info("\n中性化后（含成本）:")
for k, v in metrics_net.items():
    if '累积' not in k:
        logger.info(f"  {k}: {v:+.4f}")

#%% ========== 多空/多头/空头分解 ==========
logger.info("\n--- 多空分解 ---")
port_pos, port_neg = get_ls_post(combo_use.copy())
port_ls = port_pos + port_neg

ret_ave = totalRet.reindex(index=combo_use.index, columns=combo_use.columns)
ret_ave_mean = ret_ave.mean(axis=1)

rets_pos = (port_pos.shift(2) * ret_aligned).sum(axis=1) - ret_ave_mean
rets_neg = (port_neg.shift(2) * ret_aligned).sum(axis=1) + ret_ave_mean

rets_pos = rets_pos[rets_pos.index >= sd_per]
rets_neg = rets_neg[rets_neg.index >= sd_per]

logger.info(f"  多头年化收益: {rets_pos.mean() * 252:+.4f}  夏普: {rets_pos.mean() / rets_pos.std() * np.sqrt(252):+.3f}")
logger.info(f"  空头年化收益: {rets_neg.mean() * 252:+.4f}  夏普: {rets_neg.mean() / rets_neg.std() * np.sqrt(252):+.3f}")

#%% ========== 分组收益 ==========
logger.info("\n--- 5分组收益 ---")
combo_q = combo_use.copy()
combo_q[listed[combo_q.index[0]:].reindex(index=combo_q.index, columns=combo_q.columns) == 0] = np.nan
qr = GetQuantileRet(combo_q, ret_aligned, 5, 2)
qr = qr[qr.index >= sd_per]
qr_cum = qr.cumsum()
qr_annual = qr.mean() * 252
logger.info(f"  各组年化收益: {qr_annual.values}")

#%% ========== 行业/风格中性化前后对比 ==========
logger.info("\n--- 行业/风格中性化前后对比 ---")

def calc_style_exposure(pos, style_exposure, styles):
    """计算组合的风格暴露"""
    result = {}
    for name in styles:
        if name in style_exposure:
            se = style_exposure[name].reindex(index=pos.index, columns=pos.columns)
            exp_ts = (pos * se).sum(axis=1)
            result[name] = exp_ts
    return pd.DataFrame(result)

# 中性化前风格暴露
style_exp_raw = calc_style_exposure(port_raw, style_exposure, STYLES)
style_exp_neutral = calc_style_exposure(port_neutral, style_exposure, STYLES)

# 中性化前后风格暴露对比
fig_style, axes_style = plt.subplots(2, 1, figsize=(16, 10))

for i, (exp_df, title) in enumerate([
    (style_exp_raw, 'Before Neutralization'),
    (style_exp_neutral, 'After Barra Neutralization')
]):
    ax = axes_style[i]
    for name in STYLES:
        if name in exp_df.columns:
            ax.plot(exp_df.index, exp_df[name].values, label=name, alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Portfolio Style Exposure ({title})', fontsize=12, fontweight='bold')
    ax.legend(ncol=5, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
neutralize_cmp_path = os.path.join(output_dir, 'neutralize_comparison.png')
plt.savefig(neutralize_cmp_path, dpi=150, bbox_inches='tight')
plt.close()

# 统计中性化效果
logger.info("\n  中性化前后风格暴露均值（绝对值）:")
for name in STYLES:
    if name in style_exp_raw.columns and name in style_exp_neutral.columns:
        raw_mean = style_exp_raw[name].abs().mean()
        neu_mean = style_exp_neutral[name].abs().mean()
        logger.info(f"    {name:12s}: 前={raw_mean:.4f}  后={neu_mean:.4f}  降低={raw_mean - neu_mean:+.4f}")

#%% ========== 绘制净值曲线 ==========
logger.info("\n--- 绘制净值曲线 ---")
fig, axes = plt.subplots(3, 2, figsize=(20, 16))

# 1. 策略累计收益 vs 基准（左上）
ax1 = axes[0, 0]
port_cum = (1 + port_ret_neutral_net).cumprod()
bench_cum = (1 + bench_ret[bench_ret.index.intersection(port_ret_neutral_net.index)]).cumprod()
idx = port_cum.index
ax1.plot(idx, port_cum.values, label=f'Portfolio (Net)', linewidth=1.5, color='steelblue')
ax1.plot(idx, bench_cum.loc[idx].values, label='CSI 1000 (Equal-weight)', linewidth=1.5, color='darkorange')
ax1.set_title('Strategy Net Value vs Benchmark', fontsize=13, fontweight='bold')
ax1.set_ylabel('Net Value (Initial=1)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. 超额收益累计（右上）
ax2 = axes[0, 1]
excess_ret = port_ret_neutral_net - bench_ret[bench_ret.index.intersection(port_ret_neutral_net.index)]
excess_cum = (1 + excess_ret).cumprod()
ax2.plot(excess_cum.index, excess_cum.values, label='Cumulative Excess Return',
         linewidth=1.5, color='green')
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(excess_cum.index, 1, excess_cum.values.flatten(),
                 where=(excess_cum.values.flatten() >= 1), alpha=0.3, color='green')
ax2.fill_between(excess_cum.index, 1, excess_cum.values.flatten(),
                 where=(excess_cum.values.flatten() < 1), alpha=0.3, color='red')
ax2.set_title('Cumulative Excess Return (Net of Cost)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 3. 中性化前后对比（左中）
ax3 = axes[1, 0]
port_cum_raw = (1 + port_ret_raw).cumprod()
idx_raw = port_cum_raw.index
ax3.plot(idx_raw, port_cum_raw.values, label='Before Neutralization', linewidth=1.2, alpha=0.7)
ax3.plot(port_cum.index, port_cum.values, label='After Neutralization (Net)', linewidth=1.5, color='steelblue')
ax3.plot(bench_cum.index, bench_cum.values, label='Benchmark', linewidth=1.2, alpha=0.5)
ax3.set_title('Neutralization: Before vs After', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. 换手率（右中）
ax4 = axes[1, 1]
ax4.fill_between(turnover_daily.index, 0, turnover_daily.values, alpha=0.5, color='steelblue')
ax4.plot(turnover_daily.index, turnover_daily.rolling(20).mean().values,
         color='darkred', linewidth=1.5, label='20-day MA')
ax4.axhline(y=turnover_daily.mean(), color='orange', linestyle='--',
            label=f'Mean={turnover_daily.mean():.4f}')
ax4.set_title('Daily Turnover Rate', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 多空分解（左下）
ax5 = axes[2, 0]
ls_common = port_ret_neutral.index.intersection(rets_pos.index).intersection(rets_neg.index)
ax5.plot(ls_common, rets_pos.loc[ls_common].cumsum().values, label='Long Side', linewidth=1.2, color='green')
ax5.plot(ls_common, rets_neg.loc[ls_common].cumsum().values, label='Short Side', linewidth=1.2, color='red')
ax5.plot(ls_common, port_ret_neutral.loc[ls_common].cumsum().values,
         label='Long-Short', linewidth=1.5, color='steelblue')
ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax5.set_title('Long-Short Decomposition', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 5分组收益（右下）
ax6 = axes[2, 1]
qr_cum_aligned = qr_cum.reindex(qr.index)
for col in qr_cum_aligned.columns:
    ax6.plot(qr_cum_aligned.index, qr_cum_aligned[col].values,
             label=f'Q{col+1} (AR={qr_annual[col]:+.2%})', linewidth=1.2)
ax6.set_title('Quintile Portfolio Cumulative Returns', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
equity_path = os.path.join(output_dir, 'equity_curve.png')
plt.savefig(equity_path, dpi=150, bbox_inches='tight')
plt.close()
logger.info(f"  净值曲线已保存: {equity_path}")

#%% ========== 月度收益分析 ==========
logger.info("\n--- 月度收益分析 ---")
excess_monthly = excess_ret.resample('M').sum()
monthly_win_rate = (excess_monthly > 0).mean()
years = excess_monthly.index.year.unique()
yearly_excess = excess_monthly.groupby(excess_monthly.index.year).sum()

logger.info(f"  月度胜率: {monthly_win_rate:.2%}")
logger.info("\n  年度超额收益:")
for yr in years:
    if yr in yearly_excess.index:
        logger.info(f"    {yr}: {yearly_excess[yr]:+.4f}")

#%% ========== 总结输出 ==========
logger.info("\n" + "=" * 70)
logger.info("=== 回测总结 ===")
logger.info("=" * 70)
logger.info(f"\n  回测区间: {sd_per} ~ {ed}")
logger.info(f"  因子数量: {len(selected_names)}")
logger.info(f"  加权方式: IC_IR 加权")
logger.info(f"  中性化: Barra 10风格 + 行业中性")
logger.info(f"\n  策略年化收益:     {metrics_net['年化收益']:.4f}")
logger.info(f"  策略年化波动:     {metrics_net['年化波动']:.4f}")
logger.info(f"  策略夏普比率:     {metrics_net['夏普比率']:.3f}")
logger.info(f"  策略最大回撤:     {metrics_net['最大回撤']:.3%}")
logger.info(f"\n  年化超额收益:     {metrics_net['年化超额收益']:.4f}")
logger.info(f"  跟踪误差:         {metrics_net['跟踪误差']:.4f}")
logger.info(f"  信息比率:         {metrics_net['信息比率']:.3f}")
logger.info(f"  超额最大回撤:     {metrics_net['超额最大回撤']:.3%}")
logger.info(f"  月度胜率:         {metrics_net['月度胜率']:.2%}")
logger.info(f"\n  日均换手率:       {avg_turnover:.4f}")
logger.info(f"  年化换手率:       {avg_turnover * 252:.2f}")

logger.info("\n完成！")
