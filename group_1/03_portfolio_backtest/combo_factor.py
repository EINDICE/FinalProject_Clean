"""
combo_factor.py
第三阶段 + 第四阶段准备：Barra 风格暴露分析 + 组合因子构建

功能：
1. 加载 10 个达标因子的标准化数据
2. 分析每个因子与 10 个 Barra 风格因子的截面相关系数（热力图）
3. 计算因子纯 Alpha 比例（1 - R² of Barra 回归）
4. 合成组合因子（IC_IR 加权 / 等权 / IC 加权）
5. 行业+风格中性化
6. 保存组合因子数据供回测使用
"""

import os, sys, logging
# __file__: group_1/03_portfolio_backtest/combo_factor.py → 需3级dirname到项目根目录
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
_log_file = os.path.join(_LOG_DIR, 'combo_factor.log')
_fh = logging.FileHandler(_log_file, encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(_fh)
from sklearn.linear_model import LinearRegression

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

sd = '2016-01-01'
sd_per = '2017-01-01'
ed = '2025-12-31'

output_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'group1')

#%% ========== 加载数据 ==========
logger.info("=== 加载因子数据 ===")
selected_factors = pd.read_pickle(os.path.join(save_dir, 'selected_factors.pkl'))
factor_rets = pd.read_pickle(os.path.join(save_dir, 'factor_rets.pkl'))
selected_names = list(selected_factors.keys())
logger.info(f"  达标因子: {selected_names}")

logger.info("\n=== 加载Barra风格暴露数据 ===")
style_exposure = {}
for name in STYLES:
    path = path1000_barra + f'{name}.pkl'
    if os.path.exists(path):
        se = pd.read_pickle(path)[sd:ed]
        se.index = pd.to_datetime(se.index)
        style_exposure[name] = se
        logger.info(f"  {name}: {se.shape}")

# 加载行业数据
hy = pd.read_pickle(path1000_matrix + 'hy.pkl')[sd:ed]
hy.index = pd.to_datetime(hy.index)

# 加载总收益
totalRet = pd.read_pickle(path1000_matrix + 'totalRet.pkl')[sd:ed]
totalRet.index = pd.to_datetime(totalRet.index)

# 加载指数权重
idxWgt = pd.read_pickle(path1000 + 'idxWgt.pkl')
idxWgt.index = pd.to_datetime(idxWgt.index)

# 上市掩码
matrixList = os.listdir(path1000_matrix)
dt_tmp = {}
for v in matrixList:
    if v == 'vol.pkl':
        dt_tmp['vol'] = pd.read_pickle(path1000_matrix + v)
        dt_tmp['vol'] = dt_tmp['vol'][dt_tmp['vol'].index >= sd]
        dt_tmp['vol'].index = pd.to_datetime(dt_tmp['vol'].index)
        break

listed = dt_tmp['vol'].copy()
listed[listed.isna()] = 0
listed = (listed.cumsum() > 0).shift(20)

#%% ========== Barra 风格暴露分析 ==========
logger.info("\n" + "=" * 70)
logger.info("=== Barra 风格暴露分析 ===")
logger.info("=" * 70)

# 对齐日期
all_dates = selected_factors[selected_names[0]].index

# 1. 计算每个因子与各Barra风格的截面相关系数
logger.info("\n--- 因子 × 风格 截面相关系数矩阵 ---")
exposure_corr = pd.DataFrame(index=selected_names, columns=STYLES)

for fname in selected_names:
    f = selected_factors[fname].copy()
    for sname in STYLES:
        if sname in style_exposure:
            se = style_exposure[sname].copy()
            # 对齐
            common_dates = f.index.intersection(se.index)
            common_cols = f.columns.intersection(se.columns)
            if len(common_dates) > 0 and len(common_cols) > 100:
                f_aligned = f.loc[common_dates, common_cols]
                se_aligned = se.loc[common_dates, common_cols]
                # 按日计算截面相关，取均值
                daily_corr = []
                for d in common_dates:
                    f_row = f_aligned.loc[d].dropna()
                    se_row = se_aligned.loc[d].dropna()
                    common = f_row.index.intersection(se_row.index)
                    if len(common) > 50:
                        daily_corr.append(f_row[common].corr(se_row[common]))
                exposure_corr.loc[fname, sname] = np.mean(daily_corr) if daily_corr else np.nan

exposure_corr = exposure_corr.astype(float)
logger.info('\n' + str(exposure_corr.round(3)))

# 2. 热力图
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# 左边: 因子 × 风格暴露相关矩阵
ax1 = axes[0]
im1 = ax1.imshow(exposure_corr.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6,
                 aspect='equal', interpolation='nearest')
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('Correlation', fontsize=10)
ax1.set_xticks(range(len(STYLES)))
ax1.set_yticks(range(len(selected_names)))
ax1.set_xticklabels(STYLES, rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels(selected_names, fontsize=9)
# 标注数值
for i in range(len(selected_names)):
    for j in range(len(STYLES)):
        val = exposure_corr.iloc[i, j]
        if not np.isnan(val):
            text_color = 'white' if abs(val) > 0.3 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=9, color=text_color)
ax1.set_title('Factor × Barra Style Exposure Correlation', fontsize=14, fontweight='bold')
ax1.set_ylabel('Alpha Factors')
ax1.set_xlabel('Barra Style Factors')

# 3. 纯Alpha比例分析 (1 - R² of Barra回归)
logger.info("\n--- 因子纯Alpha比例分析 ---")
alpha_ratio = {}
for fname in selected_names:
    f = selected_factors[fname].copy()
    # 按日做Barra回归
    r2_list = []
    for d in all_dates:
        if d not in f.index:
            continue
        f_row = f.loc[d].dropna()
        # 构建X矩阵
        X_data = []
        valid_stocks = f_row.index.copy()
        for sname in STYLES:
            if sname in style_exposure:
                se_row = style_exposure[sname].loc[d] if d in style_exposure[sname].index else pd.Series(index=valid_stocks)
                valid_stocks = valid_stocks.intersection(se_row.dropna().index)
                X_data.append(se_row)
        if len(valid_stocks) < 50:
            continue
        X = np.column_stack([x[valid_stocks].fillna(0).values for x in X_data])
        y = f_row[valid_stocks].values
        if X.shape[1] >= 2 and len(y) >= 50:
            try:
                reg = LinearRegression().fit(X, y)
                r2_list.append(reg.score(X, y))
            except:
                continue
    alpha_ratio[fname] = 1 - np.mean(r2_list) if r2_list else np.nan
    logger.info(f"  {fname:25s}  R²(Barra)={np.mean(r2_list):.3f}  纯Alpha比例={alpha_ratio[fname]:.3f}")

# 右边: 纯Alpha比例柱状图
ax2 = axes[1]
fnames_list = list(alpha_ratio.keys())
ratios = [alpha_ratio[n] for n in fnames_list]
colors = ['#2ecc71' if r > 0.5 else '#f39c12' if r > 0.3 else '#e74c3c' for r in ratios]
bars = ax2.barh(range(len(fnames_list)), ratios, color=colors)
ax2.set_yticks(range(len(fnames_list)))
ax2.set_yticklabels(fnames_list, fontsize=9)
ax2.set_xlabel('Pure Alpha Ratio (1 - R²)', fontsize=12)
ax2.set_title('Factor Pure Alpha Ratio\n(green>0.5, yellow>0.3, red<0.3)', fontsize=13, fontweight='bold')
ax2.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlim(0, 1)
for i, (v, n) in enumerate(zip(ratios, fnames_list)):
    ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)

plt.tight_layout()
heatmap_path = os.path.join(output_dir, 'barra_exposure_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
logger.info(f"\n  热力图已保存: {heatmap_path}")

# 4. 主要风格暴露识别
logger.info("\n--- 各因子主要风格暴露（|corr| > 0.4） ---")
for fname in selected_names:
    row = exposure_corr.loc[fname]
    major = row[abs(row) > 0.4].sort_values(key=abs, ascending=False)
    if len(major) > 0:
        logger.info(f"  {fname}: {dict(major.round(3))}")
    else:
        logger.info(f"  {fname}: 无明显风格暴露（纯Alpha因子）")

#%% ========== 组合因子构建 ==========
logger.info("\n" + "=" * 70)
logger.info("=== 组合因子构建 ===")
logger.info("=" * 70)

# 计算各因子的 IC 和 IC_IR（用于加权）
ic_stats = {}
for fname in selected_names:
    f = selected_factors[fname].copy()
    # 延迟2期计算IC
    ics = f.shift(2).corrwith(totalRet, axis=1)
    ics = ics[ics.index >= sd_per]
    ic_stats[fname] = {
        'ic_mean': ics.mean(),
        'ic_std': ics.std(),
        'ic_ir': ics.mean() / ics.std() * np.sqrt(252) if ics.std() > 0 else 0
    }

# 1) IC_IR 加权
ic_ir_series = pd.Series({n: ic_stats[n]['ic_ir'] for n in selected_names})
ic_ir_weights = ic_ir_series.abs() / ic_ir_series.abs().sum()

# 2) IC 加权
ic_mean_series = pd.Series({n: ic_stats[n]['ic_mean'] for n in selected_names})
ic_weights = ic_mean_series.abs() / ic_mean_series.abs().sum()

# 3) 等权
equal_weights = pd.Series(1.0 / len(selected_names), index=selected_names)

logger.info("\n因子权重:")
logger.info(f"{'因子名':<25s} {'IC_mean':>8s} {'IC_IR':>8s} {'等权':>8s} {'IC_IR加权':>8s}")
logger.info("-" * 65)
for fname in selected_names:
    logger.info(f"{fname:<25s} {ic_mean_series[fname]:>+8.4f} {ic_ir_series[fname]:>+8.2f} "
          f"{equal_weights[fname]:>8.4f} {ic_ir_weights[fname]:>8.4f}")


def build_combo_factor(selected_factors, weights):
    """合成组合因子"""
    out = None
    for fname, w in weights.items():
        if fname not in selected_factors:
            continue
        f = selected_factors[fname].copy().fillna(0)
        if out is None:
            out = f * w
        else:
            out = out + f * w
    # 截面再标准化
    combo = pn_TransNorm(out.round(4))
    return combo


combo_icir = build_combo_factor(selected_factors, ic_ir_weights)
combo_ic = build_combo_factor(selected_factors, ic_weights)
combo_equal = build_combo_factor(selected_factors, equal_weights)

logger.info(f"\n  组合因子 IC_IR 加权: {combo_icir.shape}")
logger.info(f"  组合因子 IC 加权:   {combo_ic.shape}")
logger.info(f"  组合因子 等权:      {combo_equal.shape}")


#%% ========== 行业 + 风格中性化 ==========
def barra_neutralize(combo, style_exposure, hy, totalRet):
    """
    对组合因子做 Barra 行业+风格中性化
    使用截面回归取残差: combo ~ Styles + Industries
    """
    result = combo.copy()
    dates = combo.index

    for d in dates:
        if d not in combo.index:
            continue
        y = combo.loc[d].dropna()
        if len(y) < 100:
            continue

        # 风格暴露
        X_style = np.zeros((len(y), len(STYLES)))
        valid_mask = np.ones(len(y), dtype=bool)
        for k, sname in enumerate(STYLES):
            if sname in style_exposure and d in style_exposure[sname].index:
                se = style_exposure[sname].loc[d].reindex(y.index)
                X_style[:, k] = se.fillna(0).values
                valid_mask = valid_mask & se.notna().values

        # 行业哑变量
        if d in hy.index:
            hy_row = hy.loc[d].reindex(y.index)
            valid_mask = valid_mask & hy_row.notna().values

        if valid_mask.sum() < 50:
            continue

        yv = y[valid_mask].values
        # 行业哑变量
        hy_valid = hy_row[valid_mask].values.astype(int)
        ind_dummies = pd.get_dummies(hy_valid).values
        # 去掉最后一列避免共线性
        if ind_dummies.shape[1] > 1:
            ind_dummies = ind_dummies[:, :-1]

        X = np.column_stack([X_style[valid_mask]] + [ind_dummies])
        if X.shape[1] < 2:
            continue

        try:
            reg = LinearRegression().fit(X, yv)
            resid = yv - reg.predict(X)
            result.loc[d, y[valid_mask].index] = resid
        except:
            continue

    return result


logger.info("\n=== 行业+风格中性化 ===")
combo_icir_neutral = barra_neutralize(combo_icir, style_exposure, hy, totalRet)
logger.info(f"  中性化后组合因子: {combo_icir_neutral.shape}")

# 中性化后再标准化
combo_final = pn_TransNorm(combo_icir_neutral.round(4))

#%% ========== 保存组合因子 ==========
os.makedirs(save_dir, exist_ok=True)
combo_final.to_pickle(os.path.join(save_dir, 'combo_factor.pkl'))
combo_icir.to_pickle(os.path.join(save_dir, 'combo_raw.pkl'))

# 保存中性化前后对比数据
pd.to_pickle({
    'combo_raw': combo_icir,
    'combo_neutral': combo_final,
    'weights': ic_ir_weights.to_dict(),
    'selected_names': selected_names,
    'exposure_corr': exposure_corr,
    'alpha_ratio': alpha_ratio,
}, os.path.join(save_dir, 'combo_full.pkl'))

logger.info(f"\n  组合因子已保存到: {save_dir}")

logger.info("\n第三阶段+组合因子构建完成！")
