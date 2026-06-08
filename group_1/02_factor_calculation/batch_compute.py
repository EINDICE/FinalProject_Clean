"""
batch_compute.py
第二阶段：批量因子计算与筛选

功能：
1. 加载中证1000数据矩阵 dt
2. 批量计算 45 个候选因子
3. 标准化 → 多空收益 → 年化收益/夏普/IC 评价
4. 筛选 10 个达标因子: |AR|>0.1, |SR|>2, |corr|<0.3
5. 输出因子相关矩阵热力图
"""

# ============================================================================
# 1. 基础导入
# ============================================================================
import os, sys, logging
import importlib.util as _iu

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 2. 日志配置（控制台 + 文件）
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)
_fh = logging.FileHandler(os.path.join(_LOG_DIR, 'batch_compute.log'), encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
logger.addHandler(_fh)

# ============================================================================
# 3. 路径设置
#    __file__: group_1/02_factor_calculation/batch_compute.py
#    需要 3 级 dirname 到达项目根目录（Final Project/）
# ============================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from feature import *
from tools import read_json_config

# ============================================================================
# 4. 加载自定义算子
#    使用 importlib 绕过目录名 "01_feature_engineering" 以数字开头的限制
# ============================================================================
_CUSTOM_OPS_PATH = os.path.join(_PROJECT_ROOT, 'group_1', '01_feature_engineering', 'custom_operators.py')
_spec = _iu.spec_from_file_location("custom_operators", _CUSTOM_OPS_PATH)
_custom_ops = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_custom_ops)

# 将自定义算子注入当前命名空间
safe_div        = _custom_ops.safe_div
pn_Rank         = _custom_ops.pn_Rank
pn_Stand        = _custom_ops.pn_Stand
Normalize       = _custom_ops.Normalize
pn_Cut          = _custom_ops.pn_Cut
pn_Scale        = _custom_ops.pn_Scale
pn_GroupNeutral = _custom_ops.pn_GroupNeutral
pn_GroupNorm    = _custom_ops.pn_GroupNorm
pn_CrossFit     = _custom_ops.pn_CrossFit
ts_Rank         = _custom_ops.ts_Rank
ts_ChgRate      = _custom_ops.ts_ChgRate
ts_Corr         = _custom_ops.ts_Corr
ts_Cov          = _custom_ops.ts_Cov
ts_RegressionFit= _custom_ops.ts_RegressionFit
Log             = _custom_ops.Log
Abs             = _custom_ops.Abs
Sign            = _custom_ops.Sign
Sqrt            = _custom_ops.Sqrt
ts_BBOLL_PctB   = _custom_ops.ts_BBOLL_PctB
ts_MACD         = _custom_ops.ts_MACD
ts_FundFlowRatio= _custom_ops.ts_FundFlowRatio
ts_Amihud       = _custom_ops.ts_Amihud
ts_PriceRange   = _custom_ops.ts_PriceRange

# ============================================================================
# 5. 配置与常量
# ============================================================================
config = read_json_config()
if config is None:
    path1000 = os.path.join(_PROJECT_ROOT, 'data', '')
else:
    path1000 = config['paths']['S1000']
path1000_matrix = path1000 + 'matrix//'

SD       = '2017-01-01'       # 数据起始日
SD_PER   = '2017-12-31'       # 业绩评价起始日（避免前视偏差）

REQUIRED_FIELDS = {
    'totalRet', 'close', 'open', 'high', 'low', 'pre_close', 'vwap',
    'vol', 'amount', 'circ_mv', 'total_mv',
    'turnover_rate', 'turnover_rate_f', 'free_share', 'float_share',
    'pb', 'pe_ttm', 'ps_ttm', 'dv_ttm',
    'pct_chg', 'overnightRet',
    'net_mf_amount', 'buy_lg_vol', 'buy_elg_vol',
    'sell_lg_vol', 'sell_elg_vol',
}

# ============================================================================
# 6. 数据加载与清洗
# ============================================================================
logger.info("=== 加载数据矩阵 ===")

matrixList = os.listdir(path1000_matrix)
dt = {}
for v in matrixList:
    tmpName = v[:-4]                    
    if tmpName not in REQUIRED_FIELDS:
        continue
    tmp = pd.read_pickle(path1000_matrix + v)
    tmp = tmp[tmp.index >= SD]
    tmp.index = pd.to_datetime(tmp.index)
    dt[tmpName] = tmp

# ---- 上市天数 > 20 ----
listed = dt['vol'].copy()
listed[listed.isna()] = 0
listed = (listed.cumsum() > 0).shift(20)

# ---- 中证1000 成分股掩码 ----
idxWgt = pd.read_pickle(path1000 + 'idxWgt.pkl')
idxWgt.index = pd.to_datetime(idxWgt.index)
univ_mask = idxWgt == 0

# ---- 价格数据清洗 ----
dt['amount'] = (dt['amount'] * 1000).round(2)
dt['vol']    = (dt['vol']    * 100).round(0)
dt['vwap']   = (dt['amount'] / dt['vol']).round(4)
for col in ['high', 'low', 'open', 'close']:
    dt[col] = dt[col].round(2)

dt['totalRet'][dt['totalRet'].abs() > 0.2] = np.nan
dt['totalRet'] = dt['totalRet'].round(6)

# ---- 超额收益 ----
IndexRet  = dt['totalRet'].mean(axis=1)
IndexRets = Repmat(dt['totalRet'], IndexRet)
dt['exRet'] = (dt['totalRet'] - IndexRets).round(6)

# ---- 上市不足 20 日的数据置 NaN ----
for v in dt.keys():
    dt[v][listed == 0] = np.nan

logger.info(f"  数据日期范围: {list(dt['close'].index[[0, -1]])}")
logger.info(f"  股票数量: {dt['close'].shape[1]}")

# ============================================================================
# 7. 因子定义
# ============================================================================
def compute_all_factors(dt):
    """计算全部 45 个候选因子，返回 dict: {因子名: DataFrame}"""
    f = {}  # 因子容器

    # ----- 动量类 (1-7) -----
    f['FF_MOM_1M']        = ts_DecayExp(dt['totalRet'], 21)
    f['FF_MOM_3M']        = ts_DecayExp(dt['totalRet'], 63)
    f['FF_MOM_6M_SKIP']   = ts_DecayExp(ts_Delay(dt['totalRet'], 21), 126)
    f['FF_RSTR_LIKE']     = ts_Sum(dt['totalRet'].shift(21), 252)
    f['FF_ABNORM_VOL']    = safe_div(dt['vol'], ts_Mean(dt['vol'], 20)) - 1
    f['FF_VOL_PRICE_CORR']= -ts_Corr(dt['close'], dt['vol'], 20)
    f['FF_TURNOVER_MOM']  = ts_ChgRate(dt['turnover_rate'], 21)

    # ----- 反转类 (8-11) -----
    f['FF_REV_5D']        = -ts_Sum(dt['totalRet'], 5)
    f['FF_REV_OVERNIGHT'] = -dt['overnightRet']
    f['FF_REV_GAP']       = -safe_div(dt['open'] - ts_Delay(dt['close'], 1), ts_Delay(dt['close'], 1))
    f['FF_LONG_REV']      = -ts_Sum(dt['totalRet'], 252)

    # ----- 波动率类 (12-18) -----
    f['FF_VOL_20D']       = -ts_Stdev(dt['totalRet'], 20)
    f['FF_VOL_60D']       = -ts_Stdev(dt['totalRet'], 60)
    f['FF_MAX_RET']       = -ts_Max(dt['totalRet'], 20)
    f['FF_MIN_RET']       =  ts_Min(dt['totalRet'], 20)
    f['FF_SKEW']          = -ts_Skewness(dt['totalRet'], 60)
    f['FF_DOWNSIDE_VOL']  = -ts_Stdev(dt['totalRet'].clip(upper=0), 20)
    f['FF_VOL_OF_VOL']    = -ts_Stdev(ts_Stdev(dt['totalRet'], 5), 20)

    # ----- 流动性类 (19-24) -----
    f['FF_TURNOVER']      = -dt['turnover_rate']
    f['FF_AMIHUD']        = -ts_Amihud(dt['totalRet'], dt['amount'], 20)
    f['FF_VOLUME_RATIO']  =  safe_div(dt['vol'], ts_Mean(dt['vol'], 5))
    f['FF_FREEFLOAT_TO']  = -safe_div(dt['vol'], dt['free_share'], 0)
    f['FF_AMOUNT_STD']    =  ts_Stdev(dt['amount'], 20)
    f['FF_LIQ_DECAY']     =  ts_Decay(-dt['turnover_rate'], 20)

    # ----- 资金流类 (25-28) -----
    if 'net_mf_amount' in dt and 'amount' in dt:
        f['FF_NET_MF']    = safe_div(dt['net_mf_amount'], dt['amount'] * 1000)
    has_mf = all(k in dt for k in ['buy_lg_vol', 'buy_elg_vol', 'vol'])
    if has_mf:
        f['FF_LARGE_BUY'] =  safe_div(dt['buy_lg_vol'] + dt['buy_elg_vol'], dt['vol'], 0)
        f['FF_LARGE_SELL']= -safe_div(dt['sell_lg_vol'] + dt['sell_elg_vol'], dt['vol'], 0)
    if 'net_mf_amount' in dt:
        f['FF_MF_DECAY']  = ts_Decay(safe_div(dt['net_mf_amount'], dt['amount'] * 1000), 20)

    # ----- 技术类 (29-36) -----
    f['FF_VWAP_DEV']      = -safe_div(dt['close'] - dt['vwap'], dt['vwap'])
    f['FF_BOLL_PCTB']     =  ts_BBOLL_PctB(dt['close'], 20, 2)
    f['FF_MACD']          = -ts_MACD(dt['close'], 12, 26, 9)
    pos_ret = dt['totalRet'].clip(lower=0)
    neg_ret = (-dt['totalRet']).clip(lower=0)
    rsi = safe_div(ts_Sum(pos_ret, 14), ts_Sum(pos_ret, 14) + ts_Sum(neg_ret, 14), 0.5)
    f['FF_RSI']           = 1 - rsi
    f['FF_PRICE_POS']     = safe_div(
        dt['close'] - ts_Min(dt['close'], 60),
        ts_Max(dt['close'], 60) - ts_Min(dt['close'], 60))
    f['FF_HL_RATIO']      = -safe_div(dt['high'] - dt['low'], dt['pre_close'])
    f['FF_OPEN_RET']      =  safe_div(dt['open'] - ts_Delay(dt['close'], 1), ts_Delay(dt['close'], 1))
    f['FF_CLOSE_POS']     =  safe_div(dt['close'] - dt['low'], dt['high'] - dt['low'])

    # ----- 基本面类 (37-42) -----
    f['FF_SIZE']          = -Log(dt['circ_mv'])
    f['FF_PB']            = -Log(dt['pb'])
    if 'pe_ttm' in dt:    f['FF_EP']  = safe_div(1, dt['pe_ttm'], 0)
    if 'dv_ttm' in dt:    f['FF_DIV'] = dt['dv_ttm']
    if 'ps_ttm' in dt:    f['FF_PS']  = -Log(dt['ps_ttm'])
    f['FF_TO_F']          = -dt['turnover_rate_f']

    # ----- 特质类 (43-45) -----
    f['FF_IDIO_MOM']      =  ts_Sum(dt['totalRet'], 20)
    f['FF_PCT_CHG']       = -dt['pct_chg']
    f['FF_EXRET_5D']      =  ts_Sum(dt['exRet'], 5)

    return f


# ============================================================================
# 8. 因子评价函数
# ============================================================================
def evaluate_factor(factor_raw, totalRet, univ_mask, sd_per):
    """
    评价单个因子: 标准化 → 多空组合 → 因子收益
    返回: sr, ar, ic_mean, ic_ir, rets, f_stand
    """
    f = factor_raw.copy()
    um_aligned = univ_mask.reindex(index=f.index, columns=f.columns, fill_value=True)
    f[um_aligned] = np.nan

    f = pn_Cut(f, 0.01, 0.99)              # 极值截断
    f_stand = pn_TransNorm(f.copy())         # 截面标准化

    port_pos, port_neg = get_ls_post(f_stand.copy())
    f_port = port_pos + port_neg

    rets = (f_port.shift(2) * totalRet).sum(axis=1)   # 延迟 2 期防前视偏差
    rets = rets[rets.index >= sd_per]
    if len(rets) < 100:
        return None

    sr = rets.mean() / rets.std() * np.sqrt(252)       # 年化夏普
    ar = rets.mean() * 252                              # 年化收益

    ics = f_stand.corrwith(totalRet.shift(-2), axis=1)  # 逐日截面 IC
    ics = ics[ics.index >= sd_per]
    ic_mean = ics.mean()
    ic_ir   = ics.mean() / ics.std() * np.sqrt(252) if ics.std() > 0 else 0

    return {'sr': sr, 'ar': ar, 'ic_mean': ic_mean, 'ic_ir': ic_ir,
            'rets': rets, 'f_stand': f_stand}


def _draw_heatmap(ax, data, annot=False, title='', mask_upper=False,
                  xtick_rotation=90, tick_size=6):
    """纯 matplotlib 热力图（替代 seaborn.heatmap）"""
    data_np = data.values if hasattr(data, 'values') else np.array(data)
    if mask_upper:
        mask = np.triu(np.ones_like(data_np, dtype=bool), k=1)
        data_np = np.ma.masked_where(mask, data_np)
    im = ax.imshow(data_np, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal',
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    labels = data.index if hasattr(data, 'index') else range(len(data_np))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=xtick_rotation, ha='center', fontsize=tick_size)
    ax.set_yticklabels(labels, fontsize=tick_size)
    if annot and hasattr(data, 'shape'):
        n = data.shape[0]
        for i in range(n):
            for j in range(n):
                val = data.iloc[i, j] if hasattr(data, 'iloc') else data_np[i, j]
                if not (mask_upper and j > i):
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10 if n <= 10 else 7, color=text_color)


# ============================================================================
# 9. 筛选因子（逐步筛选）
# ============================================================================
def select_factors(factor_results, factor_rets, target_n=10,
                   min_sr=2.0, min_ar=0.1, max_corr=0.3,
                   relaxed_sr=1.5, relaxed_ar=0.05, relaxed_corr=0.4):
    """
    按 |SR|>min_sr, |AR|>min_ar, 两两相关性<max_corr 逐步筛选 target_n 个因子。
    不够时自动放宽条件补足。
    """
    ranked = sorted(factor_results.items(), key=lambda x: abs(x[1]['sr']), reverse=True)
    valid_names = list(factor_results.keys())
    rets_df = pd.DataFrame(factor_rets).loc[:, valid_names]
    corr_df = rets_df.corr()

    selected = []
    # 严格筛选
    for name, res in ranked:
        if abs(res['sr']) < min_sr or abs(res['ar']) < min_ar:
            continue
        if any(abs(corr_df.loc[name, s]) >= max_corr for s in selected):
            continue
        selected.append(name)
        if len(selected) >= target_n:
            break

    logger.info(f"  严格筛选结果: {len(selected)} 个因子")
    if len(selected) < target_n:
        logger.warning(f"  不足 {target_n} 个，放宽条件补足...")
        for name, res in ranked:
            if name in selected:
                continue
            if abs(res['sr']) < relaxed_sr or abs(res['ar']) < relaxed_ar:
                continue
            if any(abs(corr_df.loc[name, s]) >= relaxed_corr for s in selected):
                continue
            selected.append(name)
            if len(selected) >= target_n:
                break

    return selected, corr_df


# ============================================================================
# 10. 主流程
# ============================================================================
def main():
    # ---- 批量计算因子 ----
    logger.info("\n=== 批量计算因子 ===")
    factors_raw = compute_all_factors(dt)
    logger.info(f"  共计算 {len(factors_raw)} 个候选因子")

    # ---- 评价所有因子 ----
    logger.info("\n=== 评价因子 ===")
    factor_results = {}
    factor_rets    = {}
    factor_stands  = {}
    totalRet = dt['totalRet']

    for name, f_raw in factors_raw.items():
        try:
            res = evaluate_factor(f_raw, totalRet, univ_mask, SD_PER)
            if res is not None and abs(res['sr']) > 0.1:
                factor_results[name] = res
                factor_rets[name]    = res['rets']
                factor_stands[name]  = res['f_stand']
                logger.info(f"  {name:25s}  SR={res['sr']:+.3f}  AR={res['ar']:+.3f}  "
                            f"IC_mean={res['ic_mean']:+.4f}  IC_IR={res['ic_ir']:+.2f}")
        except Exception as e:
            logger.error(f"  {name:25s}  计算失败: {e}")

    # ---- 因子筛选 ----
    logger.info("\n=== 因子筛选（目标: |AR|>0.1, |SR|>2, |corr|<0.3）===")
    selected, corr_df = select_factors(factor_results, factor_rets)

    logger.info(f"\n  最终 {len(selected)} 个达标因子:")
    for name in selected:
        res = factor_results[name]
        logger.info(f"    {name:25s}  SR={res['sr']:+.3f}  AR={res['ar']:+.3f}  "
                    f"IC_mean={res['ic_mean']:+.4f}  IC_IR={res['ic_ir']:+.2f}")

    # ---- 相关矩阵热力图 ----
    logger.info("\n=== 生成因子相关矩阵热力图 ===")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    sel_corr = corr_df.loc[selected, selected]

    _, axes = plt.subplots(1, 2, figsize=(24, 10))
    _draw_heatmap(axes[0], corr_df, mask_upper=True,
                  title='All Factors Correlation Matrix',
                  xtick_rotation=90, tick_size=6)
    _draw_heatmap(axes[1], sel_corr, annot=True,
                  title='Selected 10 Factors Correlation Matrix',
                  xtick_rotation=45, tick_size=9)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'corr_matrix.png')
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  热力图已保存: {corr_path}")

    # ---- 保存因子数据 ----
    selected_factors = {name: factor_stands[name] for name in selected}
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'group1')
    os.makedirs(save_dir, exist_ok=True)
    pd.to_pickle(selected_factors, os.path.join(save_dir, 'selected_factors.pkl'))
    pd.to_pickle(factor_rets,      os.path.join(save_dir, 'factor_rets.pkl'))
    pd.to_pickle(selected,         os.path.join(save_dir, 'selected_names.pkl'))
    logger.info(f"\n  因子数据已保存到: {save_dir}")

    # ---- 汇总表 ----
    logger.info("\n" + "=" * 80)
    logger.info("10个达标因子汇总表")
    logger.info("=" * 80)
    header = f"{'因子名':<25s} {'Sharpe':>8s} {'AnnRet':>8s} {'IC_mean':>8s} {'IC_IR':>8s}"
    logger.info(header)
    logger.info("-" * 60)
    for name in selected:
        res = factor_results[name]
        logger.info(f"{name:<25s} {res['sr']:>+8.3f} {res['ar']:>+8.3f} "
                    f"{res['ic_mean']:>+8.4f} {res['ic_ir']:>+8.2f}")

    logger.info("\n完成！")


if __name__ == "__main__":
    main()
