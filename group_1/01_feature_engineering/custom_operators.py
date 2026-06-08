"""
custom_operators.py
补充的算子函数 —— 用于复现已有关子库尚未覆盖的因子逻辑

所有算子返回与输入同形状的 DataFrame。
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. 安全除法 / 比率算子（避免除零）
# ============================================================
def safe_div(a, b, fill=0.0):
    """
    安全除法：a / b, b==0 或 NaN 时用 fill 填充
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
        out = out.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return out


# ============================================================
# 2. 截面排名百分位（返回 0~1 之间的百分比）
# ============================================================
def pn_Rank(X):
    """截面排名百分位: rank(axis=1, pct=True)"""
    return X.rank(axis=1, pct=True)


# ============================================================
# 3. 截面标准化 (Z-score)
# ============================================================
def pn_Stand(X):
    """截面 Z-score 标准化: (x - μ) / σ"""
    return X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1), axis=0)


def Normalize(X):
    """同 pn_Stand"""
    return pn_Stand(X)


# ============================================================
# 4. 截面极值截断 (Winsorize)
# ============================================================
def pn_Cut(X, low=None, high=None):
    """
    截面截断：按分位数或指定值截断异常值
    low/high: 如 low=0.01, high=0.99 表示按1%和99%分位数截断
    也可以传入标量值（如 low=-3, high=3 表示按绝对值截断）
    """
    values = X.values.astype(float).copy()
    for i in range(values.shape[0]):
        row = values[i]
        valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        row_valid = row[valid]
        lo = np.percentile(row_valid, low * 100) if low is not None and 0 < low < 1 else low
        hi = np.percentile(row_valid, high * 100) if high is not None and 0 < high < 1 else high
        if lo is not None:
            row_valid = np.clip(row_valid, lo, None)
        if hi is not None:
            row_valid = np.clip(row_valid, None, hi)
        row[valid] = row_valid
        values[i] = row
    return pd.DataFrame(values, index=X.index, columns=X.columns)


# ============================================================
# 5. 截面缩放
# ============================================================
def pn_Scale(X):
    """截面 min-max 缩放至 [0,1]"""
    _min = X.min(axis=1)
    _max = X.max(axis=1)
    return X.sub(_min, axis=0).div((_max - _min).replace(0, np.nan), axis=0)


# ============================================================
# 6. 行业中性化（组内去均值）
# ============================================================
def pn_GroupNeutral(X, GrpLabel):
    """行业内去均值——消除行业影响"""
    result = X.copy()
    for col in X.columns:
        s = X[col]
        g = GrpLabel[col] if col in GrpLabel.columns else np.nan
        if pd.isna(g):
            result[col] = s
        else:
            result[col] = s - g
    # 逐日逐行业去均值
    dates = X.index
    for d in dates:
        x_row = X.loc[d]
        g_row = GrpLabel.loc[d] if d in GrpLabel.index else pd.Series(index=X.columns)
        for grp in g_row.unique():
            mask = g_row == grp
            if mask.sum() > 0:
                vals = x_row[mask]
                result.loc[d, mask] = vals - vals.mean()
    return result


def pn_GroupNorm(X, GrpLabel):
    """行业内 Z-score 标准化"""
    result = X.copy()
    dates = X.index
    for d in dates:
        x_row = X.loc[d]
        g_row = GrpLabel.loc[d] if d in GrpLabel.index else pd.Series(index=X.columns)
        for grp in g_row.unique():
            mask = g_row == grp
            vals = x_row[mask]
            if mask.sum() > 1:
                m, s = vals.mean(), vals.std()
                if s > 0:
                    result.loc[d, mask] = (vals - m) / s
    return result


# ============================================================
# 7. 截面回归取残差
# ============================================================
def pn_CrossFit(X, Y):
    """
    截面回归取残差：X 对 Y 回归的残差
    用于中性化（如对市值/行业做正交）
    """
    result = X.copy()
    dates = X.index
    for d in dates:
        x_row = X.loc[d].dropna()
        y_row = Y.loc[d].dropna() if isinstance(Y, pd.DataFrame) else Y.loc[d]
        common = x_row.index.intersection(y_row.index if isinstance(y_row, pd.Series) else y_row.columns)
        if len(common) < 10:
            continue
        xx = x_row[common].values.astype(float)
        yy = y_row[common].values.astype(float) if isinstance(y_row, pd.Series) else y_row.loc[common].values.astype(float)
        if yy.ndim == 1:
            yy = yy.reshape(-1, 1)
        # OLS 残差
        try:
            beta = np.linalg.lstsq(yy, xx, rcond=None)[0]
            pred = yy @ beta
            resid = xx - pred
            result.loc[d, common] = resid
        except np.linalg.LinAlgError:
            continue
    return result


# ============================================================
# 8. 时序算子补充
# ============================================================
def ts_Rank(X, N):
    """过去 N 期的时序排名百分比"""
    return X.rolling(N).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)


def ts_ChgRate(X, N):
    """过去 N 期的变化率: X / shift(X, N) - 1"""
    return safe_div(X, X.shift(N)) - 1


def ts_Corr(X, Y, N):
    """
    N 日滚动相关系数（逐股票计算，返回与输入同形状的 DataFrame）
    注意：DataFrame.rolling().corr(DataFrame) 返回的是列对列的 MultiIndex，
    必须逐列用 Series.rolling().corr(Series) 才能得到 (T, N) 矩阵。
    """
    result = X.copy()
    result[:] = np.nan
    for col in X.columns:
        if col in Y.columns:
            result[col] = X[col].rolling(N).corr(Y[col])
    return result


def ts_Cov(X, Y, N):
    """
    N 日滚动协方差（逐股票计算，返回 (T, N) 矩阵）
    """
    result = X.copy()
    result[:] = np.nan
    for col in X.columns:
        if col in Y.columns:
            result[col] = X[col].rolling(N).cov(Y[col])
    return result


def ts_RegressionFit(Y, X, N):
    """
    N 日滚动回归 R²
    Y 对 X 回归的拟合优度
    """
    result = pd.DataFrame(np.nan, index=Y.index, columns=Y.columns)
    for col in Y.columns:
        y_s = Y[col]
        x_s = X[col] if isinstance(X, pd.DataFrame) and col in X.columns else X[col]
        # 滚动回归 R2 近似
        for i in range(N - 1, len(Y)):
            yy = y_s.iloc[i - N + 1: i + 1].dropna()
            xx = x_s.iloc[i - N + 1: i + 1].dropna()
            common = yy.index.intersection(xx.index)
            if len(common) < max(5, N // 2):
                continue
            yv, xv = yy[common].values, xx[common].values
            corr = np.corrcoef(xv, yv)[0, 1]
            result.iloc[i, Y.columns.get_loc(col)] = corr ** 2 if not np.isnan(corr) else np.nan
    return result


# ============================================================
# 9. 数学函数
# ============================================================
def Log(X):
    """自然对数"""
    return np.log(X.replace(0, np.nan))


def Abs(X):
    """绝对值"""
    return X.abs()


def Sign(X):
    """符号函数"""
    return np.sign(X)


def Sqrt(X):
    """平方根"""
    return np.sqrt(X.clip(lower=0))


# ============================================================
# 10. Bollinger Bands 相关
# ============================================================
def ts_BBOLL_PctB(X, N=20, k=2):
    """
    Bollinger %B: (price - lower) / (upper - lower)
    """
    mid = X.rolling(N).mean()
    std = X.rolling(N).std()
    upper = mid + k * std
    lower = mid - k * std
    return safe_div(X - lower, upper - lower, 0.5)


# ============================================================
# 11. MACD 相关
# ============================================================
def ts_MACD(X, fast=12, slow=26, signal=9):
    """
    MACD 信号线: DIF = EMA(fast) - EMA(slow), 返回 MACD 柱 = DIF - DEA
    """
    ema_fast = X.ewm(span=fast, adjust=False).mean()
    ema_slow = X.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif - dea


# ============================================================
# 12. 资金流相关
# ============================================================
def ts_FundFlowRatio(buy_vol, sell_vol, N=20):
    """
    资金流向比率: 净买入 / (买入+卖出)
    """
    net = buy_vol - sell_vol
    total = buy_vol + sell_vol
    ratio = safe_div(net, total, 0)
    return ratio.rolling(N).mean()


# ============================================================
# 13. Amihud 非流动性
# ============================================================
def ts_Amihud(ret, amount, N=20):
    """
    Amihud 非流动性: |ret| / amount 的 N 日均值
    amount 单位需统一（原数据为千元）
    """
    illiq = safe_div(ret.abs(), amount, 0)
    return illiq.rolling(N).mean()


# ============================================================
# 14. 筹码集中度 / 价格区间
# ============================================================
def ts_PriceRange(high, low, N=20):
    """价格区间比: (max_high - min_low) / mean_close"""
    hh = high.rolling(N).max()
    ll = low.rolling(N).min()
    return safe_div(hh - ll, (high + low).rolling(N).mean() / 2, 0)


print("custom_operators.py — 所有补充算子加载完毕")
