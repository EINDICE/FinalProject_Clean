
import pandas as pd
import numpy as np

#%% get data

import math
import scipy.stats as st
##  PN

def Repmat(f_,lowBan):
    a = np.array([lowBan.values]).T
    c = np.tile(a,len(f_.columns))
    try:
        z = pd.DataFrame(c, index = f_.index, columns = f_.columns)
    except:
        c = np.reshape(c,(np.shape(c)[1],np.shape(c)[2]))
        z = pd.DataFrame(c, index = f_.index, columns = f_.columns)
    return z

def pn_TransNorm(s1):  # panel normalization
    from scipy.stats import norm
    dfCleaned = s1.copy()
    rank_ = dfCleaned.rank(pct=True, axis=1)
    cut = rank_.min(axis=1) / 2
    rank_ = rank_.sub(cut, axis=0)
    out = pd.DataFrame(norm.ppf(np.array(rank_)), index=rank_.index, columns=rank_.columns)
    return out


def get_portFromFactor_both(f, num):
    f_ = f.copy()
    f_2 = f_.rank(axis=1,ascending=False,method = 'first')
    hold = f_.fillna(0) * 0
    hold[f_2 <= num] = 1
    hold = hold / Repmat(hold,hold.sum(1))
    hold2 = f_.fillna(0) * 0
    hold2[f_2 > Repmat(f_2,f_2.max(1)) - num] = 1
    hold2 = hold2 / Repmat(hold2,hold2.sum(1)) *-1
    return hold,hold2


def get_ls_post(f_D2):
    z_up = f_D2.copy()
    z_dn = f_D2.copy()
    z_up[z_up < 0] = 0
    z_dn[z_dn > 0] = 0
    z_up = z_up.div(z_up.sum(1), axis=0)
    z_dn = z_dn.div(z_dn.sum(1), axis=0) * -1
    return z_up.fillna(0), z_dn.fillna(0)


# calculator3
## TS
def ts_Delay(df2, num ):
    df = df2.copy()
    df3 = df.shift(num)   # shift, like:  df.shift(1), let yesterday's data to today
    return df3

def ts_Mean(df2, num):                # equal weight
    df = df2.copy()
    df = df.rolling(window=num).mean()
    return df



def _weighted_rolling(dataTD, w):
    """
    General weighted rolling calculation, replacing np.roll implementation.
    w: weight array, from far to near, length = window period. Automatically normalized.
    NaN handling: when window contains NaN, only calculate non-NaN values with normalized weights;
                 output NaN when non-NaN values are less than 2.
    """
    nPrds = len(w)
    if nPrds <= 0:
        return dataTD.copy() if isinstance(dataTD, pd.DataFrame) else dataTD
    w = w / w.sum()
    # Convert to numpy for acceleration
    arr = np.array(dataTD, dtype=np.float64)
    rows, cols = arr.shape
    result = np.full_like(arr, np.nan)
    for t in range(nPrds - 1, rows):
        window = arr[t - nPrds + 1:t + 1, :]  # shape (nPrds, cols)
        valid = ~np.isnan(window)
        valid_count = valid.sum(axis=0)  # shape (cols,)
        # For each column: use normalized weights for valid values to calculate weighted sum
        for j in range(cols):
            vc = valid_count[j]
            if vc < 2:
                continue
            w_valid = w[valid[:, j]]
            w_norm = w_valid / w_valid.sum()
            result[t, j] = np.dot(w_norm, window[valid[:, j], j])
    return pd.DataFrame(result, index=dataTD.index, columns=dataTD.columns)


def ts_Decay(dataTD, nPrds):
    """Linear decay weighted mean: weights increase linearly from far to near"""
    nPrds = int(nPrds)
    if nPrds <= 0:
        return dataTD.copy() if isinstance(dataTD, pd.DataFrame) else dataTD
    w = np.array([1 - 1/nPrds * (i - 1) for i in range(nPrds, 0, -1)])
    return _weighted_rolling(dataTD, w)


def ts_DecayExp(dataTD2, nPrds):
    """Exponential decay weighted mean: weights increase exponentially from far to near (EWMA style)"""
    nPrds = int(nPrds)
    if nPrds <= 0:
        return dataTD2.copy() if isinstance(dataTD2, pd.DataFrame) else dataTD2
    alpha = 1 - 2 / (nPrds + 1)
    w = np.array([alpha**i for i in range(nPrds, 0, -1)])
    return _weighted_rolling(dataTD2, w)


def ts_Max(df2, num):                # get the max value of last num trading day
    df = df2.copy()
    df = df.rolling(window=num).max()
    return df

def ts_Min(df2, num):             # get the min value of last num trading day
    df = df2.copy()
    df = df.rolling(window=num).min()
    return df

def ts_Delta(df2, num):
    df = df2.copy()
    df = df - ts_Delay(df,num)
    return df

def ts_Stdev(df2, num):             # get the min value of last num trading day
    df = df2.copy()
    df = df.rolling(num).std()
    return df

def ts_Sum(df2, num):
    dfCleaned = df2.copy()
    stds = dfCleaned.rolling(window=num).sum()
    return stds

def ts_Kurtosis(df2, num):
    dfCleaned = df2.copy()
    stds = dfCleaned.rolling(window=num).kurt()
    return stds

def ts_Skewness(df2, num):
    dfCleaned = df2.copy()
    stds = dfCleaned.rolling(window=num).skew()
    return stds

def ts_Median(df2, num):
    dfCleaned = df2.copy()
    stds = dfCleaned.rolling(window=num).median()
    return stds

# more calculator , see df.rolling
