import pathlib
import json
from datetime import datetime, time, timedelta
import os
import pandas as pd
import numpy as np

# Get config file
def read_json_config():
    # Build config file path
    current_dir = pathlib.Path(__file__).parent
    config_path = current_dir.parent.parent / "data" / "config.json"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data
    except FileNotFoundError:
        print(f"Config file does not exist: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON format error: {e}")
        return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None



# Get current date
def get_trade_date():
    """Get trading date (22:00 as cutoff)"""
    now = datetime.now()

    # If current time is before 22:00, use yesterday
    if now.time() < time(22, 0):
        now = now - timedelta(days=1)

    return now.strftime('%Y%m%d')

# Usage example
# trade_date = get_trade_date()
# print(f"Trading date: {trade_date}")


# daily_basic
# close	float	当日收盘价
# turnover_rate	float	换手率（%）
# turnover_rate_f	float	换手率（自由流通股）
# volume_ratio	float	量比
# pe_ttm	float	市盈率（TTM，亏损的PE为空）
# pb	float	市净率（总市值/净资产）
# ps_ttm	float	市销率（TTM）
# dv_ratio	float	股息率 （%）
# dv_ttm	float	股息率（TTM）（%）
# total_share	float	总股本 （万股）
# float_share	float	流通股本 （万股）
# free_share	float	自由流通股本 （万）
# total_mv	float	总市值 （万元）
# circ_mv	float	流通市值（万元）




def combine_rawDt(raw_date_trade,raw_path,df_path,typ):
    path = df_path + typ+ '.pkl'
    if os.path.exists(path):
        daily_raw_total = pd.read_pickle(path)
    else:
        daily_raw_total = pd.DataFrame(columns = ['trade_date'])

    isinDt = daily_raw_total['trade_date'].unique().tolist()
    daily_raw_total1 = []
    for v in raw_date_trade['cal_date']:
        if v in isinDt:
            continue
        tmpP = raw_path + v + '//' + typ+ '.pkl'
        tmp = pd.read_pickle(tmpP)
        daily_raw_total1.append(tmp)
    if len(daily_raw_total1) > 0:
        daily_raw_total1 = pd.concat(daily_raw_total1)
        if len(daily_raw_total) == 0:
            daily_raw_total = daily_raw_total1
        else:
            daily_raw_total = pd.concat([daily_raw_total, daily_raw_total1], axis=0)
        daily_raw_total.to_pickle(path)
    return daily_raw_total


def table2mat(daily,colname,datestr):  # turn fixed table data back to matrix
    d1 = daily[[datestr,'ts_code',colname]]
    data4 = d1.set_index([datestr,'ts_code'])
    data5 = data4.unstack()
    a = list(data5.columns)
    code = list()
    for v in a:
        code.append(v[1])
    data5.columns = code
    return data5





#%% finance

def delete_preList(income__):
    income__ = income__.sort_values(['end_date'],ascending  = False)
    quarter = {3:6,
               6:9,
               9:12,
               12:3}
    month = income__['end_date'].astype(str) .copy() 
    # income__['month'] = list( month.str[-4:-2].astype(int))
    income__['month'] = month.str[-4:-2].astype(int)
    income__ = income__[income__['month'].isin([3,6,9,12])]
    month = income__['month'].tolist()
    for v in range(len(month)):
        if v == len(month)-1 :
            continue
        elif not quarter[month[v+1]] == month[v]:
            # print(month[v],month[v-1])
            break
    income__ = income__.head(v+1)
    return income__

    
def get_Q(income__):  
    base_col = ['ts_code','ann_date','f_ann_date','end_date','report_type','comp_type','update_flag']
    # month = income__['end_date'].astype(str)  
    month = income__['month']
    
    fin_col = [x for x in income__.columns.tolist() if x not in base_col]
    income__[fin_col] = income__[fin_col].fillna(0)
    incomeQ1 = income__[month == 3]
    incomeQ = income__.copy()
    incomeQ[fin_col] = incomeQ[fin_col] - incomeQ[fin_col].shift(-1)
    incomeQ = incomeQ[month.isin([6,9,12])]
    incomeQ = pd.concat([incomeQ,incomeQ1],axis = 0)
    incomeQ = incomeQ.sort_values(by = ['end_date'],ascending = False)
    if not month.values[-1] == 3:
        incomeQ = incomeQ.head(len(incomeQ)-1)
    return incomeQ


def get_A(z,method = 'sum'):  
    base_col = ['ts_code','ann_date','f_ann_date','end_date','report_type','comp_type','update_flag']
    fin_col = [x for x in z.columns.tolist() if x not in base_col]
    incomeA = z.sort_values(by = ['end_date'],ascending = True)
    if method == 'sum':
        incomeA[fin_col] = incomeA[fin_col].rolling(4).sum()
    elif method == 'mean':
        incomeA[fin_col] = incomeA[fin_col].rolling(4).mean()
    incomeA = incomeA.iloc[3:]
    return incomeA
    


def get_uniqueIdx(balance__,income__,cash__):
    intersection = list(set(balance__['end_date'].tolist()) & set(income__['end_date'].tolist()) & set(cash__['end_date'].tolist()))
    balance__ = balance__[balance__['end_date'].isin(intersection)]
    income__ = income__[income__['end_date'].isin(intersection)]
    cash__ = cash__[cash__['end_date'].isin(intersection)]
    return balance__,income__,cash__
    

def clean_fin(balance__,income__,cash__):
    # 清洗和对齐
    # 1 删除重复行，需要后续调整， 考虑update_flag
    # balance__ = balance__.drop_duplicates(subset=['end_date'], keep='first')
    # income__ =  income__.drop_duplicates(subset=['end_date'], keep='first')
    # cash__ =  cash__.drop_duplicates(subset=['end_date'], keep='first')
    # 2 删除上市前不连续的行
    balance__ = delete_preList(balance__)
    income__ = delete_preList(income__)
    cash__ = delete_preList(cash__)
    
    # 3 季度化
    incomeQ = get_Q(income__)
    cashQ = get_Q(cash__)
    balanceQ = balance__.fillna(0)
    
    # 4 对齐end_date
    balanceQ,incomeQ,cashQ = get_uniqueIdx(balanceQ,incomeQ,cashQ)
    
    # 5 年化
    incomeA = get_A(incomeQ,'sum')
    cashA = get_A(cashQ,'sum')
    balanceA = get_A(balanceQ,'mean')
    
    incomeA.index = incomeA['end_date']
    cashA.index = cashA['end_date']
    balanceA.index = balanceA['end_date']
    
    return balanceA,incomeA,cashA 


        
def GetQuantileRet(CF,TotalRet,q_,delayNum):
    from feature import Repmat
    # rankF = CF.rank(axis = 1)
    CF2 = CF
    qr = pd.DataFrame()
    for v in range(q_):
        f_ = CF2.copy()
        lowBan = f_.quantile(v/q_,axis =1)
        upBan = f_.quantile(v/q_ + 1/q_,axis =1)
        lowBan = Repmat(f_,lowBan) 
        upBan = Repmat(f_,upBan) 
        f2 = pd.DataFrame(1,index = CF2.index,columns = CF2.columns)
        f2[f_ < lowBan] = np.nan
        f2[f_ >= upBan] = np.nan
        StraRetLine =  (f2.shift(delayNum) * TotalRet).mean(1) 
        qr[v] = StraRetLine
    return qr


def get_fit2mat(df_test2,colname,datestr,baseMatrix):  # turn fixed table data back to matrix
    df_test = df_test2.copy()
    df = df_test[[datestr,'ts_code',colname] ]
    df = df.drop_duplicates(subset=['ts_code', 'ann_date_last'], keep='last')
    data4=df.set_index([datestr,'ts_code'])
    data5=data4.unstack()
    indx = list(data5.index)
    a = list(data5.columns)
    code = list()
    for v in a:
        code.append(v[1])
    out = pd.DataFrame(data5.values)
    out.index = pd.to_datetime(indx)
    out.columns = code

    start_date = baseMatrix.index.min()
    end_date = baseMatrix.index.max()
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    out = out.reindex(full_dates).ffill()

    common_cols = out.columns.intersection(baseMatrix.columns)
    c = baseMatrix.fillna(0) * 0
    c[common_cols] = out[common_cols]
    return c