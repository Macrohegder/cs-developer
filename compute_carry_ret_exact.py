"""
按照旧版 CSstrategy_summary 的严谨定义计算 carry_ret：
  carry = ln(mean(F1_cycle) / mean(F2_cycle)) / delta_days * 365

其中：
  - F1/F2 使用 88/88A2 连续合约收盘价
  - mean 取 cycle 天（默认5天）的简单平均
  - delta_days 使用真实合约到期日差（从 DominantManager 映射的真实合约查询）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/tmp/trading-system--master-vnpy_alpharesearch/vnpy_alpharesearch")

import pandas as pd
import numpy as np
from datetime import datetime
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.utility import load_bar_df

START = datetime(2020, 1, 1)
END = datetime(2024, 12, 31)
CYCLE = 5

dominant_symbols = [
    'RB88.SHFE', 'HC88.SHFE', 'I88.DCE', 'J88.DCE', 'JM88.DCE', 
    'BU88.SHFE', 'RU88.SHFE', 'MA88.CZCE', 'TA88.CZCE', 'EG88.DCE',
    'PP88.DCE', 'L88.DCE', 'V88.DCE', 'SA88.CZCE', 'FG88.CZCE',
    'SM88.CZCE', 'SF88.CZCE', 'SR88.CZCE', 'CF88.CZCE', 'OI88.CZCE',
    'RM88.CZCE', 'M88.DCE', 'Y88.DCE', 'P88.DCE', 'A88.DCE',
    'JD88.DCE', 'AP88.CZCE', 'CJ88.CZCE', 'CY88.CZCE', 'UR88.CZCE',
    'PF88.CZCE', 'PG88.DCE', 'EB88.DCE', 'LU88.INE', 'SP88.SHFE',
    'AL88.SHFE', 'CU88.SHFE', 'ZN88.SHFE', 'NI88.SHFE', 'SN88.SHFE',
    'PB88.SHFE', 'AG88.SHFE', 'AU88.SHFE', 'SS88.SHFE', 'FU88.SHFE',
    'SC88.INE', 'NR88.SHFE', 'BC88.SHFE', 'IC88.CFFEX', 'IF88.CFFEX', 'IH88.CFFEX'
]

print("=" * 70)
print("计算 carry_ret_exact（旧版严谨定义）")
print("=" * 70)

# 1. 加载 DominantManager 映射
dc = DataCenter()
keys = [f"{ds}@1" for ds in dominant_symbols] + [f"{ds}@2" for ds in dominant_symbols]
dom = dc.load_reference_df("vnpy_dominant_contract", keys, START, END, fillna=True)

# 2. 加载合约到期日信息
contract_df = dc.load_contract_df()
# 转换 enddate 为 datetime
contract_df['enddate'] = pd.to_datetime(contract_df['enddate'], errors='coerce')

# 3. 加载 88 和 88A2 的收盘价
f1_data = {}
f2_data = {}
for ds in dominant_symbols:
    sym_88 = ds.replace('88.', '88.')
    sym_88a2 = ds.replace('88.', '88A2.')
    try:
        df1 = load_bar_df(sym_88, Interval.DAILY, START, END)
        if len(df1) > 0 and 'close_price' in df1.columns:
            f1_data[ds] = df1['close_price']
    except:
        pass
    try:
        df2 = load_bar_df(sym_88a2, Interval.DAILY, START, END)
        if len(df2) > 0 and 'close_price' in df2.columns:
            f2_data[ds] = df2['close_price']
    except:
        pass

f1_df = pd.DataFrame(f1_data)
f2_df = pd.DataFrame(f2_data)

print(f"F1 (88) 数据: {f1_df.shape}")
print(f"F2 (88A2) 数据: {f2_df.shape}")

# 4. 计算 carry_ret_exact
results = []
common_dates = f1_df.index.intersection(f2_df.index)

for date in common_dates:
    row_carry = {}
    for ds in dominant_symbols:
        if ds not in f1_df.columns or ds not in f2_df.columns:
            continue
        
        # 获取当日映射的真实合约
        col1 = f"{ds}@1"
        col2 = f"{ds}@2"
        if col1 not in dom.columns or col2 not in dom.columns:
            continue
        
        c1 = dom.loc[date, col1] if date in dom.index else None
        c2 = dom.loc[date, col2] if date in dom.index else None
        
        if pd.isna(c1) or pd.isna(c2):
            continue
        
        # 去掉交易所后缀，查询到期日
        c1_base = c1.split('.')[0] if '.' in c1 else c1
        c2_base = c2.split('.')[0] if '.' in c2 else c2
        
        if c1_base not in contract_df.index or c2_base not in contract_df.index:
            continue
        
        expiry1 = contract_df.loc[c1_base, 'enddate']
        expiry2 = contract_df.loc[c2_base, 'enddate']
        
        if pd.isna(expiry1) or pd.isna(expiry2):
            continue
        
        delta_days = (expiry2 - expiry1).days
        if delta_days <= 0:
            continue
        
        # 取 cycle 天均值
        f1_series = f1_df[ds]
        f2_series = f2_df[ds]
        
        idx = f1_series.index.get_loc(date)
        if idx < CYCLE - 1:
            continue
        
        f1_mean = f1_series.iloc[idx - CYCLE + 1:idx + 1].mean()
        f2_mean = f2_series.iloc[idx - CYCLE + 1:idx + 1].mean()
        
        if f1_mean <= 0 or f2_mean <= 0:
            continue
        
        carry = np.log(f1_mean / f2_mean) / delta_days * 365.0
        row_carry[ds] = carry
    
    results.append(row_carry)

carry_exact_df = pd.DataFrame(results, index=common_dates)
carry_exact_df.index.name = 'datetime'

print(f"\ncarry_ret_exact 因子矩阵: {carry_exact_df.shape}")
print(f"  样本数: {carry_exact_df.count().sum()} 个非空值")
print(f"  均值: {carry_exact_df.mean().mean():.4f}")
print(f"  标准差: {carry_exact_df.std().mean():.4f}")

# 保存到 DataCenter
print("\n保存因子到 DataCenter...")
for vt_symbol in carry_exact_df.columns:
    series = carry_exact_df[vt_symbol].dropna()
    if len(series) == 0:
        continue
    factor_id = dc.get_factor_id(
        "carry_ret_exact", vt_symbol, "", "d", "cs_developer"
    )
    key = str(factor_id)
    dc.save_reference_series("vnpy_factor_data", key, series)

print("因子已保存")
carry_exact_df.to_csv("factor_carry_ret_exact.csv")
print("同时保存到 factor_carry_ret_exact.csv")
