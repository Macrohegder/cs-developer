#!/usr/bin/env python3
"""
手动构建 889A2 后复权连续合约数据

逻辑：
1. 从 DominantManager 获取每个品种每天的次主力合约映射 (@2)
2. 加载每个具体次主力合约的日线 close
3. 在合约切换点计算复权因子：adj = 新合约首日 close / 旧合约末日 close
4. 从后向前累积复权因子，调整历史价格
5. 输出 wide-format DataFrame (index=datetime, columns=dominant_symbols)

输出:
    factor_system/data/secondary_889a2_close.parquet
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import traceback

import pandas as pd
import numpy as np
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.utility import load_bar_df

sys.path.insert(0, "/root/cs_developer")

DATA_DIR = Path("/root/cs_developer/factor_system/data")

DOMINANT_SYMBOLS = [
    'A88.DCE', 'AG88.SHFE', 'AL88.SHFE', 'AP88.CZCE', 'AU88.SHFE',
    'B88.DCE', 'BU88.SHFE', 'C88.DCE', 'CF88.CZCE', 'CJ88.CZCE',
    'CS88.DCE', 'CU88.SHFE', 'CY88.CZCE', 'EB88.DCE', 'EG88.DCE',
    'FG88.CZCE', 'FU88.SHFE', 'HC88.SHFE', 'I88.DCE', 'J88.DCE',
    'JD88.DCE', 'JM88.DCE', 'L88.DCE', 'LH88.DCE', 'LU88.INE',
    'M88.DCE', 'MA88.CZCE', 'NI88.SHFE', 'OI88.CZCE', 'P88.DCE',
    'PB88.SHFE', 'PF88.CZCE', 'PG88.DCE', 'PK88.CZCE', 'PP88.DCE',
    'RB88.SHFE', 'RM88.CZCE', 'RU88.SHFE', 'SA88.CZCE', 'SC88.INE',
    'SF88.CZCE', 'SI88.GFEX', 'SM88.CZCE', 'SN88.SHFE', 'SP88.SHFE',
    'SR88.CZCE', 'SS88.SHFE', 'UR88.CZCE', 'V88.DCE', 'Y88.DCE',
    'ZN88.SHFE'
]

START = datetime(2020, 1, 1)
END = datetime(2024, 12, 31)


def build_889a2_for_symbol(symbol: str, dom_df: pd.DataFrame) -> pd.Series:
    """为单个品种构建 889A2 后复权价格序列"""
    
    col = f"{symbol}@2"
    if col not in dom_df.columns:
        return pd.Series(dtype=float)
    
    contract_series = dom_df[col].dropna()
    if len(contract_series) == 0:
        return pd.Series(dtype=float)
    
    # 找出合约切换点
    rolls = contract_series[contract_series != contract_series.shift(1)].copy()
    if len(rolls) <= 1:
        # 只有一合约，无需复权
        return pd.Series(dtype=float)
    
    # 收集所有合约
    all_contracts = rolls.unique()
    
    # 加载每个合约的 close 数据（缓存）
    contract_closes: Dict[str, pd.Series] = {}
    for c in all_contracts:
        try:
            df = load_bar_df(c, Interval.DAILY, START, END)
            if len(df) > 0 and 'close_price' in df.columns:
                contract_closes[c] = df['close_price']
        except Exception:
            pass
    
    # 构建 88A2 原始价格序列（直接拼接合约 close）
    raw_prices = []
    for date, contract in contract_series.items():
        if contract in contract_closes:
            close_on_date = contract_closes[contract].asof(date)
            if pd.notna(close_on_date):
                raw_prices.append((date, close_on_date, contract))
    
    if len(raw_prices) == 0:
        return pd.Series(dtype=float)
    
    raw_df = pd.DataFrame(raw_prices, columns=['datetime', 'close', 'contract'])
    raw_df.set_index('datetime', inplace=True)
    raw_df.sort_index(inplace=True)
    
    # 后复权：从最新日期开始，cum_adj = 1.0
    # 向后遍历，遇到切换点时 cum_adj *= (新合约首日 / 旧合约末日)
    # 注意：rolls 的第一个点是整个序列的起点，不需要复权
    
    cum_adj = 1.0
    adj_factors = {raw_df.index[-1]: cum_adj}  # 最新日期 adj = 1.0
    
    # 从倒数第二个切换点开始，向后（历史方向）遍历
    roll_dates = rolls.index.tolist()
    
    for i in range(len(roll_dates) - 1, 0, -1):
        new_contract_start = roll_dates[i]
        old_contract_start = roll_dates[i - 1]
        
        old_contract = rolls.loc[old_contract_start]
        new_contract = rolls.loc[new_contract_start]
        
        if old_contract not in contract_closes or new_contract not in contract_closes:
            continue
        
        # 旧合约的最后一天：切换日前一天
        old_close_series = contract_closes[old_contract]
        # 找到切换日之前的最后一个交易日
        old_dates = old_close_series.index[old_close_series.index < new_contract_start]
        if len(old_dates) == 0:
            continue
        old_last_date = old_dates[-1]
        old_last_close = old_close_series.loc[old_last_date]
        
        # 新合约的第一天：切换日当天（raw_df 中已有）
        new_first_close = raw_df.loc[new_contract_start, 'close']
        
        if pd.isna(old_last_close) or pd.isna(new_first_close) or old_last_close == 0:
            continue
        
        adj = new_first_close / old_last_close
        cum_adj *= adj
        
        # 从 old_contract_start 到 new_contract_start 之前的所有日期使用这个 cum_adj
        adj_factors[old_contract_start] = cum_adj
    
    # 应用复权因子
    # 对于每个日期，找到对应的 cum_adj
    result = raw_df['close'].copy()
    sorted_adj_dates = sorted(adj_factors.keys())
    
    for i in range(len(sorted_adj_dates)):
        adj_date = sorted_adj_dates[i]
        adj_val = adj_factors[adj_date]
        
        if i + 1 < len(sorted_adj_dates):
            next_date = sorted_adj_dates[i + 1]
            mask = (result.index >= adj_date) & (result.index < next_date)
        else:
            mask = result.index >= adj_date
        
        result.loc[mask] *= adj_val
    
    return result


def main():
    print("=" * 70)
    print("构建 889A2 后复权数据")
    print("=" * 70)
    
    dc = DataCenter()
    
    # 加载所有 @2 映射
    print("\n[1/3] 加载次主力合约映射...")
    keys = [f"{s}@2" for s in DOMINANT_SYMBOLS]
    dom_df = dc.load_reference_df("vnpy_dominant_contract", keys, START, END, fillna=True)
    print(f"      映射记录: {dom_df.shape}")
    
    # 逐个品种构建 889A2
    print("\n[2/3] 构建 889A2 后复权价格序列...")
    all_889a2 = {}
    success = 0
    failed = 0
    
    for symbol in DOMINANT_SYMBOLS:
        try:
            s = build_889a2_for_symbol(symbol, dom_df)
            if len(s) > 0:
                all_889a2[symbol] = s
                success += 1
                print(f"  [OK] {symbol:15s} | {len(s):4d} days")
            else:
                failed += 1
                print(f"  [SKIP] {symbol:15s} | no data")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {symbol:15s} | {e}")
    
    print(f"\n      成功: {success} | 失败: {failed}")
    
    # 合并为 wide DataFrame
    if len(all_889a2) == 0:
        print("[ERROR] 无有效数据")
        return
    
    print("\n[3/3] 保存 wide-format DataFrame...")
    df_wide = pd.DataFrame(all_889a2)
    df_wide.sort_index(inplace=True)
    
    # 同时保存 close 和计算 returns（用于后续因子计算）
    out_path = DATA_DIR / "secondary_889a2_close.parquet"
    df_wide.to_parquet(out_path)
    print(f"      保存: {out_path} | shape={df_wide.shape}")
    
    # 验证：检查大跳价频率
    print("\n[验证] 889A2 大跳价频率...")
    for symbol in list(all_889a2.keys())[:5]:
        s = all_889a2[symbol]
        ret = s.pct_change().abs()
        big = ret[ret > 0.03]
        print(f"  {symbol:15s} | big jumps: {len(big):2d}/{len(s)} ({len(big)/len(s)*100:.1f}%)")
    
    print("\n[OK] 889A2 构建完成!")


if __name__ == "__main__":
    main()
