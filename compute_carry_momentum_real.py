#!/usr/bin/env python3
"""
carry_momentum 真实合约版本 — 用 DominantManager 映射的具体合约计算

逻辑：
1. 从 @1/@2 映射获取每个交易日的主力/次主力合约
2. 加载每个具体合约的日线 close
3. 对于每个交易日 t：
   - 主力合约 = @1[t]，计算其 20 日收益 = close[t] / close[t-20] - 1
   - 次主力合约 = @2[t]，计算其 20 日收益
   - carry_momentum_real[t] = 主力收益 - 次主力收益
4. 保存因子到 DataCenter

优势：完全避开连续合约跳价，使用真实可交易合约价格
"""

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
sys.path.insert(0, "/root/cs_developer/factor_system")

from factor_system.factor_engine import FactorEngine

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
CYCLE = 20


def compute_real_carry_momentum(symbol: str, dom_df: pd.DataFrame) -> pd.Series:
    """为单个品种计算真实合约 carry_momentum"""
    
    col1 = f"{symbol}@1"
    col2 = f"{symbol}@2"
    
    if col1 not in dom_df.columns or col2 not in dom_df.columns:
        return pd.Series(dtype=float)
    
    s1 = dom_df[col1].dropna()
    s2 = dom_df[col2].dropna()
    
    # 收集所有合约
    all_contracts = set(s1.unique()) | set(s2.unique())
    
    # 加载每个合约的 close
    contract_closes: Dict[str, pd.Series] = {}
    for contract in all_contracts:
        try:
            df = load_bar_df(contract, Interval.DAILY, START, END)
            if len(df) > 0 and 'close_price' in df.columns:
                contract_closes[contract] = df['close_price']
        except Exception:
            pass
    
    # 对于每个交易日，找到主力/次主力合约，计算 20 日收益
    common_dates = s1.index.intersection(s2.index)
    results = []
    
    for date in common_dates:
        c1 = s1.loc[date]  # 主力
        c2 = s2.loc[date]  # 次主力
        
        ret1 = np.nan
        ret2 = np.nan
        
        if c1 in contract_closes:
            cs = contract_closes[c1]
            # 找 date 和 date-20 的价格
            t0 = cs.asof(date)
            t_20 = cs.asof(date - pd.Timedelta(days=30))  # 放宽到 30 天，避免假期
            # 确保实际间隔约 20 个交易日
            if pd.notna(t0) and pd.notna(t_20) and t_20 > 0:
                # 检查实际交易日间隔
                dates_in_range = cs.index[(cs.index <= date) & (cs.index >= date - pd.Timedelta(days=40))]
                if len(dates_in_range) >= CYCLE:
                    ret1 = t0 / t_20 - 1.0
        
        if c2 in contract_closes:
            cs = contract_closes[c2]
            t0 = cs.asof(date)
            t_20 = cs.asof(date - pd.Timedelta(days=30))
            if pd.notna(t0) and pd.notna(t_20) and t_20 > 0:
                dates_in_range = cs.index[(cs.index <= date) & (cs.index >= date - pd.Timedelta(days=40))]
                if len(dates_in_range) >= CYCLE:
                    ret2 = t0 / t_20 - 1.0
        
        if pd.notna(ret1) and pd.notna(ret2):
            results.append((date, ret1 - ret2))
    
    if len(results) == 0:
        return pd.Series(dtype=float)
    
    df_result = pd.DataFrame(results, columns=['datetime', 'value'])
    df_result.set_index('datetime', inplace=True)
    return df_result['value'].sort_index()


def main():
    print("=" * 70)
    print("carry_momentum 真实合约版本计算")
    print("=" * 70)
    
    dc = DataCenter()
    
    # 加载所有 @1/@2 映射
    print("\n[1/3] 加载 DominantManager 映射...")
    keys = []
    for s in DOMINANT_SYMBOLS:
        keys.append(f"{s}@1")
        keys.append(f"{s}@2")
    
    dom_df = dc.load_reference_df("vnpy_dominant_contract", keys, START, END, fillna=True)
    print(f"      映射记录: {dom_df.shape}")
    
    # 逐个品种计算
    print("\n[2/3] 计算真实合约 carry_momentum...")
    all_results = {}
    
    for i, symbol in enumerate(DOMINANT_SYMBOLS):
        try:
            s = compute_real_carry_momentum(symbol, dom_df)
            if len(s) > 0:
                all_results[symbol] = s
                print(f"  [{i+1:2d}/{len(DOMINANT_SYMBOLS)}] {symbol:15s} | {len(s):4d} days | mean={s.mean():+.4f} | std={s.std():.4f}")
            else:
                print(f"  [{i+1:2d}/{len(DOMINANT_SYMBOLS)}] {symbol:15s} | [SKIP] no data")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(DOMINANT_SYMBOLS)}] {symbol:15s} | [FAIL] {e}")
    
    print(f"\n      成功: {len(all_results)} / {len(DOMINANT_SYMBOLS)}")
    
    # 构建 wide DataFrame
    if len(all_results) == 0:
        print("[ERROR] 无有效数据")
        return
    
    print("\n[3/3] 保存到 DataCenter...")
    df_wide = pd.DataFrame(all_results)
    df_wide.sort_index(inplace=True)
    
    out_path = DATA_DIR / "carry_momentum_real.parquet"
    df_wide.to_parquet(out_path)
    print(f"      保存: {out_path} | shape={df_wide.shape}")
    
    # 保存到 DataCenter
    engine = FactorEngine(
        dominant_symbols=list(all_results.keys()),
        start=START,
        end=END,
        verbose=False
    )
    engine.save_factor(df_wide, name="carry_momentum_real", parameter="cycle20", author="factor_system")
    print("      已保存到 DataCenter")
    
    # 统计
    print("\n[统计] 真实合约版 vs 889 版对比")
    try:
        df_889 = pd.read_parquet(DATA_DIR / "carry_momentum_889_2020-01-01_2024-12-31.parquet")
    except:
        df_889 = None
    
    if df_889 is not None:
        common_cols = [c for c in df_wide.columns if c in df_889.columns]
        for symbol in common_cols[:5]:
            s_real = df_wide[symbol].dropna()
            s_889 = df_889[symbol].dropna()
            common_idx = s_real.index.intersection(s_889.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(s_real.loc[common_idx], s_889.loc[common_idx])[0, 1]
                print(f"  {symbol:15s} real-vs-889 corr={corr:.3f}")
    
    print("\n[OK] 完成!")


if __name__ == "__main__":
    main()
