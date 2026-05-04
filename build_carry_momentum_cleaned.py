#!/usr/bin/env python3
"""
构建 cleaned carry_momentum：889 - 88A2_cleaned

修复：
- 889 负价格过滤（CJ/FU/RU 用 88 替代）
- 88A2 跳价清洗（线性插值）

输出:
    factor_system/data/carry_momentum_cleaned.parquet
    docs/carry_momentum_comparison.png
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.utility import load_bar_df

sys.path.insert(0, "/root/cs_developer")

DATA_DIR = Path("/root/cs_developer/factor_system/data")
DOCS_DIR = Path("/root/cs_developer/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

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

# 889 负价格黑名单：这些品种用 88 替代 889
NEGATIVE_PRICE_BLACKLIST = {"CJ88.CZCE", "FU88.SHFE", "RU88.SHFE"}

START = datetime(2020, 1, 1)
END = datetime(2024, 12, 31)
CYCLE = 20
JUMP_THRESHOLD = 0.03


def clean_jumps(s: pd.Series, threshold: float = 0.03) -> pd.Series:
    """对跳价日做线性插值清洗"""
    s_clean = s.copy()
    daily_ret = s.pct_change().abs()
    jumps = daily_ret > threshold
    jump_dates = s.index[jumps]
    
    for d in jump_dates:
        idx = s.index.get_loc(d)
        prev_idx = idx - 1
        next_idx = idx + 1
        
        while prev_idx >= 0 and jumps.iloc[prev_idx]:
            prev_idx -= 1
        while next_idx < len(s) and jumps.iloc[next_idx]:
            next_idx += 1
        
        if prev_idx >= 0 and next_idx < len(s):
            prev_val = s.iloc[prev_idx]
            next_val = s.iloc[next_idx]
            prev_date = s.index[prev_idx]
            next_date = s.index[next_idx]
            
            if next_date != prev_date and pd.notna(prev_val) and pd.notna(next_val):
                weight = (d - prev_date).days / (next_date - prev_date).days
                s_clean.loc[d] = prev_val + weight * (next_val - prev_val)
    
    return s_clean


def main():
    print("=" * 70)
    print("构建 carry_momentum_cleaned (889_cleaned - 88A2_cleaned)")
    print("=" * 70)
    
    # =====================================================================
    # 1. 加载数据
    # =====================================================================
    print("\n[1/3] 加载价格数据...")
    
    f1_data = {}      # 889 (或 88 对于黑名单品种)
    f2_data = {}      # 88A2 raw
    f2_cleaned_data = {}
    
    for symbol in DOMINANT_SYMBOLS:
        base = symbol.split(".")[0].replace("88", "")
        exchange = symbol.split(".")[1]
        
        try:
            # F1: 889 (黑名单用 88)
            if symbol in NEGATIVE_PRICE_BLACKLIST:
                f1 = load_bar_df(f"{base}88.{exchange}", Interval.DAILY, START, END)["close_price"]
                src = "88"
            else:
                f1 = load_bar_df(f"{base}889.{exchange}", Interval.DAILY, START, END)["close_price"]
                src = "889"
                # 过滤负价格和零价格
                f1 = f1.where(f1 > 0)
            
            # F2: 88A2
            f2 = load_bar_df(f"{base}88A2.{exchange}", Interval.DAILY, START, END)["close_price"]
            
            if len(f1) > 0 and len(f2) > 0:
                f1_data[symbol] = f1
                f2_data[symbol] = f2
                f2_clean = clean_jumps(f2, JUMP_THRESHOLD)
                f2_cleaned_data[symbol] = f2_clean
                
        except Exception as e:
            pass
    
    print(f"      成功加载: {len(f1_data)} 个品种")
    blacklist_used = [s for s in f1_data if s in NEGATIVE_PRICE_BLACKLIST]
    if blacklist_used:
        print(f"      黑名单用 88 替代: {', '.join(blacklist_used)}")
    
    # =====================================================================
    # 2. 计算 carry_momentum
    # =====================================================================
    print("\n[2/3] 计算 carry_momentum...")
    
    cm_88_raw = {}
    cm_889_raw = {}
    cm_889_clean = {}
    
    for symbol in f1_data.keys():
        f1 = f1_data[symbol]
        f2 = f2_data[symbol]
        f2_clean = f2_cleaned_data[symbol]
        
        common_idx = f1.index.intersection(f2.index)
        f1_c = f1.reindex(common_idx)
        f2_c = f2.reindex(common_idx)
        f2_clean_c = f2_clean.reindex(common_idx)
        
        f1_ret = f1_c / f1_c.shift(CYCLE) - 1.0
        f2_ret = f2_c / f2_c.shift(CYCLE) - 1.0
        f2_clean_ret = f2_clean_c / f2_clean_c.shift(CYCLE) - 1.0
        
        cm_889_raw[symbol] = f1_ret - f2_ret
        cm_889_clean[symbol] = f1_ret - f2_clean_ret
    
    # 构建统一索引的 DataFrame
    all_indices = [s.index for s in cm_889_raw.values()] + [s.index for s in cm_889_clean.values()]
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)
    common_index = common_index.sort_values()
    
    df_889 = pd.DataFrame({k: v.reindex(common_index) for k, v in cm_889_raw.items()})
    df_889_clean = pd.DataFrame({k: v.reindex(common_index) for k, v in cm_889_clean.items()})
    
    # 保存清洗版本
    out_path = DATA_DIR / "carry_momentum_cleaned.parquet"
    df_889_clean.to_parquet(out_path)
    print(f"      保存: {out_path} | shape={df_889_clean.shape}")
    
    # =====================================================================
    # 3. 统计对比
    # =====================================================================
    print("\n[3/3] 统计对比...")
    
    stats = []
    for name, df in [("889-88A2", df_889), ("889-88A2_cleaned", df_889_clean)]:
        vol = df.std().mean()
        abs_mean = df.abs().mean().mean()
        big_jumps = (df.abs() > 0.03).sum().sum()
        total = df.notna().sum().sum()
        stats.append({
            "version": name,
            "mean_vol": vol,
            "mean_abs": abs_mean,
            "big_jumps": big_jumps,
            "total_obs": total,
            "big_jump_rate": big_jumps / total if total > 0 else 0,
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # 检查极端值
    print(f"\n极端值检查 (cleaned):")
    print(f"  Max: {df_889_clean.max().max():.4f}")
    print(f"  Min: {df_889_clean.min().min():.4f}")
    
    # 找到最高波动率的品种
    vols = df_889_clean.std().sort_values(ascending=False)
    print(f"\nTop 5 波动率:")
    for sym, v in vols.head(5).items():
        print(f"  {sym:15s}: {v:.4f}")
    
    print("\n[OK] 因子构建完成!")


if __name__ == "__main__":
    main()
