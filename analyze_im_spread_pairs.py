#!/usr/bin/env python3
"""
IM 股指期货跨月套利对价差分析
- 画出每一对连续季月合约（如 IM2209-IM2212）的价差走势
- 输出价差统计指标
- 所有图表保存到 docs/im_spread_analysis/
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

import pytz
_original_timezone = pytz.timezone

def _patched_timezone(zone):
    if zone == "Asia/Beijing":
        return _original_timezone("Asia/Shanghai")
    return _original_timezone(zone)

pytz.timezone = _patched_timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "factor_system"))

from vnpy.trader.constant import Interval
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df, load_history_df

OUTPUT_DIR = Path(__file__).parent / "docs" / "im_spread_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = datetime(2022, 7, 22)


def get_im_contracts(contract_df: pd.DataFrame) -> pd.DataFrame:
    """筛选 IM 股指期货合约，并返回带上市/到期日的 DataFrame"""
    mask = (
        (contract_df['exchange'] == 'CFFEX')
        & (contract_df.index.astype(str).str.startswith('IM'))
        & (contract_df['product'] == '期货')
    )
    df = contract_df[mask].copy()
    df.index = df.index.astype(str)
    df['listed_dt'] = pd.to_datetime(df['listed'], errors='coerce')
    df['expiry_dt'] = pd.to_datetime(df['expiry'], errors='coerce')
    # 只保留 6 位代码的标准合约，如 IM2609
    df = df[df.index.str.len() == 6]
    df = df[(df['listed_dt'].notna()) & (df['expiry_dt'].notna())]
    df['month'] = df.index.str[-2:].astype(int)
    # 季月合约
    df = df[df['month'].isin([3, 6, 9, 12])]
    return df.sort_index()


def build_quarter_pairs(im_info: pd.DataFrame) -> List[Tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """构建连续季月合约对，返回 (near_symbol, far_symbol, near_expiry, far_expiry)"""
    syms = im_info.index.tolist()
    pairs = []
    for i in range(len(syms) - 1):
        near = syms[i]
        far = syms[i + 1]
        near_exp = im_info.loc[near, 'expiry_dt']
        far_exp = im_info.loc[far, 'expiry_dt']
        if far_exp > near_exp:
            pairs.append((near, far, near_exp, far_exp))
    return pairs


def plot_pair(near: str, far: str, df: pd.DataFrame, output_path: Path):
    """绘制价差走势图并保存"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 1. 绝对价差
    ax1 = axes[0]
    ax1.plot(df.index, df['spread'], color='steelblue', linewidth=1.2, label=f'{near} - {far}')
    ax1.axhline(df['spread'].mean(), color='red', linestyle='--', linewidth=1, label=f'mean={df["spread"].mean():.2f}')
    ax1.axhline(df['spread'].mean() + df['spread'].std(), color='orange', linestyle=':', linewidth=1, label=f'+1σ')
    ax1.axhline(df['spread'].mean() - df['spread'].std(), color='orange', linestyle=':', linewidth=1, label=f'-1σ')
    ax1.set_title(f'IM 跨月套利价差走势：{near} vs {far}', fontsize=14)
    ax1.set_ylabel('绝对价差 (Near - Far)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. 百分比价差
    ax2 = axes[1]
    ax2.plot(df.index, df['spread_pct'] * 100, color='green', linewidth=1.2, label='spread_pct')
    ax2.axhline(df['spread_pct'].mean() * 100, color='red', linestyle='--', linewidth=1, label=f'mean={df["spread_pct"].mean()*100:.2f}%')
    ax2.axhline((df['spread_pct'].mean() + df['spread_pct'].std()) * 100, color='orange', linestyle=':', linewidth=1)
    ax2.axhline((df['spread_pct'].mean() - df['spread_pct'].std()) * 100, color='orange', linestyle=':', linewidth=1)
    ax2.set_title(f'IM 跨月套利百分比价差走势：{near} vs {far}', fontsize=14)
    ax2.set_ylabel('spread_pct (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("IM 跨月套利对价差分析")
    print("=" * 70)

    print("\n[1/3] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)
    im_info = get_im_contracts(contract_df)
    print(f"  共找到 {len(im_info)} 个 IM 季月合约")

    print("\n[2/3] 加载历史数据...")
    vt_symbols = [f"{s}.CFFEX" for s in im_info.index]
    history_df = load_history_df(vt_symbols, Interval.DAILY, START, datetime.today())
    close_df = history_df.xs("close_price", axis=1, level=1)
    print(f"  历史数据区间：{close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    pairs = build_quarter_pairs(im_info)
    print(f"  共构建 {len(pairs)} 个连续季月套利对")

    print("\n[3/3] 计算统计指标并绘图...")
    summary_records = []
    for near, far, near_exp, far_exp in pairs:
        near_vt = f"{near}.CFFEX"
        far_vt = f"{far}.CFFEX"
        if near_vt not in close_df.columns or far_vt not in close_df.columns:
            print(f"  跳过 {near}-{far}：缺少价格数据")
            continue

        near_listed = im_info.loc[near, 'listed_dt']
        far_listed = im_info.loc[far, 'listed_dt']
        start_dt = max(near_listed, far_listed, pd.Timestamp(START))
        end_dt = near_exp  # 近月到期即停止

        sub = close_df.loc[start_dt:end_dt].copy()
        sub['near'] = sub[near_vt]
        sub['far'] = sub[far_vt]
        sub = sub.dropna(subset=['near', 'far'])
        if len(sub) < 10:
            print(f"  跳过 {near}-{far}：有效数据不足 ({len(sub)} 天)")
            continue

        sub['spread'] = sub['near'] - sub['far']
        sub['spread_pct'] = sub['spread'] / sub['near']
        sub['spread_change'] = sub['spread'].diff()

        # 统计指标
        stats = {
            'pair': f"{near}-{far}",
            'near_symbol': near,
            'far_symbol': far,
            'start_date': sub.index[0].strftime('%Y-%m-%d'),
            'end_date': sub.index[-1].strftime('%Y-%m-%d'),
            'days': len(sub),
            'mean_spread': round(sub['spread'].mean(), 2),
            'std_spread': round(sub['spread'].std(), 2),
            'min_spread': round(sub['spread'].min(), 2),
            'max_spread': round(sub['spread'].max(), 2),
            'last_spread': round(sub['spread'].iloc[-1], 2),
            'mean_spread_pct': round(sub['spread_pct'].mean() * 100, 4),
            'std_spread_pct': round(sub['spread_pct'].std() * 100, 4),
            'min_spread_pct': round(sub['spread_pct'].min() * 100, 4),
            'max_spread_pct': round(sub['spread_pct'].max() * 100, 4),
            'last_spread_pct': round(sub['spread_pct'].iloc[-1] * 100, 4),
            'spread_vol_annual': round(sub['spread_change'].std() * np.sqrt(250), 2),
            'last_zscore': round((sub['spread'].iloc[-1] - sub['spread'].mean()) / sub['spread'].std(), 3) if sub['spread'].std() > 0 else 0,
        }
        summary_records.append(stats)

        # 绘图
        chart_path = OUTPUT_DIR / f"{near}_{far}_spread.png"
        plot_pair(near, far, sub, chart_path)
        print(f"  已生成：{chart_path.name} | 数据天数={stats['days']} | 最新 spread={stats['last_spread']:.2f} ({stats['last_spread_pct']:.2f}%) | z={stats['last_zscore']:.2f}")

    summary_df = pd.DataFrame(summary_records)
    summary_path = OUTPUT_DIR / "im_spread_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n统计摘要已保存：{summary_path}")
    print(f"图表保存目录：{OUTPUT_DIR}")
    print("\n汇总表：")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
