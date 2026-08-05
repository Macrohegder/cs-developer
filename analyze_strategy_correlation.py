#!/usr/bin/env python3
"""
三品种日历价差策略相关性分析

使用各品种最优参数的单策略日收益序列：
- IF @ 25% vol_threshold
- IC @ 15% vol_threshold
- IM @ 15% vol_threshold

分析：
- 日收益相关性
- 日盈亏相关性
- 同涨同跌比例
- 回撤重叠情况
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path("/root/quant/cs_developer/docs/ic_im_vol_scan")
OUTPUT_DIR = BASE_DIR


def load_pnl(product: str, filename: str) -> pd.Series:
    df = pd.read_csv(BASE_DIR / filename, index_col=0, parse_dates=True)
    return df["net_pnl"]


def main():
    print("="*70)
    print("三品种日历价差策略相关性分析")
    print("="*70)

    # 加载各品种日盈亏
    pnl_series = {
        "IF": load_pnl("IF", "vt25_if_pnl.csv"),
        "IC": load_pnl("IC", "optimal_ic_pnl.csv"),
        "IM": load_pnl("IM", "optimal_im_pnl.csv"),
    }

    # 对齐日期
    common_index = pnl_series["IF"].index.intersection(pnl_series["IC"].index).intersection(pnl_series["IM"].index)
    print(f"\n共同交易日: {len(common_index)} 天")

    pnl_df = pd.DataFrame({p: s.reindex(common_index).fillna(0) for p, s in pnl_series.items()})

    # 计算日收益（基于 1000 万本金）
    capital = 10_000_000
    ret_df = pnl_df / capital

    # 1. 日盈亏相关性
    pnl_corr = pnl_df.corr()
    print("\n【日盈亏相关性】")
    print(pnl_corr.round(4).to_string())

    # 2. 日收益率相关性
    ret_corr = ret_df.corr()
    print("\n【日收益率相关性】")
    print(ret_corr.round(4).to_string())

    # 3. 同涨同跌分析
    print("\n【同涨同跌统计】")
    pairs = [("IF", "IC"), ("IF", "IM"), ("IC", "IM")]
    for p1, p2 in pairs:
        s1 = pnl_df[p1]
        s2 = pnl_df[p2]
        same_direction = ((s1 > 0) & (s2 > 0)) | ((s1 < 0) & (s2 < 0))
        both_nonzero = (s1 != 0) & (s2 != 0)
        overlap_days = both_nonzero.sum()
        same_dir_days = (same_direction & both_nonzero).sum()
        print(f"  {p1}-{p2}: 同时持仓天数 {overlap_days}, 同向天数 {same_dir_days}, 同向比例 {same_dir_days/overlap_days:.2%}" if overlap_days > 0 else f"  {p1}-{p2}: 无同时持仓")

    # 4. 回撤重叠分析
    print("\n【回撤重叠分析】")
    # 计算每个品种的回撤标记（当日回撤 > 0.1%）
    dd_flags = {}
    for p in ["IF", "IC", "IM"]:
        pnl = pnl_df[p]
        balance = capital + pnl.cumsum()
        running_max = balance.cummax()
        dd = (balance - running_max) / running_max
        dd_flags[p] = dd < -0.001  # 回撤超过 0.1%

    dd_df = pd.DataFrame(dd_flags)
    for p1, p2 in pairs:
        both_dd = dd_df[p1] & dd_df[p2]
        either_dd = dd_df[p1] | dd_df[p2]
        print(f"  {p1}-{p2}: 共同回撤天数 {both_dd.sum()}, 任一回撤天数 {either_dd.sum()}, 重叠比例 {both_dd.sum()/either_dd.sum():.2%}" if either_dd.sum() > 0 else f"  {p1}-{p2}: 无回撤")

    # 5. 月度收益相关性
    monthly_pnl = pnl_df.resample("ME").sum()
    monthly_corr = monthly_pnl.corr()
    print("\n【月度盈亏相关性】")
    print(monthly_corr.round(4).to_string())

    # 6. 滚动 60 日相关性
    rolling_corr = {}
    for p1, p2 in pairs:
        rolling_corr[f"{p1}-{p2}"] = ret_df[p1].rolling(60).corr(ret_df[p2])

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 相关性热力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pnl_corr, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=axes[0], square=True)
    axes[0].set_title("Daily PnL Correlation")

    sns.heatmap(monthly_corr, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=axes[1], square=True)
    axes[1].set_title("Monthly PnL Correlation")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_correlation_heatmap.png", dpi=150)
    plt.close()

    # 滚动相关性图
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, series in rolling_corr.items():
        ax.plot(series.index, series, label=label)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Rolling 60-Day Return Correlation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_rolling_correlation.png", dpi=150)
    plt.close()

    # 保存 CSV
    pnl_corr.to_csv(OUTPUT_DIR / "strategy_pnl_correlation.csv")
    ret_corr.to_csv(OUTPUT_DIR / "strategy_return_correlation.csv")
    monthly_corr.to_csv(OUTPUT_DIR / "strategy_monthly_correlation.csv")

    print("\n" + "="*70)
    print("分析完成")
    print(f"结果保存到: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
