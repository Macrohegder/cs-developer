#!/usr/bin/env python3
"""
因子IC（Information Coefficient）分析模块

功能：
1. 加载因子数据（从 DataCenter）
2. 加载价格数据（88合约收盘价）
3. 计算多期 forward returns（1d, 5d, 10d, 20d）
4. 计算日度 IC（Spearman 秩相关系数）
5. 汇总统计：IC均值、标准差、IR、IC>0比例、t统计量
6. 分组收益分析（Quantile Returns）
7. 生成对比图表（matplotlib 保存 PNG）

支持多因子对比分析。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.utility import load_bar_df

# =============================================================================
# 配置
# =============================================================================
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

START = datetime(2011, 1, 1)
END = datetime(2026, 4, 30)

# 持有期（交易日）
HOLDING_PERIODS = [1, 5, 10, 20]

# 因子配置列表
FACTORS = [
    {
        "name": "spread_zscore",
        "parameter": "lookback20",
        "author": "futures_term_structure",
        "label": "Spread Z-Score (Mean Reversion)",
        "color": "steelblue"
    },
    {
        "name": "spread_return",
        "parameter": "lookback10",
        "author": "futures_term_structure",
        "label": "Spread Return (Momentum Reversal)",
        "color": "coral"
    }
]

OUTPUT_DIR = "/root/cs_developer"


# =============================================================================
# 数据加载
# =============================================================================
def load_factor(factor_config: dict) -> pd.DataFrame:
    """从 DataCenter 加载因子数据"""
    dc = DataCenter()
    df = dc.load_factor_df(
        vt_symbols=DOMINANT_SYMBOLS,
        name=factor_config["name"],
        interval="d",
        parameter=factor_config["parameter"],
        author=factor_config["author"],
        start=START,
        end=END
    )
    return df


def load_spread_data() -> pd.DataFrame:
    """
    加载所有品种的价差数据（F1=88主力连续 - F2=88A2次主力连续）
    返回 spread_df，列为 dominant_symbols，行为日期
    """
    print("加载价差数据（F1 - F2）...")
    f1_dict = {}
    f2_dict = {}

    for symbol in DOMINANT_SYMBOLS:
        f2_symbol = symbol.replace("88.", "88A2.")
        try:
            df1 = load_bar_df(symbol, Interval.DAILY, START, END)
            df2 = load_bar_df(f2_symbol, Interval.DAILY, START, END)
            if len(df1) > 0 and len(df2) > 0:
                f1_dict[symbol] = df1['close_price']
                f2_dict[symbol] = df2['close_price']
        except Exception as e:
            print(f"  跳过 {symbol}: {e}")

    f1_df = pd.DataFrame(f1_dict)
    f2_df = pd.DataFrame(f2_dict)

    # 对齐列
    common_cols = f1_df.columns.intersection(f2_df.columns)
    f1_df = f1_df[common_cols]
    f2_df = f2_df[common_cols]

    # 计算价差
    spread_df = f1_df - f2_df
    print(f"  价差数据加载完成: {spread_df.shape}")
    return spread_df


# =============================================================================
# Forward Returns 计算（基于价差）
# =============================================================================
def calc_forward_returns(spread_df: pd.DataFrame, periods: List[int]) -> Dict[int, pd.DataFrame]:
    """
    计算多期 forward returns（基于价差变化）
    return[t] = (spread[t+period] - spread[t]) / |spread[t]|
    """
    fwd_returns = {}
    for p in periods:
        # 价差变化率
        fwd_returns[p] = (spread_df.shift(-p) - spread_df) / spread_df.abs()
    return fwd_returns


# =============================================================================
# IC 计算
# =============================================================================
def calc_daily_ic(factor_df: pd.DataFrame, fwd_return_df: pd.DataFrame) -> pd.Series:
    """
    计算日度 IC（Spearman 秩相关系数）
    每天 cross-sectional 计算因子值与未来收益率的秩相关
    """
    ic_series = pd.Series(index=factor_df.index, dtype=float)

    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_return_df.loc[dt]

        # 对齐并去除NaN
        valid = f.notna() & r.notna()
        if valid.sum() < 5:  # 至少需要5个样本
            ic_series[dt] = np.nan
            continue

        f_valid = f[valid]
        r_valid = r[valid]

        # Spearman 秩相关系数
        corr, _ = stats.spearmanr(f_valid, r_valid)
        ic_series[dt] = corr

    return ic_series


def calc_ic_stats(ic_series: pd.Series) -> dict:
    """计算 IC 汇总统计量"""
    ic_clean = ic_series.dropna()
    if len(ic_clean) == 0:
        return {}

    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    ic_positive_ratio = (ic_clean > 0).sum() / len(ic_clean)

    # t统计量（检验IC均值是否显著不为0）
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_clean))) if std_ic > 0 else 0

    # 年化化IR（按252交易日）
    ir_annual = mean_ic * np.sqrt(252) / std_ic if std_ic > 0 else 0

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "ir_annual": ir_annual,
        "ic_positive_ratio": ic_positive_ratio,
        "t_stat": t_stat,
        "ic_count": len(ic_clean),
    }


# =============================================================================
# 分组收益分析（Quantile Analysis）
# =============================================================================
def calc_quantile_returns(factor_df: pd.DataFrame, fwd_return_df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    计算分组收益：每天按因子分n组，计算每组平均收益，再时序平均
    """
    daily_quantile_returns = []

    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_return_df.loc[dt]

        valid = f.notna() & r.notna()
        if valid.sum() < n_quantiles * 3:  # 每组至少3个
            continue

        f_valid = f[valid]
        r_valid = r[valid]

        try:
            labels = [f"Q{i+1}" for i in range(n_quantiles)]
            q_labels = pd.qcut(f_valid, n_quantiles, labels=labels, duplicates='drop')
            if q_labels.isna().sum() > 0:
                continue
            group_ret = r_valid.groupby(q_labels).mean()
            daily_quantile_returns.append(group_ret)
        except Exception:
            continue

    if not daily_quantile_returns:
        return pd.DataFrame()

    quantile_df = pd.DataFrame(daily_quantile_returns)
    return quantile_df.mean()


# =============================================================================
# 图表生成
# =============================================================================
def plot_ic_timeseries(ic_results: Dict[str, pd.Series], holding_period: int, output_path: str):
    """IC 时间序列图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    for factor_label, ic_series in ic_results.items():
        ax1.plot(ic_series.index, ic_series.values, label=factor_label, alpha=0.7, linewidth=0.8)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Daily IC')
    ax1.set_title(f'Daily IC Time Series ({holding_period}D Forward Return)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 累计IC
    ax2 = axes[1]
    for factor_label, ic_series in ic_results.items():
        cum_ic = ic_series.fillna(0).cumsum()
        ax2.plot(cum_ic.index, cum_ic.values, label=factor_label, linewidth=1.2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('Cumulative IC')
    ax2.set_xlabel('Date')
    ax2.set_title('Cumulative IC')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  IC时间序列图已保存: {output_path}")


def plot_ic_distribution(ic_results: Dict[str, pd.Series], holding_period: int, output_path: str):
    """IC 分布直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['steelblue', 'coral']
    for idx, (factor_label, ic_series) in enumerate(ic_results.items()):
        ic_clean = ic_series.dropna()
        ax.hist(ic_clean, bins=50, alpha=0.5, label=factor_label, color=colors[idx % len(colors)], density=True)

        # 标注均值线
        mean_ic = ic_clean.mean()
        ax.axvline(mean_ic, color=colors[idx % len(colors)], linestyle='--', linewidth=1.5,
                   label=f'{factor_label} Mean={mean_ic:.3f}')

    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('IC')
    ax.set_ylabel('Density')
    ax.set_title(f'IC Distribution ({holding_period}D Forward Return)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  IC分布图已保存: {output_path}")


def plot_ic_decay(ic_stats_all: dict, output_path: str):
    """IC 衰减图（不同持有期下的IC和IR）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IC均值衰减
    ax1 = axes[0]
    for factor_label, stats_dict in ic_stats_all.items():
        periods = sorted(stats_dict.keys())
        ic_means = [stats_dict[p]["mean_ic"] for p in periods]
        ax1.plot(periods, ic_means, marker='o', label=factor_label, linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Forward Return Period (Days)')
    ax1.set_ylabel('Mean IC')
    ax1.set_title('IC Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IR衰减
    ax2 = axes[1]
    for factor_label, stats_dict in ic_stats_all.items():
        periods = sorted(stats_dict.keys())
        irs = [stats_dict[p]["ir"] for p in periods]
        ax2.plot(periods, irs, marker='o', label=factor_label, linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Forward Return Period (Days)')
    ax2.set_ylabel('IR')
    ax2.set_title('IR Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  IC/IR衰减图已保存: {output_path}")


def plot_quantile_returns(quantile_results: dict, holding_period: int, output_path: str):
    """分组收益柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(quantile_results[list(quantile_results.keys())[0]]))
    width = 0.35

    colors = ['steelblue', 'coral']
    for idx, (factor_label, q_ret) in enumerate(quantile_results.items()):
        offset = (idx - 0.5) * width if len(quantile_results) > 1 else 0
        ax.bar(x + offset, q_ret.values * 100, width, label=factor_label, color=colors[idx % len(colors)])

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(q_ret.index)
    ax.set_ylabel('Average Forward Return (%)')
    ax.set_xlabel('Factor Quantile')
    ax.set_title(f'Quantile Returns ({holding_period}D Forward Return)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  分组收益图已保存: {output_path}")


# =============================================================================
# 主分析流程
# =============================================================================
def main():
    print("=" * 80)
    print("因子 IC 分析模块")
    print("=" * 80)

    # 1. 加载价差数据（只需要加载一次）
    spread_df = load_spread_data()

    # 2. 计算多期 forward returns（基于价差）
    print("\n计算多期 forward returns（基于价差变化）...")
    fwd_returns = calc_forward_returns(spread_df, HOLDING_PERIODS)
    for p in HOLDING_PERIODS:
        print(f"  {p}D forward return shape: {fwd_returns[p].shape}")

    # 存储各因子的分析结果
    ic_stats_all = {}  # {factor_label: {period: stats}}

    for factor_config in FACTORS:
        print(f"\n{'='*60}")
        print(f"分析因子: {factor_config['label']}")
        print(f"{'='*60}")

        # 加载因子
        factor_df = load_factor(factor_config)
        print(f"因子数据: {factor_df.shape}")

        ic_stats_all[factor_config["label"]] = {}

        for period in HOLDING_PERIODS:
            print(f"\n  持有期: {period}D")

            # 对齐因子和收益数据
            common_index = factor_df.index.intersection(fwd_returns[period].index)
            f = factor_df.loc[common_index]
            r = fwd_returns[period].loc[common_index]

            # 计算日度IC
            ic_series = calc_daily_ic(f, r)

            # IC统计
            stats_dict = calc_ic_stats(ic_series)
            if stats_dict:
                ic_stats_all[factor_config["label"]][period] = stats_dict
                print(f"    Mean IC : {stats_dict['mean_ic']:+.4f}")
                print(f"    Std IC  : {stats_dict['std_ic']:.4f}")
                print(f"    IR      : {stats_dict['ir']:+.4f}")
                print(f"    IR(ann) : {stats_dict['ir_annual']:+.4f}")
                print(f"    IC>0    : {stats_dict['ic_positive_ratio']*100:.1f}%")
                print(f"    t-stat  : {stats_dict['t_stat']:+.4f}")
                print(f"    N       : {stats_dict['ic_count']}")

            # 分组收益分析（只在5D持有期做，避免输出过多）
            if period == 5:
                q_ret = calc_quantile_returns(f, r, n_quantiles=5)
                if not q_ret.empty:
                    print(f"    Quantile Returns:")
                    for q, ret in q_ret.items():
                        print(f"      {q}: {ret*100:+.4f}%")

    # =============================================================================
    # 对比图表输出
    # =============================================================================
    print(f"\n{'='*60}")
    print("生成对比图表...")
    print(f"{'='*60}")

    # 为每个持有期绘制IC时间序列和分布图
    for period in [5, 10]:  # 重点展示5D和10D
        # 重新计算IC序列用于绘图
        ic_series_dict = {}
        for factor_config in FACTORS:
            factor_df = load_factor(factor_config)
            common_index = factor_df.index.intersection(fwd_returns[period].index)
            f = factor_df.loc[common_index]
            r = fwd_returns[period].loc[common_index]
            ic_series = calc_daily_ic(f, r)
            ic_series_dict[factor_config["label"]] = ic_series

        plot_ic_timeseries(ic_series_dict, period, f"{OUTPUT_DIR}/ic_timeseries_{period}d.png")
        plot_ic_distribution(ic_series_dict, period, f"{OUTPUT_DIR}/ic_distribution_{period}d.png")

    # IC/IR 衰减图
    plot_ic_decay(ic_stats_all, f"{OUTPUT_DIR}/ic_decay.png")

    # 分组收益图（5D）
    quantile_results = {}
    for factor_config in FACTORS:
        factor_df = load_factor(factor_config)
        common_index = factor_df.index.intersection(fwd_returns[5].index)
        f = factor_df.loc[common_index]
        r = fwd_returns[5].loc[common_index]
        q_ret = calc_quantile_returns(f, r, n_quantiles=5)
        if not q_ret.empty:
            quantile_results[factor_config["label"]] = q_ret
    if quantile_results:
        plot_quantile_returns(quantile_results, 5, f"{OUTPUT_DIR}/ic_quantile_returns_5d.png")

    # =============================================================================
    # 汇总报告
    # =============================================================================
    print(f"\n{'='*80}")
    print("【IC 分析汇总报告】")
    print(f"{'='*80}")

    for period in HOLDING_PERIODS:
        print(f"\n{'-'*60}")
        print(f"持有期: {period}D Forward Return")
        print(f"{'-'*60}")
        print(f"{'因子':<40} {'Mean IC':>10} {'Std IC':>10} {'IR':>10} {'IC>0%':>10} {'t-stat':>10}")

        for factor_config in FACTORS:
            label = factor_config["label"]
            stats_dict = ic_stats_all.get(label, {}).get(period, {})
            if stats_dict:
                print(f"{label:<40} {stats_dict['mean_ic']:>+10.4f} {stats_dict['std_ic']:>10.4f} "
                      f"{stats_dict['ir']:>+10.4f} {stats_dict['ic_positive_ratio']*100:>9.1f}% {stats_dict['t_stat']:>+10.4f}")
            else:
                print(f"{label:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print(f"\n{'='*80}")
    print("IC 分析完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
