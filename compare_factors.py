#!/usr/bin/env python3
"""
双因子对比分析脚本
对比 spread_zscore（均值回归）与 spread_return（价差动量反转）的回测结果
使用 matplotlib 生成 PNG 图片，不使用 plotly
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，不依赖图形界面
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 配置路径
# =============================================================================
ZSCORE_PNL = "/root/futures_term_structure_strategies/result_zscore_pnl.csv"
RETURN_PNL = "/root/futures_term_structure_strategies/result_spread_return_pnl.csv"
ZSCORE_PRODUCT = "/root/futures_term_structure_strategies/result_zscore_product.csv"
RETURN_PRODUCT = "/root/futures_term_structure_strategies/result_spread_return_product.csv"
OUTPUT_DIR = "/root/futures_term_structure_strategies"


def load_pnl(path):
    """加载净值曲线数据"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df['returns'] = df['balance'].pct_change().fillna(0)
    return df


def calc_stats(pnl_df, name):
    """计算核心绩效指标"""
    returns = pnl_df['returns']
    balance = pnl_df['balance']

    total_return = balance.iloc[-1] / balance.iloc[0] - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    # 最大回撤
    cummax = balance.cummax()
    drawdown = (balance - cummax) / cummax
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # 胜率
    profit_days = (returns > 0).sum()
    loss_days = (returns < 0).sum()
    win_rate = profit_days / (profit_days + loss_days) if (profit_days + loss_days) > 0 else 0

    return {
        'name': name,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_days': profit_days,
        'loss_days': loss_days,
        'start_balance': balance.iloc[0],
        'end_balance': balance.iloc[-1],
    }


def plot_comparison(zscore_pnl, return_pnl, output_path):
    """绘制净值曲线与回撤对比图（双轴）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # 净值曲线
    ax1 = axes[0]
    zscore_cum = (zscore_pnl['balance'] / zscore_pnl['balance'].iloc[0] - 1) * 100
    return_cum = (return_pnl['balance'] / return_pnl['balance'].iloc[0] - 1) * 100

    ax1.plot(zscore_cum.index, zscore_cum.values, label='Spread Z-Score (Mean Reversion)', color='blue', linewidth=1.2)
    ax1.plot(return_cum.index, return_cum.values, label='Spread Return (Momentum Reversal)', color='red', linewidth=1.2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Factor Comparison: Equity Curve')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    ax2 = axes[1]
    zscore_dd = zscore_pnl['ddpercent'] * 100
    return_dd = return_pnl['ddpercent'] * 100

    ax2.fill_between(zscore_dd.index, zscore_dd.values, 0, color='blue', alpha=0.3, label='Z-Score Drawdown')
    ax2.fill_between(return_dd.index, return_dd.values, 0, color='red', alpha=0.3, label='Return Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Drawdown Comparison')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  净值曲线对比图已保存: {output_path}")


def plot_annual_returns(zscore_pnl, return_pnl, output_path):
    """绘制年度收益对比柱状图"""
    zscore_pnl['year'] = zscore_pnl.index.year
    return_pnl['year'] = return_pnl.index.year

    zscore_annual = zscore_pnl.groupby('year').apply(lambda x: (1 + x['returns']).prod() - 1) * 100
    return_annual = return_pnl.groupby('year').apply(lambda x: (1 + x['returns']).prod() - 1) * 100

    years = sorted(set(zscore_annual.index) | set(return_annual.index))
    z_vals = [zscore_annual.get(y, 0) for y in years]
    r_vals = [return_annual.get(y, 0) for y in years]

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, z_vals, width, label='Spread Z-Score', color='steelblue')
    bars2 = ax.bar(x + width/2, r_vals, width, label='Spread Return', color='coral')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Annual Return Comparison by Year')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  年度收益对比图已保存: {output_path}")


def plot_product_comparison(zscore_prod, return_prod, output_path, top_n=20):
    """绘制分品种盈亏对比（取总盈亏绝对值最大的 top_n 品种）"""
    # product csv 的列是品种，行是日期，需要转置并求和
    zscore_prod = zscore_prod.set_index('datetime')
    return_prod = return_prod.set_index('datetime')

    zscore_total = zscore_prod.sum().reset_index()
    zscore_total.columns = ['product', 'zscore_pnl']

    return_total = return_prod.sum().reset_index()
    return_total.columns = ['product', 'return_pnl']

    # 合并两个因子的分品种收益
    merged = pd.merge(zscore_total, return_total, on='product', how='outer').fillna(0)

    merged['abs_sum'] = merged['zscore_pnl'].abs() + merged['return_pnl'].abs()
    merged = merged.sort_values('abs_sum', ascending=False).head(top_n)

    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, merged['zscore_pnl'] / 10000, width, label='Spread Z-Score', color='steelblue')
    ax.bar(x + width/2, merged['return_pnl'] / 10000, width, label='Spread Return', color='coral')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(merged['product'], rotation=45, ha='right')
    ax.set_ylabel('Total P&L (x10,000)')
    ax.set_title(f'Top {top_n} Product P&L Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  分品种盈亏对比图已保存: {output_path}")


def print_comparison_table(zscore_stats, return_stats):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("【双因子回测对比分析】")
    print("=" * 80)

    print(f"\n{'指标':<25} {'Spread Z-Score (均值回归)':>25} {'Spread Return (动量反转)':>25}")
    print("-" * 80)
    print(f"{'总收益率':<25} {zscore_stats['total_return']*100:>24.2f}% {return_stats['total_return']*100:>24.2f}%")
    print(f"{'年化收益率':<25} {zscore_stats['annual_return']*100:>24.2f}% {return_stats['annual_return']*100:>24.2f}%")
    print(f"{'年化波动率':<25} {zscore_stats['volatility']*100:>24.2f}% {return_stats['volatility']*100:>24.2f}%")
    print(f"{'夏普比率':<25} {zscore_stats['sharpe_ratio']:>25.2f} {return_stats['sharpe_ratio']:>25.2f}")
    print(f"{'最大回撤':<25} {zscore_stats['max_drawdown']*100:>24.2f}% {return_stats['max_drawdown']*100:>24.2f}%")
    print(f"{'卡玛比率':<25} {zscore_stats['calmar_ratio']:>25.2f} {return_stats['calmar_ratio']:>25.2f}")
    print(f"{'日胜率':<25} {zscore_stats['win_rate']*100:>24.2f}% {return_stats['win_rate']*100:>24.2f}%")
    print(f"{'盈利天数':<25} {zscore_stats['profit_days']:>25} {return_stats['profit_days']:>25}")
    print(f"{'亏损天数':<25} {zscore_stats['loss_days']:>25} {return_stats['loss_days']:>25}")
    print(f"{'初始资金':<25} {zscore_stats['start_balance']:>25,.0f} {return_stats['start_balance']:>25,.0f}")
    print(f"{'期末资金':<25} {zscore_stats['end_balance']:>25,.2f} {return_stats['end_balance']:>25,.2f}")

    print("\n" + "=" * 80)
    print("【因子逻辑说明】")
    print("-" * 80)
    print("Spread Z-Score (均值回归):")
    print("  - 因子 = (当前价差 - 20日均价差) / 20日标准差")
    print("  - 做空前10%（Z-Score最高，价差偏离均值向上）")
    print("  - 做多后10%（Z-Score最低，价差偏离均值向下）")
    print("  - 假设：价差会均值回归")
    print()
    print("Spread Return (动量反转):")
    print("  - 因子 = (当前价差 - 10日前价差) / |10日前价差|")
    print("  - 做空前10%（过去10天价差涨幅最大）")
    print("  - 做多后10%（过去10天价差跌幅最大）")
    print("  - 假设：价差动量过度后会反转")

    print("\n" + "=" * 80)
    print("【对比结论】")
    print("-" * 80)
    total_diff = (return_stats['total_return'] - zscore_stats['total_return']) * 100
    sharpe_diff = return_stats['sharpe_ratio'] - zscore_stats['sharpe_ratio']
    dd_diff = (return_stats['max_drawdown'] - zscore_stats['max_drawdown']) * 100

    if return_stats['total_return'] > zscore_stats['total_return']:
        print(f"  • 总收益: Spread Return 优于 Z-Score，高出 {total_diff:.2f}%")
    else:
        print(f"  • 总收益: Z-Score 优于 Spread Return，高出 {abs(total_diff):.2f}%")

    if return_stats['sharpe_ratio'] > zscore_stats['sharpe_ratio']:
        print(f"  • 夏普比率: Spread Return 更优 ({sharpe_diff:+.2f})")
    else:
        print(f"  • 夏普比率: Z-Score 更优 ({sharpe_diff:+.2f})")

    if abs(return_stats['max_drawdown']) < abs(zscore_stats['max_drawdown']):
        print(f"  • 最大回撤: Spread Return 控制更好 ({dd_diff:+.2f}%)")
    else:
        print(f"  • 最大回撤: Z-Score 控制更好 ({dd_diff:+.2f}%)")

    print("=" * 80)


def main():
    print("加载回测结果数据...")
    zscore_pnl = load_pnl(ZSCORE_PNL)
    return_pnl = load_pnl(RETURN_PNL)
    zscore_prod = pd.read_csv(ZSCORE_PRODUCT)
    return_prod = pd.read_csv(RETURN_PRODUCT)

    print("计算绩效指标...")
    zscore_stats = calc_stats(zscore_pnl, 'Spread Z-Score')
    return_stats = calc_stats(return_pnl, 'Spread Return')

    print("生成对比图表...")
    plot_comparison(zscore_pnl, return_pnl, f"{OUTPUT_DIR}/compare_equity_curve.png")
    plot_annual_returns(zscore_pnl, return_pnl, f"{OUTPUT_DIR}/compare_annual_returns.png")
    plot_product_comparison(zscore_prod, return_prod, f"{OUTPUT_DIR}/compare_product_pnl.png")

    print_comparison_table(zscore_stats, return_stats)

    print("\n对比分析完成，所有图片已保存为 PNG 格式。")


if __name__ == "__main__":
    main()
