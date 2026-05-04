#!/usr/bin/env python3
"""
批量回测入口 — 对接 factor_system 与策略回测框架

功能：
1. 加载所有预计算因子（parquet）
2. 对每个因子执行向量化回测（BatchBacktestEngine）
3. 自动选择多空方向（取夏普更高者）
4. 输出绩效对比表（IC vs 回测指标）
5. 绘制净值曲线对比图

使用示例：
    python run_batch_backtest.py --holding-period 5 --trading-signal 0.2 --leverage 2.0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from backtest_engine import BatchBacktestEngine
from factor_registry import get_registry


# =============================================================================
# 默认配置
# =============================================================================

DEFAULT_SYMBOLS = [
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

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"


def parse_args():
    parser = argparse.ArgumentParser(description="批量因子回测")
    parser.add_argument("--factors", type=str, default="",
                        help="逗号分隔的因子列表，默认所有")
    parser.add_argument("--holding-period", type=int, default=5,
                        help="持仓周期（默认5天）")
    parser.add_argument("--trading-signal", type=float, default=0.2,
                        help="每端交易比例（默认0.2=20%）")
    parser.add_argument("--leverage", type=float, default=2.0,
                        help="名义杠杆倍数（默认2.0）")
    parser.add_argument("--commission", type=float, default=0.0001,
                        help="单边手续费率（默认0.0001=万1）")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean"],
                        help="仓位汇总模式")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help="回测开始日期")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help="回测结束日期")
    parser.add_argument("--output-dir", type=str, default="/root/cs_developer/factor_system/reports",
                        help="输出目录")
    parser.add_argument("--skip-plot", action="store_true",
                        help="跳过绘图")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="详细输出")
    return parser.parse_args()


def load_ic_results(report_dir: Path) -> pd.DataFrame:
    """加载 IC 分析结果"""
    ic_path = report_dir / "factor_summary_5d.csv"
    if ic_path.exists():
        return pd.read_csv(ic_path)
    return pd.DataFrame()


def run_batch_backtest(args) -> pd.DataFrame:
    """执行批量回测"""
    
    print("=" * 70)
    print("批量因子回测 — BatchBacktestEngine")
    print("=" * 70)
    print(f"回测区间: {args.start} ~ {args.end}")
    print(f"持仓周期: {args.holding_period}天")
    print(f"交易比例: {args.trading_signal:.0%}")
    print(f"杠杆倍数: {args.leverage}x")
    print(f"手续费率: {args.commission:.4f}")
    print(f"汇总模式: {args.aggregation}")
    print("=" * 70)
    
    # 初始化引擎
    engine = BatchBacktestEngine(
        dominant_symbols=DEFAULT_SYMBOLS,
        start=args.start,
        end=args.end,
        verbose=args.verbose
    )
    
    # 确定因子列表
    if args.factors:
        factor_names = [f.strip() for f in args.factors.split(",")]
    else:
        factor_names = engine._extract_factor_names()
    
    print(f"\n待回测因子数: {len(factor_names)}")
    
    # 执行批量回测
    results = engine.run_all(
        factor_names=factor_names,
        holding_period=args.holding_period,
        trading_signal=args.trading_signal,
        aggregation=args.aggregation,
        leverage=args.leverage,
        commission=args.commission,
    )
    
    # 生成对比表
    compare_df = engine.compare_results(results, sort_by="sharpe_ratio", ascending=False)
    
    return compare_df, results


def merge_with_ic(compare_df: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    """将回测结果与 IC 分析合并"""
    if ic_df.empty:
        return compare_df
    
    merged = compare_df.merge(
        ic_df.rename(columns={
            "factor": "factor_name",
            "mean_ic": "ic_5d",
            "ir": "ir_5d",
            "t_stat": "t_stat_5d",
        }),
        on="factor_name",
        how="left"
    )
    
    # 重新排列列
    cols = ["factor_name", "category", "ic_5d", "ir_5d", "long_low",
            "annual_return", "annual_vol", "sharpe_ratio", "max_drawdown",
            "calmar_ratio", "win_rate", "avg_turnover", "total_days"]
    cols = [c for c in cols if c in merged.columns]
    remaining = [c for c in merged.columns if c not in cols]
    merged = merged[cols + remaining]
    
    return merged


def save_results(merged_df: pd.DataFrame, results: dict, args):
    """保存结果到文件"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存 CSV
    csv_path = output_dir / f"backtest_comparison_{timestamp}.csv"
    merged_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[OK] 对比表已保存: {csv_path}")
    
    # 2. 保存 Markdown 报告
    md_path = output_dir / f"backtest_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 因子批量回测报告\n\n")
        f.write(f"**回测区间**: {args.start} ~ {args.end}\n\n")
        f.write(f"**持仓周期**: {args.holding_period}天\n\n")
        f.write(f"**交易比例**: {args.trading_signal:.0%}\n\n")
        f.write(f"**杠杆倍数**: {args.leverage}x\n\n")
        f.write(f"**手续费率**: {args.commission:.4f}（单边）\n\n")
        f.write(f"**汇总模式**: {args.aggregation}\n\n")
        f.write("## 绩效对比表\n\n")
        
        # 格式化输出
        display_df = merged_df.copy()
        for col in ["annual_return", "annual_vol", "max_drawdown", "win_rate", "avg_turnover"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        for col in ["sharpe_ratio", "calmar_ratio", "ic_5d", "ir_5d"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        
        f.write(display_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Top 3 因子详细分析
        f.write("## Top 3 因子详细分析\n\n")
        for i, row in merged_df.head(3).iterrows():
            f.write(f"### {i+1}. {row['factor_name']}\n\n")
            f.write(f"- **类别**: {row.get('category', 'N/A')}\n")
            f.write(f"- **IC(5D)**: {row.get('ic_5d', 'N/A'):.4f}\n")
            f.write(f"- **IR(5D)**: {row.get('ir_5d', 'N/A'):.3f}\n")
            f.write(f"- **方向**: long_low={row.get('long_low', 'N/A')}\n")
            f.write(f"- **年化收益**: {row['annual_return']:.2%}\n")
            f.write(f"- **夏普比率**: {row['sharpe_ratio']:.2f}\n")
            f.write(f"- **最大回撤**: {row['max_drawdown']:.2%}\n")
            f.write(f"- **Calmar**: {row['calmar_ratio']:.2f}\n")
            f.write(f"- **胜率**: {row['win_rate']:.1%}\n")
            f.write("\n")
    
    print(f"[OK] Markdown 报告已保存: {md_path}")
    
    # 3. 保存每个因子的净值序列（用于后续分析）
    nav_path = output_dir / f"backtest_nav_{timestamp}.csv"
    nav_data = {}
    for name, res in results.items():
        nav_data[name] = res.nav_series
    
    if nav_data:
        nav_df = pd.DataFrame(nav_data)
        nav_df.to_csv(nav_path)
        print(f"[OK] 净值序列已保存: {nav_path}")
    
    return csv_path, md_path


def plot_comparison(merged_df: pd.DataFrame, results: dict, args):
    """绘制对比图"""
    if args.skip_plot:
        return
    
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 夏普 vs IC 散点图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if "ic_5d" in merged_df.columns and "sharpe_ratio" in merged_df.columns:
        ax = axes[0, 0]
        colors = merged_df["category"].astype("category").cat.codes
        scatter = ax.scatter(
            merged_df["ic_5d"],
            merged_df["sharpe_ratio"],
            c=colors,
            cmap="tab10",
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("IC (5D)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("IC vs Backtest Sharpe")
        
        # 标注 carry_ret
        carry_row = merged_df[merged_df["factor_name"] == "carry_ret"]
        if not carry_row.empty:
            ax.annotate(
                "carry_ret",
                (carry_row["ic_5d"].values[0], carry_row["sharpe_ratio"].values[0]),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=9,
                color="red",
                fontweight="bold"
            )
    
    # 2. 夏普排名柱状图
    ax = axes[0, 1]
    top10 = merged_df.nlargest(10, "sharpe_ratio")
    colors = ["green" if s > 0 else "red" for s in top10["sharpe_ratio"]]
    bars = ax.barh(range(len(top10)), top10["sharpe_ratio"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["factor_name"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Top 10 Factors by Sharpe")
    ax.axvline(0, color="black", linewidth=0.8)
    
    # 3. 年化收益 vs 最大回撤
    ax = axes[1, 0]
    if "annual_return" in merged_df.columns and "max_drawdown" in merged_df.columns:
        ax.scatter(
            merged_df["max_drawdown"],
            merged_df["annual_return"],
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Max Drawdown")
        ax.set_ylabel("Annual Return")
        ax.set_title("Return vs Drawdown")
        
        carry_row = merged_df[merged_df["factor_name"] == "carry_ret"]
        if not carry_row.empty:
            ax.annotate(
                "carry_ret",
                (carry_row["max_drawdown"].values[0], carry_row["annual_return"].values[0]),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=9,
                color="red",
                fontweight="bold"
            )
    
    # 4. Top 5 因子净值曲线
    ax = axes[1, 1]
    top5 = merged_df.nlargest(5, "sharpe_ratio")["factor_name"].tolist()
    for name in top5:
        if name in results:
            nav = results[name].nav_series
            ax.plot(nav.index, nav.values, label=name, linewidth=1.2)
    
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.set_title("Top 5 Factor NAV Curves")
    ax.legend(loc="upper left", fontsize=8)
    
    plt.tight_layout()
    plot_path = output_dir / f"backtest_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 对比图已保存: {plot_path}")


def main():
    args = parse_args()
    
    # 执行回测
    compare_df, results = run_batch_backtest(args)
    
    # 加载 IC 结果并合并
    ic_df = load_ic_results(Path(args.output_dir))
    merged_df = merge_with_ic(compare_df, ic_df)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("批量回测结果汇总")
    print("=" * 70)
    
    display_cols = ["factor_name", "category", "long_low", "annual_return",
                    "sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate"]
    display_cols = [c for c in display_cols if c in merged_df.columns]
    print(merged_df[display_cols].head(15).to_string(index=False))
    
    # 保存结果
    save_results(merged_df, results, args)
    
    # 绘图
    plot_comparison(merged_df, results, args)
    
    print("\n" + "=" * 70)
    print("批量回测全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
