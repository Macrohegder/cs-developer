#!/usr/bin/env python3
"""
Skew 因子参数网格搜索 — 高效版本

优化点：
- 因子已缓存，跳过 Step 1
- StrategyBacktester 只实例化/加载数据一次，复用历史数据循环回测多组参数
- 全区间 2015-2025 回测
"""

import sys
from pathlib import Path
from datetime import datetime
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from pandas import DataFrame
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance
from strategies.cross_sectional_strategy import CrossSectionalStrategy


# =============================================================================
# 搜索配置
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

START = datetime(2015, 1, 1)
END = datetime(2025, 12, 31)
CAPITAL = 10_000_000
COMMISSION = 0.0001

# 参数网格
HOLDING_PERIODS = [3, 5, 10]
TRADING_SIGNALS = [0.1, 0.2, 0.3]
LEVERAGES = [1.0, 2.0, 3.0]


def run_grid_search():
    print("=" * 80)
    print("Skew 因子参数网格搜索")
    print("=" * 80)
    print(f"区间: {START.date()} ~ {END.date()}")
    print(f"参数网格: holding_period={HOLDING_PERIODS}, trading_signal={TRADING_SIGNALS}, leverage={LEVERAGES}")
    print(f"总组合数: {len(HOLDING_PERIODS) * len(TRADING_SIGNALS) * len(LEVERAGES)}")
    print()

    # 1. 初始化回测器并加载数据（只执行一次）
    print("[1/2] 加载历史行情数据...")
    backtester = StrategyBacktester(
        DOMINANT_SYMBOLS,
        Interval.DAILY,
        START,
        END,
        CAPITAL
    )
    backtester.load_data()
    print("  数据加载完成，开始网格搜索...\n")

    # 2. 循环参数组合
    results = []
    total = len(HOLDING_PERIODS) * len(TRADING_SIGNALS) * len(LEVERAGES)
    idx = 0

    for hp, ts, lev in product(HOLDING_PERIODS, TRADING_SIGNALS, LEVERAGES):
        idx += 1
        print(f"[{idx}/{total}] hp={hp}, ts={ts}, lev={lev}x")

        strategy_setting = {
            "holding_period": hp,
            "trading_signal": ts,
            "factor_name": "skew",
            "factor_parameter": "lookback180",
            "factor_author": "cross_sectional",
            "long_low": True,
            "aggregation": "sum",
            "leverage": lev,
        }

        try:
            target_df = backtester.run_backtesting(
                CrossSectionalStrategy,
                strategy_setting
            )

            result = calculate_portfolio_performance(
                target_df,
                Interval.DAILY,
                commission=COMMISSION,
                capital=CAPITAL,
                risk_free=0,
                plot_chart=False
            )

            stats = result["statistics"]
            results.append({
                "holding_period": hp,
                "trading_signal": ts,
                "leverage": lev,
                "total_return": stats.get("total_return", 0),
                "annual_return": stats.get("annual_return", 0),
                "max_ddpercent": stats.get("max_ddpercent", 0),
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
                "calmar_ratio": stats.get("calmar_ratio", 0),
                "profit_days": stats.get("profit_days", 0),
                "loss_days": stats.get("loss_days", 0),
            })
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 3. 输出结果表格
    print("\n" + "=" * 80)
    print("网格搜索结果")
    print("=" * 80)

    # 按夏普排序
    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    header = f"{'hp':>3} {'ts':>5} {'lev':>4} | {'总收益':>8} {'年化':>7} {'最大回撤':>9} {'夏普':>6} {'Calmar':>7}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['holding_period']:>3} {r['trading_signal']:>5.1f} {r['leverage']:>4.1f}x | "
            f"{r['total_return']:>+7.2f}% {r['annual_return']:>+6.2f}% "
            f"{r['max_ddpercent']:>8.2f}% {r['sharpe_ratio']:>+5.2f} {r['calmar_ratio']:>6.2f}"
        )

    # 输出最优组合
    best = results[0]
    print("\n" + "=" * 80)
    print("最优参数组合（按夏普排序）")
    print("=" * 80)
    print(f"  holding_period : {best['holding_period']}")
    print(f"  trading_signal : {best['trading_signal']}")
    print(f"  leverage       : {best['leverage']}x")
    print(f"  总收益         : {best['total_return']:+.2f}%")
    print(f"  年化收益       : {best['annual_return']:+.2f}%")
    print(f"  最大回撤       : {best['max_ddpercent']:.2f}%")
    print(f"  夏普比率       : {best['sharpe_ratio']:+.2f}")
    print(f"  Calmar         : {best['calmar_ratio']:.2f}")

    # 保存结果到 CSV
    import pandas as pd
    df = DataFrame(results)
    out_path = "/root/cs_developer/result_grid_search_skew.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  详细结果已保存: {out_path}")


if __name__ == "__main__":
    run_grid_search()
