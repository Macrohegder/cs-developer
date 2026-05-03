#!/usr/bin/env python3
"""
期限结构策略回测 — 时间序列模式（每个品种独立做反转）
"""

from datetime import datetime

from vnpy.trader.constant import Interval

from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from timeseries_term_structure_strategy import TimeSeriesTermStructureStrategy


CONFIG = {
    "dominant_symbols": [
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
    ],

    "start": datetime(2019, 1, 1),
    "end": datetime(2025, 12, 31),

    "capital": 10_000_000,

    "strategy_setting": {
        "holding_period": 10,
        "max_positions": 10,
        "factor_name": "spread_return",
        "factor_parameter": "lookback10",
        "factor_author": "futures_term_structure"
    },

    "commission": 0.0001,
    "risk_free": 0,
    "plot_chart": False,
}


def step2_strategy_backtest(config: dict):
    print("\n" + "=" * 70)
    print("Step 2: 时间序列反转策略回测（每个品种独立判断）")
    print("=" * 70)

    dominant_symbols = config["dominant_symbols"]
    start = config["start"]
    end = config["end"]
    capital = config["capital"]

    backtester = StrategyBacktester(
        dominant_symbols,
        Interval.DAILY,
        start,
        end,
        capital
    )

    print("加载历史行情数据...")
    backtester.load_data()

    print("运行回测...")
    target_df = backtester.run_backtesting(
        TimeSeriesTermStructureStrategy,
        config["strategy_setting"]
    )
    print(f"  回测完成，目标仓位形状: {target_df.shape}")

    non_zero_days = (target_df != 0).any(axis=1).sum()
    print(f"  有仓位的交易日: {non_zero_days}/{len(target_df)}")
    print(f"  涉及的具体合约数: {len(target_df.columns)}")

    return target_df


def step3_performance_analysis(target_df, config: dict) -> dict:
    print("\n" + "=" * 70)
    print("Step 3: 绩效分析")
    print("=" * 70)

    result = calculate_portfolio_performance(
        target_df,
        Interval.DAILY,
        commission=config["commission"],
        capital=config["capital"],
        risk_free=config["risk_free"],
        plot_chart=config["plot_chart"]
    )

    stats = result["statistics"]
    print("\n【策略绩效指标】")
    for key, value in stats.items():
        print(f"  {key:<20s}: {value}")

    return result


def save_results(target_df, result: dict, config: dict) -> None:
    import pandas as pd

    prefix = "/root/cs_developer/result_timeseries"

    target_path = f"{prefix}_target.csv"
    target_df.to_csv(target_path)
    print(f"\n  目标仓位已保存: {target_path}")

    if "overall" in result:
        pnl_path = f"{prefix}_pnl.csv"
        result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(pnl_path)
        print(f"  净值曲线已保存: {pnl_path}")

    if "product" in result:
        product_path = f"{prefix}_product.csv"
        result["product"].to_csv(product_path)
        print(f"  分品种盈亏已保存: {product_path}")


def main():
    config = CONFIG

    target_df = step2_strategy_backtest(config)
    result = step3_performance_analysis(target_df, config)
    save_results(target_df, result, config)

    print("\n" + "=" * 70)
    print("时间序列反转策略回测执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()
