#!/usr/bin/env python3
"""
有色金属板块期限结构策略回测 — 仅交易 CU, AL, ZN, NI, SN, AU, AG, PB
每次只选 1 对多空（价差涨幅最大 vs 最小）
"""

from datetime import datetime

from vnpy.trader.constant import Interval

from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from nonferrous_metal_strategy import NonferrousMetalTermStructureStrategy


CONFIG = {
    "dominant_symbols": [
        'CU88.SHFE',   # 铜
        'AL88.SHFE',   # 铝
        'ZN88.SHFE',   # 锌
        'NI88.SHFE',   # 镍
        'SN88.SHFE',   # 锡
        'AU88.SHFE',   # 黄金
        'AG88.SHFE',   # 白银
        'PB88.SHFE',   # 铅
    ],

    # 回测时间范围：从所有品种都有数据的最早日期开始
    "start": datetime(2015, 3, 27),
    "end": datetime(2025, 12, 31),

    "capital": 10_000_000,

    "strategy_setting": {
        "holding_period": 10,
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
    print("Step 2: 有色金属策略回测（8品种，每次1对多空）")
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
        NonferrousMetalTermStructureStrategy,
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

    prefix = "/root/futures_term_structure_strategies/result_nonferrous_metal"

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
    print("有色金属策略回测执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()
