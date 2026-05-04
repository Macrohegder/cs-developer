#!/usr/bin/env python3
"""
Carry_ret 因子完整回测验证 — 使用 StrategyBacktester

目的：验证向量化回测引擎的结果，对比简化版与完整版回测的差异
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance
from strategies.cross_sectional_strategy import CrossSectionalStrategy


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
    "start": datetime(2020, 1, 1),
    "end": datetime(2024, 12, 31),
    "capital": 10_000_000,
    "strategy_setting": {
        "holding_period": 5,
        "trading_signal": 0.2,
        "factor_name": "carry_ret",
        "factor_parameter": "",
        "factor_author": "factor_system",
        "long_low": False,      # IC 正相关 → 做多高 carry
        "aggregation": "sum",
        "leverage": 2.0,
    },
    "commission": 0.0001,
    "risk_free": 0,
    "plot_chart": False,
}


def main():
    config = CONFIG
    
    print("=" * 70)
    print("Carry_ret 因子完整回测（StrategyBacktester）")
    print("=" * 70)
    
    # 创建回测器
    backtester = StrategyBacktester(
        config["dominant_symbols"],
        Interval.DAILY,
        config["start"],
        config["end"],
        config["capital"]
    )
    
    # 加载数据
    print("\n[Step 1] 加载历史行情数据...")
    backtester.load_data()
    
    # 运行回测
    print("\n[Step 2] 运行回测...")
    target_df = backtester.run_backtesting(
        CrossSectionalStrategy,
        config["strategy_setting"]
    )
    
    non_zero_days = (target_df != 0).any(axis=1).sum()
    print(f"  有仓位的交易日: {non_zero_days}/{len(target_df)}")
    print(f"  涉及的具体合约数: {len(target_df.columns)}")
    
    # 绩效分析
    print("\n[Step 3] 绩效分析...")
    result = calculate_portfolio_performance(
        target_df,
        Interval.DAILY,
        commission=config["commission"],
        capital=config["capital"],
        risk_free=config["risk_free"],
        plot_chart=config["plot_chart"]
    )
    
    stats = result["statistics"]
    print("\n【完整回测绩效指标】")
    for key, value in stats.items():
        print(f"  {key:<20s}: {value}")
    
    # 保存结果
    prefix = "/root/cs_developer/result_carry_ret_full"
    target_df.to_csv(f"{prefix}_target.csv")
    result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
    result["product"].to_csv(f"{prefix}_product.csv")
    print(f"\n[OK] 结果已保存到 {prefix}_*.csv")
    
    print("\n" + "=" * 70)
    print("完整回测执行完毕！")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    main()
