"""
回测 carry_ret_exact（旧版严谨定义）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "factor_system"))
sys.path.insert(0, "/tmp/trading-system--master-vnpy_alpharesearch/vnpy_alpharesearch")

import pandas as pd
import numpy as np
from datetime import datetime
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance
from strategies.cross_sectional_strategy import CrossSectionalStrategy
from factor_system.factor_registry import FactorRegistry, FactorMeta

START = datetime(2020, 1, 1)
END = datetime(2024, 12, 31)

dominant_symbols = [
    'RB88.SHFE', 'HC88.SHFE', 'I88.DCE', 'J88.DCE', 'JM88.DCE', 
    'BU88.SHFE', 'RU88.SHFE', 'MA88.CZCE', 'TA88.CZCE', 'EG88.DCE',
    'PP88.DCE', 'L88.DCE', 'V88.DCE', 'SA88.CZCE', 'FG88.CZCE',
    'SM88.CZCE', 'SF88.CZCE', 'SR88.CZCE', 'CF88.CZCE', 'OI88.CZCE',
    'RM88.CZCE', 'M88.DCE', 'Y88.DCE', 'P88.DCE', 'A88.DCE',
    'JD88.DCE', 'AP88.CZCE', 'CJ88.CZCE', 'CY88.CZCE', 'UR88.CZCE',
    'PF88.CZCE', 'PG88.DCE', 'EB88.DCE', 'LU88.INE', 'SP88.SHFE',
    'AL88.SHFE', 'CU88.SHFE', 'ZN88.SHFE', 'NI88.SHFE', 'SN88.SHFE',
    'PB88.SHFE', 'AG88.SHFE', 'AU88.SHFE', 'SS88.SHFE', 'FU88.SHFE',
    'SC88.INE', 'NR88.SHFE', 'BC88.SHFE', 'IC88.CFFEX', 'IF88.CFFEX', 'IH88.CFFEX'
]

# 注册因子
registry = FactorRegistry()
registry.register(FactorMeta(
    name="carry_ret_exact",
    category="carry",
    sub_category="term_structure",
    description="旧版严谨carry = ln(mean(F1_cycle)/mean(F2_cycle)) / 真实到期日差 * 365",
    params={"cycle": 5},
    data_requirements=["close_price", "contract_expiry"],
    author="cs_developer",
    source="cs_developer",
    lookback_days=5,
    ic_direction=1,  # 高carry做多
))

# 加载预计算因子
carry_exact_df = pd.read_csv("factor_carry_ret_exact.csv", parse_dates=["datetime"], index_col="datetime")

# 保存到 DataCenter
dc = DataCenter()
for vt_symbol in carry_exact_df.columns:
    series = carry_exact_df[vt_symbol].dropna()
    if len(series) == 0:
        continue
    factor_id = dc.get_factor_id(
        "carry_ret_exact", vt_symbol, "", "d", "cs_developer"
    )
    key = str(factor_id)
    dc.save_reference_series("vnpy_factor_data", key, series)

print("=" * 70)
print("运行 carry_ret_exact 完整回测")
print("=" * 70)

backtester = StrategyBacktester(dominant_symbols, Interval.DAILY, START, END, 10_000_000)
backtester.load_data()

strategy_setting = {
    "holding_period": 1,
    "trading_signal": 0.0,
    "factor_name": "carry_ret_exact",
    "factor_parameter": "",
    "factor_author": "cs_developer",
    "long_low": False,  # 高carry做多
    "aggregation": "top_n",
    "leverage": 1.0,
}

target_df = backtester.run_backtesting(CrossSectionalStrategy, strategy_setting)

non_zero_days = (target_df != 0).any(axis=1).sum()
print(f"\n有仓位的交易日: {non_zero_days}/{len(target_df)}")

result = calculate_portfolio_performance(
    target_df,
    Interval.DAILY,
    commission=0.0001,
    capital=10_000_000,
    risk_free=0.02,
    plot_chart=False
)

stats = result["statistics"]
print("\n【完整回测绩效指标】")
for key, value in stats.items():
    print(f"  {key:<20s}: {value}")

# 保存结果
prefix = "/root/cs_developer/result_carry_ret_exact"
target_df.to_csv(f"{prefix}_target.csv")
result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
result["product"].to_csv(f"{prefix}_product.csv")
print(f"\n[OK] 结果已保存到 {prefix}_*.csv")
