"""
计算纯 889 动量因子并跑完整回测
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
from vnpy_alpharesearch.utility import load_bar_df
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance
from strategies.cross_sectional_strategy import CrossSectionalStrategy

START = datetime(2020, 1, 1)
END = datetime(2024, 12, 31)
CYCLE = 20

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

print("=" * 70)
print("计算纯 889 动量因子 (F1 的 20 日收益)")
print("=" * 70)

momentum_889 = {}
for ds in dominant_symbols:
    sym_889 = ds.replace('88.', '889.')
    try:
        df = load_bar_df(sym_889, Interval.DAILY, START, END)
        if len(df) > 0:
            close = df['close_price']
            ret = close / close.shift(CYCLE) - 1
            momentum_889[ds] = ret
    except Exception as e:
        pass

factor_df = pd.DataFrame(momentum_889)
factor_df.index.name = 'datetime'
print(f"因子矩阵: {factor_df.shape}")

# 保存到 DataCenter
dc = DataCenter()
for vt_symbol in factor_df.columns:
    series = factor_df[vt_symbol]
    series = series.dropna()
    if len(series) == 0:
        continue
    factor_id = dc.get_factor_id(
        "momentum_889", vt_symbol, "", "d", "cs_developer"
    )
    key = str(factor_id)
    dc.save_reference_series("vnpy_factor_data", key, series)

print(f"因子已保存到 DataCenter")

# 运行回测
print("\n" + "=" * 70)
print("运行完整回测")
print("=" * 70)

backtester = StrategyBacktester(dominant_symbols, Interval.DAILY, START, END, 10_000_000)
backtester.load_data()

strategy_setting = {
    "holding_period": 1,
    "trading_signal": 0.0,
    "factor_name": "momentum_889",
    "factor_parameter": "",
    "factor_author": "cs_developer",
    "long_low": False,
    "aggregation": "top_n",
    "leverage": 1.0,
}

target_df = backtester.run_backtesting(CrossSectionalStrategy, strategy_setting)

non_zero_days = (target_df != 0).any(axis=1).sum()
print(f"有仓位的交易日: {non_zero_days}/{len(target_df)}")

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
prefix = "/root/cs_developer/result_momentum_889"
target_df.to_csv(f"{prefix}_target.csv")
result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
result["product"].to_csv(f"{prefix}_product.csv")
print(f"\n[OK] 结果已保存到 {prefix}_*.csv")
