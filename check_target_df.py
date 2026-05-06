#!/usr/bin/env python3
"""检查 raw vs roll5 的 target_df 统计特征"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
from datetime import datetime
from math import floor

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester

from strategies.cross_sectional_strategy import CrossSectionalStrategy

import pickle
with open('/root/cs_developer/carry_variants_cache.pkl', 'rb') as f:
    variants = pickle.load(f)

symbols = variants['raw'].columns.tolist()
start = datetime(2011, 1, 1)
end = datetime(2026, 3, 31)

class CarryVariantStrategy(CrossSectionalStrategy):
    def __init__(self, vt_symbols, interval, start, end, capital, setting, output):
        super().__init__(vt_symbols, interval, start, end, capital, setting, output)
        self.factor_data = setting.get("factor_data")
    
    def calculate_target(self, history_df):
        current_dt = self._current_dt
        if current_dt not in self.factor_data.index:
            return pd.Series(dtype=float)
        factor_series = self.factor_data.loc[current_dt].dropna()
        factor_series = factor_series[factor_series.index.isin(self.vt_symbols)]
        if len(factor_series) == 0:
            return pd.Series(dtype=float)
        factor_series.sort_values(inplace=True)
        choose = int(floor(self.trading_signal * len(factor_series)))
        if choose < 1:
            choose = 1
        target_capital = self.daily_capital / (2 * choose)
        signal_series = pd.Series(0, index=factor_series.index, dtype=float)
        if self.long_low:
            signal_series.iloc[:choose] = 1
            signal_series.iloc[-choose:] = -1
        else:
            signal_series.iloc[:choose] = -1
            signal_series.iloc[-choose:] = 1
        target_data = {}
        for dominant_symbol, signal in signal_series.items():
            if signal == 0:
                continue
            try:
                vt_symbol, size = self.dm.calculate_trading_volume(
                    self.current_dt, dominant_symbol, target_capital
                )
                if size > 0:
                    target_data[vt_symbol] = int(signal * size)
            except Exception:
                pass
        return pd.Series(target_data)

backtester = StrategyBacktester(
    vt_symbols=symbols,
    interval=Interval.DAILY,
    start=start,
    end=end,
    capital=10_000_000,
)
backtester.load_data()

for variant_name in ['raw', 'roll5']:
    carry_df = variants[variant_name]
    setting = {
        "holding_period": 5,
        "trading_signal": 0.2,
        "long_low": False,
        "aggregation": "sum",
        "factor_name": "",
        "factor_data": carry_df,
    }
    
    target_df = backtester.run_backtesting(CarryVariantStrategy, setting)
    
    # Stats
    non_zero = (target_df != 0).sum().sum()
    total_cells = target_df.size
    pos = (target_df > 0).sum().sum()
    neg = (target_df < 0).sum().sum()
    
    # Daily position changes
    daily_pos = target_df.abs().sum(axis=1)
    avg_daily_pos = daily_pos.mean()
    max_daily_pos = daily_pos.max()
    
    # Changes per day
    changes = (target_df.diff().fillna(0) != 0).sum().sum()
    
    print(f"\n=== {variant_name} ===")
    print(f"  target_df shape: {target_df.shape}")
    print(f"  non-zero entries: {non_zero} / {total_cells} ({non_zero/total_cells*100:.1f}%)")
    print(f"  positive: {pos}, negative: {neg}")
    print(f"  avg daily position (contracts): {avg_daily_pos:.1f}")
    print(f"  max daily position (contracts): {max_daily_pos:.1f}")
    print(f"  total position changes: {changes}")
    print(f"  avg daily change: {changes / len(target_df):.1f}")
