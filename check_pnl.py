#!/usr/bin/env python3
"""直接比较 raw vs roll5 的 portfolio performance（无monkey-patch）"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
from datetime import datetime
from math import floor

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

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

for variant_name in ['raw', 'roll5']:
    print(f"\n{'='*50}")
    print(f"Testing {variant_name}, hp=5, ts=20%")
    print(f"{'='*50}")
    
    backtester = StrategyBacktester(
        vt_symbols=symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=10_000_000,
    )
    backtester.load_data()
    
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
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=0.0001,
        capital=10_000_000,
        plot_chart=False,
    )
    
    stats = result["statistics"]
    print(f"  total_return: {stats['total_return']}")
    print(f"  annual_return: {stats['annual_return']}")
    print(f"  sharpe: {stats['sharpe_ratio']}")
    print(f"  max_dd: {stats['max_ddpercent']}")
    print(f"  calmar: {stats['calmar_ratio']}")
