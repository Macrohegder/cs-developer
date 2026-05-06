#!/usr/bin/env python3
"""
Carry 因子 rolling 对比实验 - 缓存加速版（完成剩余回测）
"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import os
import pandas as pd
from datetime import datetime
from math import floor
from typing import List

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance
from vnpy_alpharesearch.utility import load_bar_df as original_load_bar_df

from strategies.cross_sectional_strategy import CrossSectionalStrategy

# Monkey-patch load_bar_df with LRU cache
_bar_cache = {}
def cached_load_bar_df(vt_symbol, interval, start, end):
    from pandas import Timestamp
    import numpy as np
    # Convert numpy int / python int nanoseconds to Timestamp
    if isinstance(start, (int, np.integer)):
        start = Timestamp(start)
    if isinstance(end, (int, np.integer)):
        end = Timestamp(end)
    # Ensure pandas Timestamp for key generation
    if hasattr(start, 'strftime'):
        start_key = start.strftime('%Y%m%d')
    else:
        start_key = str(start)
    if hasattr(end, 'strftime'):
        end_key = end.strftime('%Y%m%d')
    else:
        end_key = str(end)
    key = (vt_symbol, str(interval), start_key, end_key)
    if key not in _bar_cache:
        _bar_cache[key] = original_load_bar_df(vt_symbol, interval, start, end)
    return _bar_cache[key]

import vnpy_alpharesearch.strategy.analysis as analysis_module
analysis_module.__dict__['load_bar_df'] = cached_load_bar_df

# Also patch in utility module for any other imports
import vnpy_alpharesearch.utility as utility_module
utility_module.load_bar_df = cached_load_bar_df

DEFAULT_SYMBOLS = [
    'A88.DCE', 'AG88.SHFE', 'AL88.SHFE', 'AU88.SHFE',
    'B88.DCE', 'BU88.SHFE', 'C88.DCE', 'CF88.CZCE',
    'CS88.DCE', 'CU88.SHFE', 'EB88.DCE', 'EG88.DCE',
    'FG88.CZCE', 'FU88.SHFE', 'HC88.SHFE', 'I88.DCE',
    'J88.DCE', 'JD88.DCE', 'JM88.DCE', 'L88.DCE',
    'M88.DCE', 'MA88.CZCE', 'NI88.SHFE', 'OI88.CZCE',
    'P88.DCE', 'PB88.SHFE', 'PP88.DCE', 'RB88.SHFE',
    'RM88.CZCE', 'RU88.SHFE', 'SF88.CZCE', 'SM88.CZCE',
    'SN88.SHFE', 'SR88.CZCE', 'V88.DCE', 'Y88.DCE',
    'ZN88.SHFE'
]


class CarryVariantStrategy(CrossSectionalStrategy):
    """Carry 变体策略：从 setting 中读取预计算的 factor_data"""
    
    def __init__(self, vt_symbols, interval, start, end, capital, setting, output):
        super().__init__(vt_symbols, interval, start, end, capital, setting, output)
        self.factor_data = setting.get("factor_data")
        self.factor_name = setting.get("factor_name", "")
    
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


def run_variant_backtest(backtester, carry_df, variant_name, hp, ts, capital=10_000_000, commission=0.0001):
    setting = {
        "holding_period": hp,
        "trading_signal": ts,
        "long_low": False,
        "aggregation": "sum",
        "factor_name": "",
        "factor_data": carry_df,
    }
    
    target_df = backtester.run_backtesting(CarryVariantStrategy, setting)
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=False,
    )
    
    stats = result["statistics"]
    return {
        "variant": variant_name,
        "hp": hp,
        "ts": ts,
        "total_return": stats["total_return"],
        "annual_return": stats["annual_return"],
        "sharpe": stats["sharpe_ratio"],
        "max_drawdown": stats["max_ddpercent"],
        "calmar": stats["calmar_ratio"],
    }


def main():
    print("="*60)
    print("Carry 因子 rolling 对比实验 - 缓存加速完成剩余回测")
    print("="*60)
    
    start = datetime(2011, 1, 1)
    end = datetime(2026, 3, 31)
    
    # 1. 加载 carry 变体
    cache_file = "/root/cs_developer/carry_variants_cache.pkl"
    import pickle
    with open(cache_file, 'rb') as f:
        variants = pickle.load(f)
    for name, df in variants.items():
        print(f"  carry_{name}: {df.shape}")
    
    # 2. 预创建 backtester 并加载数据
    print("\n[预加载数据] 创建回测引擎并加载所有 K 线...")
    all_symbols = list(variants.values())[0].columns.tolist()
    backtester = StrategyBacktester(
        vt_symbols=all_symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=10_000_000,
    )
    backtester.load_data()
    print("[数据加载完成]")
    
    # 3. 只运行剩余的两个回测
    remaining = [
        ("hp=5, ts=20%", 5, 0.2, "roll5"),
        ("hp=5, ts=20%", 5, 0.2, "roll20"),
    ]
    
    results = []
    
    for config_name, hp, ts, variant_name in remaining:
        carry_df = variants[variant_name]
        print(f"\n  carry_{variant_name} (hp={hp}, ts={ts}) ...", end=" ", flush=True)
        r = run_variant_backtest(backtester, carry_df, variant_name, hp, ts)
        r['config'] = config_name
        results.append(r)
        print(f"OK → total={r['total_return']}, sharpe={r['sharpe']}, max_dd={r['max_drawdown']}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("/root/cs_developer/carry_rolling_compare_remaining.csv", index=False)
    print("\n结果已保存到 carry_rolling_compare_remaining.csv")


if __name__ == "__main__":
    main()
