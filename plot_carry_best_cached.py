#!/usr/bin/env python3
"""重新跑最优carry配置并保存净值曲线图（使用cached load_bar_df）"""
import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
import numpy as np
import pickle
from math import floor
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import (
    calculate_portfolio_pnl, calculate_overall_pnl, calculate_product_pnl
)
from strategies.cross_sectional_strategy import CrossSectionalStrategy
import vnpy_alpharesearch.utility as utility
import functools

# Monkey-patch load_bar_df with cache
original_load_bar_df = utility.load_bar_df
_bar_cache = {}

@functools.lru_cache(maxsize=5000)
def cached_load_bar_df(vt_symbol, interval, start, end):
    if hasattr(start, 'strftime'):
        start_dt = start
    else:
        start_dt = pd.to_datetime(start)
    if hasattr(end, 'strftime'):
        end_dt = end
    else:
        end_dt = pd.to_datetime(end)
    key = (vt_symbol, str(interval), start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))
    if key in _bar_cache:
        return _bar_cache[key]
    df = original_load_bar_df(vt_symbol, interval, start_dt, end_dt)
    _bar_cache[key] = df
    return df

utility.load_bar_df = cached_load_bar_df

with open('/root/cs_developer/carry_variants_cache.pkl', 'rb') as f:
    variants = pickle.load(f)

carry_df = variants['raw']
all_symbols = carry_df.columns.tolist()

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

print("="*60)
print("Plotting Carry Best (hp=1, ts=40%, raw)")
print("="*60)

backtester = StrategyBacktester(
    vt_symbols=all_symbols,
    interval=Interval.DAILY,
    start=datetime(2011, 1, 1),
    end=datetime(2026, 3, 31),
    capital=10_000_000,
)
backtester.load_data()

setting = {
    "holding_period": 1,
    "trading_signal": 0.4,
    "long_low": False,
    "aggregation": "sum",
    "factor_name": "",
    "factor_data": carry_df,
}

target_df = backtester.run_backtesting(CarryVariantStrategy, setting)
print(f"Target shape: {target_df.shape}")

# 计算净值曲线
portfolio_df = calculate_portfolio_pnl(target_df, Interval.DAILY, commission=0.0001)
overall_df = calculate_overall_pnl(portfolio_df, capital=10_000_000)
product_df = calculate_product_pnl(portfolio_df)

# 统计指标
start_bal = overall_df['balance'].iloc[0]
end_bal = overall_df['balance'].iloc[-1]
total_return = end_bal / start_bal - 1
days = len(overall_df)
annual_return = total_return / days * 240
daily_ret = overall_df['return'].mean()
daily_std = overall_df['return'].std()
sharpe = (daily_ret / daily_std) * np.sqrt(240) if daily_std > 0 else 0
max_dd = overall_df['ddpercent'].min()

print(f"\n{'='*50}")
print(f"Carry Best (hp=1, ts=40%, raw)")
print(f"{'='*50}")
print(f"  total_return:  {total_return:.2%}")
print(f"  annual_return: {annual_return:.2%}")
print(f"  sharpe:        {sharpe:.2f}")
print(f"  max_dd:        {max_dd:.2%}")
print(f"{'='*50}")

# 保存净值曲线图
fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
fig.suptitle(f'Carry Strategy (hp=1, ts=40%, raw) 2011-2026\nSharpe={sharpe:.2f}, Total={total_return:.1%}, MaxDD={max_dd:.1%}', 
             fontsize=12, fontweight='bold')

# 1. 净值曲线
ax = axes[0]
ax.plot(overall_df.index, overall_df['balance'], label='Balance', color='#1f77b4', linewidth=1.2)
ax.plot(overall_df.index, overall_df['highlevel'], label='High Water', color='#ff7f0e', linewidth=0.8, linestyle='--')
ax.fill_between(overall_df.index, overall_df['balance'], overall_df['highlevel'], alpha=0.3, color='red')
ax.set_ylabel('Balance')
ax.legend(loc='upper left')
ax.set_title('Equity Curve')
ax.grid(True, alpha=0.3)

# 2. 回撤
ax = axes[1]
ax.fill_between(overall_df.index, overall_df['drawdown'], 0, alpha=0.5, color='red')
ax.plot(overall_df.index, overall_df['drawdown'], color='darkred', linewidth=0.8)
ax.set_ylabel('Drawdown')
ax.set_title('Drawdown')
ax.grid(True, alpha=0.3)

# 3. 每日盈亏
ax = axes[2]
colors = ['green' if v >= 0 else 'red' for v in overall_df['net_pnl']]
ax.bar(overall_df.index, overall_df['net_pnl'], color=colors, alpha=0.6, width=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Daily PnL')
ax.set_title('Daily Profit & Loss')
ax.grid(True, alpha=0.3)

# 4. 品种累计盈亏（前10）
ax = axes[3]
top_products = product_df.iloc[-1].abs().nlargest(10).index
for product in top_products:
    ax.plot(product_df.index, product_df[product], label=product, linewidth=0.8, alpha=0.8)
ax.set_ylabel('Cum PnL')
ax.set_title('Top 10 Products Cumulative PnL')
ax.legend(loc='upper left', ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/cs_developer/carry_best_equity.png', dpi=150, bbox_inches='tight')
print("\nSaved: carry_best_equity.png")
