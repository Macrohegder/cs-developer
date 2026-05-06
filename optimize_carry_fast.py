#!/usr/bin/env python3
"""
Carry 因子参数快速优化 — 向量化回测（88价格，无合约映射）
正确模拟 CrossSectionalStrategy 的 holding_period 分仓机制。
"""

import sys
sys.path.insert(0, "/root/cs_developer")

import pandas as pd
import numpy as np
from datetime import datetime
from math import floor

# 加载预计算的 carry 因子
carry = pd.read_csv('/root/cs_developer/result_carry_exact_20_factor.csv', index_col=0)
carry.index = pd.to_datetime(carry.index)

# 加载88收盘价
from vnpy_alpharesearch.utility import load_bar_df
from vnpy.trader.constant import Interval

start = datetime(2015, 1, 1)
end = datetime(2024, 12, 31)

close_dict = {}
for sym in carry.columns:
    try:
        df = load_bar_df(sym, Interval.DAILY, start, end)
        if hasattr(df.index, 'tz_localize'):
            df.index = df.index.tz_localize(None)
        close_dict[sym] = df['close_price']
    except:
        pass

close_df = pd.DataFrame(close_dict)

# 对齐
carry = carry.loc[carry.index.isin(close_df.index)]
close_df = close_df.loc[close_df.index.isin(carry.index)]

# 计算日收益率
ret = close_df.pct_change()

# 参数网格
holding_periods = [5, 10, 20]
trading_signals = [0.1, 0.2, 0.3]
capital = 10_000_000

print("="*75)
print("Carry 因子参数快速优化（向量化回测，88价格，含分仓机制）")
print("="*75)
print(f"区间: {carry.index[0].date()} ~ {carry.index[-1].date()}")
print(f"品种数: {len(carry.columns)}")
print(f"交易日: {len(carry)}")
print()

results = []

for hp in holding_periods:
    for ts in trading_signals:
        # 模拟 CrossSectionalStrategy 的分仓机制
        # 每天有 capital/hp 的资金开新仓，持有 hp 天后移除
        daily_capital = capital / hp
        
        # 预计算每日信号
        daily_signals = {}  # {日期: {品种: 信号(+1/-1)}}
        
        for dt in carry.index:
            row = carry.loc[dt].dropna()
            if len(row) < 10:
                continue
            row_sorted = row.sort_values()
            choose = max(1, int(floor(ts * len(row_sorted))))
            
            signals = {}
            for sym in row_sorted.index[-choose:]:
                signals[sym] = 1  # 高carry做多
            for sym in row_sorted.index[:choose]:
                signals[sym] = -1  # 低carry做空
            
            daily_signals[dt] = signals
        
        # 计算每日盈亏（分仓叠加）
        pnl_list = []
        position_history = {}  # {开仓日期: {品种: 名义价值权重}}
        
        valid_dates = [d for d in carry.index if d in close_df.index and d in ret.index]
        
        for i, dt in enumerate(valid_dates):
            # 1. 新开今日仓位
            if dt in daily_signals:
                signals = daily_signals[dt]
                choose = len(signals) // 2  # 每端品种数
                if choose < 1:
                    choose = 1
                target_value = daily_capital / (2 * choose)
                
                new_pos = {}
                for sym, signal in signals.items():
                    if sym in close_df.columns:
                        price = close_df.loc[dt, sym]
                        if price > 0 and not pd.isna(price):
                            # 手数（简化，假设每手1单位）
                            lots = target_value / price
                            new_pos[sym] = signal * lots
                
                position_history[dt] = new_pos
            
            # 2. 移除过期持仓（超过 hp 天）
            expiry = dt - pd.Timedelta(days=hp*3)  # 足够早的日期
            keys_to_remove = [k for k in list(position_history.keys()) if k <= expiry]
            for k in keys_to_remove:
                del position_history[k]
            
            # 3. 计算今日盈亏
            if i == 0:
                pnl_list.append({'date': dt, 'pnl': 0})
                continue
            
            prev_dt = valid_dates[i-1]
            if prev_dt not in close_df.index:
                pnl_list.append({'date': dt, 'pnl': 0})
                continue
            
            total_pnl = 0
            for pos_dt, positions in position_history.items():
                if pos_dt >= dt:  # 今日新开仓不算盈亏
                    continue
                for sym, lots in positions.items():
                    if sym in ret.columns:
                        r = ret.loc[dt, sym]
                        if not pd.isna(r):
                            # 盈亏 = 手数 × 前收盘价 × 收益率
                            prev_price = close_df.loc[prev_dt, sym]
                            pnl = lots * prev_price * r
                            total_pnl += pnl
            
            pnl_list.append({'date': dt, 'pnl': total_pnl})
        
        pnl_df = pd.DataFrame(pnl_list).set_index('date')
        cum_pnl = pnl_df['pnl'].cumsum()
        total_return = cum_pnl.iloc[-1] / capital
        
        # 年化
        n_days = len(pnl_df)
        annual_return = total_return / n_days * 240
        
        # 夏普
        daily_ret = pnl_df['pnl'] / capital
        daily_ret = daily_ret[daily_ret != 0]
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(240) if daily_ret.std() > 0 else 0
        
        # 最大回撤
        cumwealth = 1 + cum_pnl / capital
        running_max = cumwealth.cummax()
        drawdown = (cumwealth - running_max) / running_max
        max_dd = drawdown.min()
        
        # 胜率
        win_rate = (daily_ret > 0).sum() / len(daily_ret) if len(daily_ret) > 0 else 0
        
        results.append({
            'hp': hp,
            'ts': ts,
            'total_ret': total_return,
            'annual_ret': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
        })
        
        print(f"hp={hp:>2}  ts={ts:.0%}  |  总收益={total_return:>7.2%}  年化={annual_return:>7.2%}  夏普={sharpe:>6.2f}  回撤={max_dd:>7.2%}  胜率={win_rate:.1%}")

# 排序
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('annual_ret', ascending=False)

print()
print("="*75)
print("按年化收益排序")
print("="*75)
print(f"{'持仓':>4} {'比例':>5} {'总收益':>8} {'年化':>8} {'夏普':>6} {'最大回撤':>8} {'胜率':>6}")
print("-"*75)
for _, row in results_df.iterrows():
    print(f"{row['hp']:>4} {row['ts']:>5.0%} {row['total_ret']:>7.1%} {row['annual_ret']:>7.1%} {row['sharpe']:>6.2f} {row['max_dd']:>7.1%} {row['win_rate']:>5.1%}")

best = results_df.iloc[0]
worst = results_df.iloc[-1]
print()
print("="*75)
print(f"最优参数: 持仓周期={int(best['hp'])}天, 交易比例={best['ts']:.0%}")
print(f"  总收益: {best['total_ret']:.2%}")
print(f"  年化: {best['annual_ret']:.2%}")
print(f"  夏普: {best['sharpe']:.2f}")
print(f"  最大回撤: {best['max_dd']:.2%}")
print()
print(f"最差参数: 持仓周期={int(worst['hp'])}天, 交易比例={worst['ts']:.0%}")
print(f"  总收益: {worst['total_ret']:.2%}")
print(f"  年化: {worst['annual_ret']:.2%}")
print("="*75)

results_df.to_csv('/root/cs_developer/carry_fast_optimization.csv', index=False)
print("\n[OK] 结果已保存到 carry_fast_optimization.csv")
