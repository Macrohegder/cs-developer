#!/usr/bin/env python3
"""
Carry 无 rolling 对照实验 — 验证高夏普的来源

使用与 A 部分完全相同的设置：
- 21 品种（原始框架品种列表）
- 期间 2011-01-01 ~ 2023-09-15
- carry = (f2-f1)/f2 / days * 365，无 rolling 平滑
- hp=1, ts=40%
- target_capital = daily_capital / (2 * n * ts) 固定分母
"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
from datetime import datetime
from math import floor
from clickhouse_driver import Client

from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

from strategies.cross_sectional_strategy import CrossSectionalStrategy


DEFAULT_SYMBOLS = [
    'A88.DCE', 'AG88.SHFE', 'AL88.SHFE', 'AU88.SHFE', 'B88.DCE',
    'BU88.SHFE', 'C88.DCE', 'CF88.CZCE', 'CU88.SHFE', 'FG88.CZCE',
    'FU88.SHFE', 'J88.DCE', 'L88.DCE', 'M88.DCE', 'P88.DCE',
    'PB88.SHFE', 'RB88.SHFE', 'RU88.SHFE', 'SR88.CZCE', 'Y88.DCE',
    'ZN88.SHFE',
]


def compute_carry_raw(symbols, start, end):
    """计算无 rolling 的原始 carry = (f2-f1)/f2 / days * 365"""
    close_88 = {}
    close_a2 = {}
    valid_symbols = []
    for sym in symbols:
        try:
            df = load_bar_df(sym, Interval.DAILY, start, end)
            if hasattr(df.index, 'tz_localize'):
                df.index = df.index.tz_localize(None)
            close_88[sym] = df['close_price']
            
            sym_a2 = sym.replace('88.', '88A2.')
            df2 = load_bar_df(sym_a2, Interval.DAILY, start, end)
            if hasattr(df2.index, 'tz_localize'):
                df2.index = df2.index.tz_localize(None)
            close_a2[sym] = df2['close_price']
            valid_symbols.append(sym)
        except:
            pass
    
    close_88_df = pd.DataFrame(close_88)
    close_a2_df = pd.DataFrame(close_a2)
    
    # 加载主力映射
    client = Client(host='localhost')
    keys = []
    for sym in valid_symbols:
        keys.append(f"{sym}@1")
        keys.append(f"{sym}@2")
    
    result = client.execute(
        'SELECT datetime, key, value FROM vnpy.vnpy_dominant_contract '
        'WHERE key IN %(keys)s AND datetime >= %(start)s AND datetime <= %(end)s '
        'ORDER BY datetime',
        {'keys': keys, 'start': start, 'end': end}
    )
    map_df = pd.DataFrame(result, columns=['datetime', 'key', 'value'])
    map_df['datetime'] = pd.to_datetime(map_df['datetime'])
    map_df = map_df.drop_duplicates(subset=['datetime', 'key'])
    map_pivot = map_df.pivot(index='datetime', columns='key', values='value')
    
    # 获取合约到期日
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    
    all_contracts = set()
    for sym in valid_symbols:
        k1 = f"{sym}@1"
        k2 = f"{sym}@2"
        if k1 in map_pivot.columns:
            all_contracts.update(map_pivot[k1].dropna().unique())
        if k2 in map_pivot.columns:
            all_contracts.update(map_pivot[k2].dropna().unique())
    
    contract_enddates = {}
    for c in all_contracts:
        c_base = c.split('.')[0]
        rows = contract_df.loc[contract_df.index == c_base, 'enddate']
        if len(rows) > 0:
            contract_enddates[c] = pd.to_datetime(rows.values[0])
    
    # 计算到期日差距
    days_to_expiry = pd.DataFrame(index=map_pivot.index, columns=valid_symbols, dtype=float)
    for sym in valid_symbols:
        k1 = f"{sym}@1"
        k2 = f"{sym}@2"
        if k1 not in map_pivot.columns or k2 not in map_pivot.columns:
            continue
        c1_series = map_pivot[k1]
        c2_series = map_pivot[k2]
        for dt in map_pivot.index:
            c1 = c1_series.get(dt)
            c2 = c2_series.get(dt)
            if pd.isna(c1) or pd.isna(c2):
                continue
            d1 = contract_enddates.get(c1)
            d2 = contract_enddates.get(c2)
            if d1 and d2:
                days = (d2 - d1).days
                if days > 0:
                    days_to_expiry.loc[dt, sym] = days
    
    # 计算 carry = (f2-f1)/f2 / days * 365
    common_idx = close_88_df.index.intersection(close_a2_df.index).intersection(days_to_expiry.index)
    f1 = close_88_df.loc[common_idx]
    f2 = close_a2_df.loc[common_idx]
    days = days_to_expiry.loc[common_idx]
    
    carry = (f2 - f1) / f2 / days * 365.0
    print(f"[Carry Raw] 有效品种: {len(valid_symbols)}, carry 均值: {carry.mean().mean():.4f}, 标准差: {carry.std().mean():.4f}")
    return carry


class CarryRawStrategy(CrossSectionalStrategy):
    def calculate_daily_target_series(self, df=None):
        dt = self.current_dt
        if dt in self.factor_data.index:
            factor_series = self.factor_data.loc[dt].copy()
        else:
            return pd.Series(dtype=float)
        
        factor_series = factor_series.dropna()
        if len(factor_series) == 0:
            return pd.Series(dtype=float)
        
        factor_series.sort_values(inplace=True)
        
        # 固定分母，与原始框架一致
        n_symbols = len(factor_series)
        target_capital = self.daily_capital / (2 * n_symbols * self.trading_signal)
        
        choose = int(floor(self.trading_signal * n_symbols))
        if choose < 1:
            choose = 1
        
        signal_series = pd.Series(0, index=factor_series.index, dtype=float)
        signal_series.iloc[:choose] = 1
        signal_series.iloc[-choose:] = -1
        
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


def main():
    start = datetime(2011, 1, 1)
    end = datetime(2023, 9, 15)
    capital = 10_000_000
    commission = 0.0001
    
    print("="*60)
    print("Carry 无 rolling 对照实验")
    print("="*60)
    
    # 1. 计算无 rolling carry
    carry_df = compute_carry_raw(DEFAULT_SYMBOLS, start, end)
    
    # 2. 回测
    print(f"\n[回测] hp=1, ts=40%, 21品种, 无rolling, 2011-2023")
    backtester = StrategyBacktester(
        vt_symbols=carry_df.columns.tolist(),
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=capital,
    )
    backtester.load_data()
    
    setting = {
        "holding_period": 1,
        "trading_signal": 0.4,
        "long_low": True,
        "aggregation": "sum",
        "factor_name": "",
        "factor_data": carry_df,
    }
    
    target_df = backtester.run_backtesting(CarryRawStrategy, setting)
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=False,
    )
    
    stats = result["statistics"]
    print("\n" + "="*60)
    print("Carry 无 rolling 回测结果 (21品种, hp=1, ts=40%)")
    print("="*60)
    print(f"  回测区间: {stats['start_date'].date()} ~ {stats['end_date'].date()}")
    print(f"  总收益: {stats['total_return']}")
    print(f"  年化收益: {stats['annual_return']}")
    print(f"  夏普比率: {stats['sharpe_ratio']}")
    print(f"  最大回撤: {stats['max_ddpercent']}")
    print(f"  卡玛比率: {stats['calmar_ratio']}")
    print("="*60)


if __name__ == "__main__":
    main()
