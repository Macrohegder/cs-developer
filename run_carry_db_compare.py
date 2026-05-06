#!/usr/bin/env python3
"""
使用数据库中预计算的 carry 因子（cross_sectional author）回测，
与自定义计算的 carry 对比，排查收益差异来源。
"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
from datetime import datetime
from math import floor

from vnpy.trader.constant import Interval
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

from strategies.cross_sectional_strategy import CrossSectionalStrategy


# 原始 notebook 的 51 品种
DOMINANT_SYMBOLS = [
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
    'ZN88.SHFE',
]


class CarryDBStrategy(CrossSectionalStrategy):
    """直接读取预计算因子数据的策略"""
    
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
        
        choose = int(floor(self.trading_signal * len(factor_series)))
        if choose < 1:
            choose = 1
        
        # 使用原始 script/carry_strategy.py 的 target_capital 逻辑
        n_symbols = len(factor_series)
        target_capital = self.daily_capital / (2 * n_symbols * self.trading_signal)
        
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


def run_db_carry_backtest():
    start = datetime(2011, 1, 1)
    end = datetime(2023, 9, 21)
    capital = 30_000_000
    commission = 0.0001
    hp = 1
    ts = 0.4
    
    # 加载数据库中的 carry 因子
    dc = DataCenter()
    carry_df = dc.load_factor_df(DOMINANT_SYMBOLS, 'carry', 'd', '', 'cross_sectional', start, end)
    print(f"Loaded DB carry factor: {carry_df.shape}")
    print(f"Date range: {carry_df.index[0]} ~ {carry_df.index[-1]}")
    
    # 过滤有效品种
    valid_symbols = [s for s in DOMINANT_SYMBOLS if s in carry_df.columns]
    carry_df = carry_df[valid_symbols]
    print(f"Valid symbols: {len(valid_symbols)}")
    
    backtester = StrategyBacktester(
        vt_symbols=valid_symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=capital,
    )
    backtester.load_data()
    
    setting = {
        "holding_period": hp,
        "trading_signal": ts,
        "long_low": True,
        "aggregation": "sum",
        "factor_name": "",
        "factor_data": carry_df,
    }
    
    target_df = backtester.run_backtesting(CarryDBStrategy, setting)
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=False,
    )
    
    stats = result["statistics"]
    print("\n" + "="*60)
    print("DB Carry 因子回测结果 (51品种, 30M, hp=1, ts=40%)")
    print("="*60)
    print(f"  回测区间: {stats['start_date'].date()} ~ {stats['end_date'].date()}")
    print(f"  总收益: {stats['total_return']}")
    print(f"  年化收益: {stats['annual_return']}")
    print(f"  夏普比率: {stats['sharpe_ratio']}")
    print(f"  最大回撤: {stats['max_ddpercent']}")
    print(f"  卡玛比率: {stats['calmar_ratio']}")
    print("="*60)


if __name__ == "__main__":
    run_db_carry_backtest()
