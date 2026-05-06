#!/usr/bin/env python3
"""
Carry 因子优化 — 37品种, hp=1 测试 + 细粒度参数网格

加载已有的 carry_20_exact.csv，在 37 品种 2015-2024 上测试：
1. hp=1, ts=40% (与原始框架对齐)
2. 细粒度网格: hp=[1,2,3,5] × ts=[0.05,0.1,0.2,0.3,0.4]
"""

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


class CarryFastStrategy(CrossSectionalStrategy):
    """直接读取预计算因子的策略"""
    
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


def run_single_backtest(carry_df, hp, ts, capital=10_000_000, commission=0.0001, long_low=False):
    """运行单个参数组合的回测"""
    start = datetime(2015, 1, 1)
    end = datetime(2024, 12, 31)
    symbols = carry_df.columns.tolist()
    
    backtester = StrategyBacktester(
        vt_symbols=symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=capital,
    )
    backtester.load_data()
    
    setting = {
        "holding_period": hp,
        "trading_signal": ts,
        "long_low": long_low,
        "aggregation": "sum",
        "factor_name": "",
        "factor_data": carry_df,
    }
    
    target_df = backtester.run_backtesting(CarryFastStrategy, setting)
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=False,
    )
    
    stats = result["statistics"]
    return {
        "hp": hp,
        "ts": ts,
        "long_low": long_low,
        "total_return": stats["total_return"],
        "annual_return": stats["annual_return"],
        "sharpe": stats["sharpe_ratio"],
        "max_drawdown": stats["max_ddpercent"],
        "calmar": stats["calmar_ratio"],
        "profit_days": stats.get("profit_days", 0),
        "loss_days": stats.get("loss_days", 0),
    }


def main():
    print("="*60)
    print("Carry 因子优化 — 37品种, hp=1 测试 + 细粒度参数网格")
    print("="*60)
    
    # 1. 加载预计算的 carry 因子
    print("\n[1/3] 加载 carry_20_exact.csv...")
    carry_df = pd.read_csv("/root/cs_developer/carry_20_exact.csv", index_col=0)
    carry_df.index = pd.to_datetime(carry_df.index)
    carry_df = carry_df.loc["2015-01-01":"2024-12-31"]
    print(f"  加载完成: {carry_df.shape}")
    
    results = []
    
    # 2. 先测试 hp=1, ts=40% (与原始框架对齐)
    print("\n[2/3] 测试 hp=1, ts=40% (与原始框架对齐)...")
    r = run_single_backtest(carry_df, hp=1, ts=0.4, long_low=False)
    results.append(r)
    print(f"  结果: total={r['total_return']}, annual={r['annual_return']}, sharpe={r['sharpe']}, max_dd={r['max_drawdown']}")
    
    # 3. 细粒度参数网格
    print("\n[3/3] 细粒度参数网格搜索...")
    hps = [1, 2, 3, 5]
    tss = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    total = len(hps) * len(tss)
    for i, hp in enumerate(hps):
        for j, ts in enumerate(tss):
            if hp == 1 and ts == 0.4:
                continue  # 已测
            idx = i * len(tss) + j + 1
            print(f"  [{idx}/{total}] hp={hp}, ts={ts}...", end=" ", flush=True)
            try:
                r = run_single_backtest(carry_df, hp=hp, ts=ts, long_low=False)
                results.append(r)
                print(f"OK → total={r['total_return']}, sharpe={r['sharpe']}, max_dd={r['max_drawdown']}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "hp": hp, "ts": ts, "long_low": False,
                    "total_return": "ERROR", "annual_return": "",
                    "sharpe": "", "max_drawdown": "", "calmar": "",
                    "profit_days": 0, "loss_days": 0,
                })
    
    # 4. 保存结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="sharpe", ascending=False)
    results_df.to_csv("/root/cs_developer/carry_hp1_optimization.csv", index=False)
    
    print("\n" + "="*60)
    print("Carry 因子优化结果 (按 Sharpe 排序)")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    print("\n结果已保存到 carry_hp1_optimization.csv")


if __name__ == "__main__":
    main()
