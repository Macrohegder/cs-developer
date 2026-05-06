#!/usr/bin/env python3
"""
Carry 因子参数优化 — 网格搜索持仓周期和交易比例

用法:
    python optimize_carry_params.py --start 2015-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from math import floor

import pandas as pd
from pandas import Series, DataFrame, concat
import numpy as np
from clickhouse_driver import Client

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

from strategies.cross_sectional_strategy import CrossSectionalStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Carry 因子参数优化")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--capital", type=int, default=10_000_000)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--holding-periods", type=str, default="5,10,20", help="逗号分隔的持仓周期")
    parser.add_argument("--trading-signals", type=str, default="0.1,0.2,0.3", help="逗号分隔的交易比例")
    parser.add_argument("--cycle", type=int, default=20, help="carry 滚动平均天数")
    return parser.parse_args()


def load_carry_factor(cycle: int, start: datetime, end: datetime) -> DataFrame:
    """加载预计算的 carry 因子"""
    path = f"/root/cs_developer/result_carry_exact_{cycle}_factor.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.loc[start:end]
        print(f"[Load] 从 {path} 加载 carry 因子 ({len(df)} 行)")
        return df
    raise FileNotFoundError(f"carry 因子文件不存在: {path}")


class CarryGridStrategy(CrossSectionalStrategy):
    """支持外部传入 factor_data 的 Carry 策略"""
    
    def calculate_daily_target_series(self, df=None):
        dt = self.current_dt
        if dt in self.factor_data.index:
            factor_series = self.factor_data.loc[dt].copy()
        else:
            return Series(dtype=float)
        
        factor_series = factor_series.dropna()
        if len(factor_series) == 0:
            return Series(dtype=float)
        
        factor_series.sort_values(inplace=True)
        
        choose = int(floor(self.trading_signal * len(factor_series)))
        if choose < 1:
            choose = 1
        
        target_capital = self.daily_capital / (2 * choose)
        signal_series = Series(0, index=factor_series.index, dtype=float)
        
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
                    self.current_dt,
                    dominant_symbol,
                    target_capital
                )
                if size > 0:
                    target_data[vt_symbol] = int(signal * size)
            except Exception:
                pass
        
        return Series(target_data)


def run_single_backtest(
    backtester: StrategyBacktester,
    carry_df: DataFrame,
    capital: int,
    commission: float,
    holding_period: int,
    trading_signal: float,
) -> Dict:
    """运行单组参数回测（复用已加载数据的 backtester）"""
    setting = {
        "holding_period": holding_period,
        "trading_signal": trading_signal,
        "long_low": False,
        "aggregation": "sum",
        "factor_name": "",
        "factor_data": carry_df,
    }
    
    target_df = backtester.run_backtesting(CarryGridStrategy, setting)
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=False,
    )
    
    return result["statistics"]


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    holding_periods = [int(x) for x in args.holding_periods.split(",")]
    trading_signals = [float(x) for x in args.trading_signals.split(",")]
    
    # 加载 carry 因子
    carry_df = load_carry_factor(args.cycle, start, end)
    symbols = carry_df.columns.tolist()
    
    print(f"\n{'='*70}")
    print(f"Carry 因子参数优化")
    print(f"{'='*70}")
    print(f"区间: {start.date()} ~ {end.date()}")
    print(f"品种数: {len(symbols)}")
    print(f"持仓周期: {holding_periods}")
    print(f"交易比例: {trading_signals}")
    print(f"佣金: {args.commission}")
    print(f"{'='*70}\n")
    
    # 预加载数据（只加载一次）
    print("\n[Preload] 加载回测数据...")
    backtester = StrategyBacktester(
        vt_symbols=symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=args.capital,
    )
    backtester.load_data()
    print("[Preload] 数据加载完成\n")
    
    results = []
    total = len(holding_periods) * len(trading_signals)
    count = 0
    
    for hp in holding_periods:
        for ts in trading_signals:
            count += 1
            print(f"[{count}/{total}] testing holding_period={hp}, trading_signal={ts:.0%}...", end=" ", flush=True)
            
            stats = run_single_backtest(
                backtester=backtester,
                carry_df=carry_df,
                capital=args.capital,
                commission=args.commission,
                holding_period=hp,
                trading_signal=ts,
            )
            
            # 解析统计值
            total_ret = float(stats['total_return'].rstrip('%')) / 100
            annual_ret = float(stats['annual_return'].rstrip('%')) / 100
            sharpe = float(stats['sharpe_ratio'])
            max_dd = float(stats['max_ddpercent'].rstrip('%')) / 100
            
            results.append({
                'holding_period': hp,
                'trading_signal': ts,
                'total_return': total_ret,
                'annual_return': annual_ret,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'profit_days': stats['profit_days'],
                'loss_days': stats['loss_days'],
                'calmar_ratio': float(stats['calmar_ratio']),
            })
            
            print(f"年化={annual_ret*100:+.2f}%, 夏普={sharpe:.2f}, 回撤={max_dd*100:.1f}%")
    
    # 汇总结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('annual_return', ascending=False)
    
    print(f"\n{'='*70}")
    print("参数优化结果汇总（按年化收益排序）")
    print(f"{'='*70}")
    print(f"{'持仓':>4} {'比例':>5} {'总收益':>8} {'年化':>8} {'夏普':>6} {'最大回撤':>8} {'卡玛':>6}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['holding_period']:>4} {row['trading_signal']:>5.0%} "
              f"{row['total_return']:>7.1%} {row['annual_return']:>7.1%} "
              f"{row['sharpe_ratio']:>6.2f} {row['max_drawdown']:>7.1%} "
              f"{row['calmar_ratio']:>6.2f}")
    
    # 最优参数
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print("最优参数组合:")
    print(f"  持仓周期: {best['holding_period']} 天")
    print(f"  交易比例: {best['trading_signal']:.0%}")
    print(f"  总收益: {best['total_return']:.2%}")
    print(f"  年化收益: {best['annual_return']:.2%}")
    print(f"  夏普比率: {best['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {best['max_drawdown']:.2%}")
    print(f"{'='*70}")
    
    # 保存结果
    results_df.to_csv('/root/cs_developer/carry_param_optimization.csv', index=False)
    print("\n[OK] 结果已保存到 carry_param_optimization.csv")


if __name__ == "__main__":
    main()
