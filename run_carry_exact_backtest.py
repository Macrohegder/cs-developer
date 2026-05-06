#!/usr/bin/env python3
"""
Carry 因子精确回测 — 使用实际到期日差距年化

因子定义:
    carry = (F1 - F2) / F1 / days_to_expiry * 365
    carry_20ma = carry.rolling(20).mean()

其中 days_to_expiry 是主力合约与次主力合约的实际到期日差距。

策略方向:
    - carry > 0 (Backwardation): 远期贴水 → 做多
    - carry < 0 (Contango): 远期升水 → 做空
    
回测参数:
    - 持仓周期: 5日
    - 交易比例: 20%
    - 做多高 carry, 做空低 carry
"""

import os
import sys
import argparse
from datetime import datetime
from math import floor
from typing import List, Dict, Optional

import pandas as pd
from pandas import Series, DataFrame, concat
import numpy as np
from clickhouse_driver import Client

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df, load_history_df
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import (
    calculate_portfolio_performance,
    calculate_product_pnl,
    calculate_product_target,
    calculate_overall_pnl,
    calculate_overall_statistics,
)

from strategies.cross_sectional_strategy import CrossSectionalStrategy


# =============================================================================
# 默认配置
# =============================================================================

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


def parse_args():
    parser = argparse.ArgumentParser(description="Carry 因子精确回测")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--capital", type=int, default=10_000_000)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--holding-period", type=int, default=5)
    parser.add_argument("--trading-signal", type=float, default=0.2)
    parser.add_argument("--cycle", type=int, default=20, help="carry 滚动平均天数")
    parser.add_argument("--symbols", type=str, default="", help="逗号分隔的品种列表")
    parser.add_argument("--plot", action="store_true", help="绘制图表")
    parser.add_argument("--save", action="store_true", default=True, help="保存结果")
    parser.add_argument("--skip-compute", action="store_true", help="跳过因子计算，直接加载已有CSV")
    return parser.parse_args()


def compute_carry_exact(
    symbols: List[str],
    start: datetime,
    end: datetime,
    cycle: int = 20
) -> pd.DataFrame:
    """
    计算 Carry 因子（使用实际到期日差距年化）
    
    Returns:
        DataFrame: index=datetime, columns=symbols, values=carry_20ma
    """
    print(f"\n[Carry Exact] 计算 carry 因子...")
    print(f"  品种数: {len(symbols)}")
    print(f"  区间: {start.date()} ~ {end.date()}")
    
    # 1. 加载88和88A2价格
    print("\n  [1/4] 加载价格数据...")
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
        except Exception as e:
            print(f"    {sym}: 跳过 ({e})")
    
    close_88_df = pd.DataFrame(close_88)
    close_a2_df = pd.DataFrame(close_a2)
    print(f"  有效品种: {len(valid_symbols)}")
    
    if len(valid_symbols) == 0:
        raise ValueError("没有有效的品种数据")
    
    # 2. 加载主力/次主力映射
    print("\n  [2/4] 加载主力映射...")
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
    print(f"  映射记录: {map_pivot.shape}")
    
    # 3. 获取合约到期日
    print("\n  [3/4] 计算到期日差距...")
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
    not_found = 0
    for c in all_contracts:
        c_base = c.split('.')[0]
        rows = contract_df.loc[contract_df.index == c_base, 'enddate']
        if len(rows) > 0:
            contract_enddates[c] = pd.to_datetime(rows.values[0])
        else:
            not_found += 1
    
    print(f"  合约: {len(all_contracts)}, 有到期日: {len(contract_enddates)}, 未找到: {not_found}")
    
    # 计算每日到期日差距
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
    
    avg_days = days_to_expiry.mean().mean()
    print(f"  平均到期日差距: {avg_days:.1f} 天")
    
    # 4. 计算 carry 因子
    print(f"\n  [4/4] 计算 carry (rolling {cycle}天)...")
    common_idx = close_88_df.index.intersection(close_a2_df.index).intersection(days_to_expiry.index)
    print(f"  共同交易日: {len(common_idx)}")
    
    f1 = close_88_df.loc[common_idx]
    f2 = close_a2_df.loc[common_idx]
    days = days_to_expiry.loc[common_idx]
    
    # carry = (f1-f2)/f1 / days * 365
    carry_raw = (f1 - f2) / f1 / days * 365.0
    # 处理异常值
    carry_raw = carry_raw.clip(lower=-2.0, upper=2.0)
    # rolling 平均
    carry_20 = carry_raw.rolling(window=cycle, min_periods=cycle//2).mean()
    
    print(f"  carry 均值: {carry_20.mean().mean():.4f}")
    print(f"  carry 标准差: {carry_20.std().mean():.4f}")
    print(f"  carry 覆盖率: {carry_20.notna().sum().sum() / carry_20.size * 100:.1f}%")
    
    return carry_20


def run_ic_analysis(factor_df: pd.DataFrame, close_df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> Dict:
    """运行 IC 分析"""
    print("\n[IC 分析]")
    
    results = {}
    for period in periods:
        # 计算 forward returns
        fwd_ret = close_df.pct_change(period).shift(-period)
        
        # 每日 IC
        ic_series = []
        for dt in factor_df.index:
            if dt not in fwd_ret.index:
                continue
            f = factor_df.loc[dt].dropna()
            r = fwd_ret.loc[dt].dropna()
            common = f.index.intersection(r.index)
            if len(common) < 5:
                continue
            ic = f[common].corr(r[common], method='spearman')
            if not np.isnan(ic):
                ic_series.append(ic)
        
        ic_series = pd.Series(ic_series)
        if len(ic_series) > 0:
            results[period] = {
                'mean_ic': ic_series.mean(),
                'ic_std': ic_series.std(),
                'ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                't_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))) if ic_series.std() > 0 else 0,
                'pos_ratio': (ic_series > 0).sum() / len(ic_series),
            }
            print(f"  {period}日 IC: mean={results[period]['mean_ic']:+.4f}, "
                  f"IR={results[period]['ir']:+.3f}, t={results[period]['t_stat']:+.2f}, "
                  f"正相关比例={results[period]['pos_ratio']:.1%}")
    
    return results


def run_backtest(
    factor_df: pd.DataFrame,
    symbols: List[str],
    start: datetime,
    end: datetime,
    capital: int = 10_000_000,
    commission: float = 0.0001,
    holding_period: int = 5,
    trading_signal: float = 0.2,
    plot_chart: bool = False,
) -> Dict:
    """运行完整回测"""
    print("\n[回测]")
    
    # 使用 StrategyBacktester + CrossSectionalStrategy
    backtester = StrategyBacktester(
        vt_symbols=symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=capital,
    )
    backtester.load_data()
    
    # 策略设置
    strategy_setting = {
        "holding_period": holding_period,
        "trading_signal": trading_signal,
        "long_low": False,  # carry > 0 做多，所以做多高值
        "aggregation": "sum",
        "factor_name": "carry_exact",
        "factor_data": factor_df,
    }
    
    # 自定义策略：直接加载预计算的因子
    class CarryExactStrategy(CrossSectionalStrategy):
        def calculate_daily_target_series(self, df=None):
            """
            重写因子获取逻辑：直接从 factor_data 读取，绕过 FactorManager
            """
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
                except Exception as e:
                    pass
            
            return Series(target_data)
    
    target_df = backtester.run_backtesting(CarryExactStrategy, strategy_setting)
    
    # 计算绩效
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=plot_chart,
    )
    
    return result


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
    
    # 1. 计算或加载 carry 因子
    if args.skip_compute and os.path.exists(f"/root/cs_developer/result_carry_exact_{args.cycle}_factor.csv"):
        print(f"[Skip] 直接加载已计算的 carry 因子...")
        carry_df = pd.read_csv(f"/root/cs_developer/result_carry_exact_{args.cycle}_factor.csv", index_col=0)
        carry_df.index = pd.to_datetime(carry_df.index)
        carry_df = carry_df.loc[start:end]
        print(f"  加载完成: {carry_df.shape}")
    else:
        carry_df = compute_carry_exact(symbols, start, end, cycle=args.cycle)
    
    # 2. 回测
    result = run_backtest(
        factor_df=carry_df,
        symbols=carry_df.columns.tolist(),
        start=start,
        end=end,
        capital=args.capital,
        commission=args.commission,
        holding_period=args.holding_period,
        trading_signal=args.trading_signal,
        plot_chart=args.plot,
    )
    
    # 5. 输出结果
    stats = result["statistics"]
    print("\n" + "="*60)
    print("Carry 因子精确回测结果")
    print("="*60)
    print(f"  回测区间: {stats['start_date'].date()} ~ {stats['end_date'].date()}")
    print(f"  总收益: {stats['total_return']}")
    print(f"  年化收益: {stats['annual_return']}")
    print(f"  夏普比率: {stats['sharpe_ratio']}")
    print(f"  最大回撤: {stats['max_ddpercent']}")
    print(f"  盈利天数: {stats['profit_days']}, 亏损天数: {stats['loss_days']}")
    print("="*60)
    
    # 6. 保存结果
    if args.save:
        prefix = f"/root/cs_developer/result_carry_exact_{args.cycle}"
        carry_df.to_csv(f"{prefix}_factor.csv")
        result["target"].to_csv(f"{prefix}_target.csv")
        result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
        result["product"].to_csv(f"{prefix}_product.csv")
        print(f"\n[OK] 结果已保存到 {prefix}_*.csv")


if __name__ == "__main__":
    main()
