#!/usr/bin/env python3
"""
Carry 因子 rolling 对比实验

对比三种 carry 计算方式:
- 无 rolling (日频原始 carry)
- rolling 3 天
- rolling 5 天
- rolling 20 天 (作为对照)

回测参数: hp=1, ts=40%, 37品种, 2011-01 ~ 2026-03
"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import os
import pandas as pd
from datetime import datetime
from math import floor
from typing import List
from clickhouse_driver import Client

from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df
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


def compute_carry_variants(
    symbols: List[str],
    start: datetime,
    end: datetime,
) -> dict:
    """计算多种 rolling 版本的 carry 因子"""
    print(f"\n[Carry Variants] 计算 carry 因子...")
    print(f"  品种数: {len(symbols)}")
    print(f"  区间: {start.date()} ~ {end.date()}")
    
    # 1. 加载88和88A2价格
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
        except Exception:
            pass
    
    close_88_df = pd.DataFrame(close_88)
    close_a2_df = pd.DataFrame(close_a2)
    print(f"  有效品种: {len(valid_symbols)}")
    
    # 2. 加载主力映射
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
    
    # 3. 获取合约到期日
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
    
    # 4. 计算到期日差距
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
    
    # 5. 计算 carry 原始值
    common_idx = close_88_df.index.intersection(close_a2_df.index).intersection(days_to_expiry.index)
    f1 = close_88_df.loc[common_idx]
    f2 = close_a2_df.loc[common_idx]
    days = days_to_expiry.loc[common_idx]
    
    carry_raw = (f1 - f2) / f1 / days * 365.0
    carry_raw = carry_raw.clip(lower=-2.0, upper=2.0)
    
    # 6. 生成不同 rolling 版本
    variants = {}
    
    # 无 rolling
    variants['raw'] = carry_raw.copy()
    
    # rolling 3
    variants['roll3'] = carry_raw.rolling(window=3, min_periods=2).mean()
    
    # rolling 5
    variants['roll5'] = carry_raw.rolling(window=5, min_periods=3).mean()
    
    # rolling 20 (对照)
    variants['roll20'] = carry_raw.rolling(window=20, min_periods=10).mean()
    
    for name, df in variants.items():
        print(f"  carry_{name}: 均值={df.mean().mean():.4f}, 标准差={df.std().mean():.4f}, 覆盖率={df.notna().sum().sum() / df.size * 100:.1f}%")
    
    return variants


class CarryVariantStrategy(CrossSectionalStrategy):
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


def run_variant_backtest(backtester, carry_df: pd.DataFrame, variant_name: str, hp: int, ts: float, capital: int = 10_000_000, commission: float = 0.0001):
    """运行单个 carry 版本的回测（复用已加载数据的 backtester）"""
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
    print("Carry 因子 rolling 对比实验")
    print("对比: raw(无rolling) vs roll3 vs roll5 vs roll20")
    print("="*60)
    
    start = datetime(2011, 1, 1)
    end = datetime(2026, 3, 31)
    
    # 1. 计算或加载 carry 变体
    cache_file = "/root/cs_developer/carry_variants_cache.pkl"
    if os.path.exists(cache_file):
        print(f"\n[加载缓存] {cache_file}")
        import pickle
        with open(cache_file, 'rb') as f:
            variants = pickle.load(f)
        for name, df in variants.items():
            print(f"  carry_{name}: {df.shape}")
    else:
        variants = compute_carry_variants(DEFAULT_SYMBOLS, start, end)
        # 保存缓存
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(variants, f)
        print(f"\n[缓存已保存] {cache_file}")
    
    # 2. 预创建 backtester 并加载数据（只执行一次）
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
    
    # 3. 回测参数
    configs = [
        ("hp=1, ts=40%", 1, 0.4),
        ("hp=5, ts=20%", 5, 0.2),
    ]
    
    results = []
    
    for config_name, hp, ts in configs:
        print(f"\n{'='*60}")
        print(f"回测参数: {config_name}")
        print(f"{'='*60}")
        
        for name, carry_df in variants.items():
            print(f"\n  carry_{name} ...", end=" ", flush=True)
            r = run_variant_backtest(backtester, carry_df, name, hp, ts)
            r['config'] = config_name
            results.append(r)
            print(f"OK → total={r['total_return']}, sharpe={r['sharpe']}, max_dd={r['max_drawdown']}")
    
    # 3. 汇总结果
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Carry Rolling 对比结果汇总")
    print("="*60)
    
    for config_name in results_df['config'].unique():
        print(f"\n【{config_name}】")
        sub = results_df[results_df['config'] == config_name].sort_values(by='sharpe', ascending=False)
        print(sub[['variant', 'total_return', 'annual_return', 'sharpe', 'max_drawdown', 'calmar']].to_string(index=False))
    
    print("\n" + "="*60)
    
    # 保存
    results_df.to_csv("/root/cs_developer/carry_rolling_compare.csv", index=False)
    print("结果已保存到 carry_rolling_compare.csv")


if __name__ == "__main__":
    main()
