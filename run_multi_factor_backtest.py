#!/usr/bin/env python3
"""
多因子快速回测 — 使用预计算的 parquet 因子数据

支持的因子（从 factor_system/data/ 加载）:
- momentum (cycle=20): ic_direction=1
- reversal (cycle=20): ic_direction=-1
- skew (cycle=60): ic_direction=-1
- term_structure_slope (cycle=20): ic_direction=1
- basic_volatility (cycle=20): ic_direction=-1
- carry_ret (cycle=5): ic_direction=1
"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import os
import pandas as pd
from datetime import datetime
from math import floor

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

from strategies.cross_sectional_strategy import CrossSectionalStrategy


# 预计算 parquet 的时间范围
PARQUET_START = "2020-01-01"
PARQUET_END = "2024-12-31"

# 因子配置: (name, parameter, ic_direction, long_low)
FACTOR_CONFIGS = [
    ("momentum", "cycle=20", 1, False),
    ("reversal", "cycle=20", -1, True),
    ("skew", "cycle=60", -1, True),
    ("term_structure_slope", "cycle=20", 1, False),
    ("basic_volatility", "cycle=20", -1, True),
    ("carry_ret", "cycle=5", 1, False),
]

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


class GenericFactorStrategy(CrossSectionalStrategy):
    """通用因子策略 — 直接从 factor_data 读取"""
    
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


def load_factor_from_parquet(factor_name: str, parameter: str) -> pd.DataFrame:
    """从 parquet 加载预计算因子"""
    # 构建文件名
    param_str = parameter.replace("=", "")
    fname = f"{factor_name}_{PARQUET_START}_{PARQUET_END}.parquet"
    path = f"/root/cs_developer/factor_system/data/{fname}"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"因子文件不存在: {path}")
    
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    
    # 过滤掉 -1.0 的填充值（缺失值标记）
    df = df.replace(-1.0, pd.NA)
    
    return df


def run_factor_backtest(factor_df: pd.DataFrame, factor_name: str, hp: int, ts: float, long_low: bool, capital: int = 10_000_000, commission: float = 0.0001):
    """运行单个因子的回测"""
    start = datetime.strptime(PARQUET_START, "%Y-%m-%d")
    end = datetime.strptime(PARQUET_END, "%Y-%m-%d")
    symbols = factor_df.columns.tolist()
    
    # 过滤在默认列表中的品种
    symbols = [s for s in symbols if s in DEFAULT_SYMBOLS]
    factor_df = factor_df[symbols]
    
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
        "factor_data": factor_df,
    }
    
    target_df = backtester.run_backtesting(GenericFactorStrategy, setting)
    
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=commission,
        capital=capital,
        plot_chart=False,
    )
    
    stats = result["statistics"]
    return {
        "factor": factor_name,
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
    print("多因子快速回测 — 使用预计算 parquet 数据")
    print("="*60)
    
    results = []
    
    # 默认回测参数
    hp = 5
    ts = 0.2
    
    total = len(FACTOR_CONFIGS)
    for i, (name, param, ic_dir, long_low) in enumerate(FACTOR_CONFIGS, 1):
        print(f"\n[{i}/{total}] 回测因子: {name} ({param})...")
        try:
            factor_df = load_factor_from_parquet(name, param)
            print(f"  加载完成: {factor_df.shape}, 品种: {len(factor_df.columns)}")
            
            r = run_factor_backtest(factor_df, name, hp, ts, long_low)
            results.append(r)
            print(f"  结果: total={r['total_return']}, annual={r['annual_return']}, sharpe={r['sharpe']}, max_dd={r['max_drawdown']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "factor": name, "hp": hp, "ts": ts,
                "total_return": "ERROR", "annual_return": "",
                "sharpe": "", "max_drawdown": "", "calmar": "",
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="sharpe", ascending=False, na_position="last")
    results_df.to_csv("/root/cs_developer/multi_factor_results.csv", index=False)
    
    print("\n" + "="*60)
    print("多因子回测结果 (按 Sharpe 排序)")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    print("\n结果已保存到 multi_factor_results.csv")


if __name__ == "__main__":
    main()
