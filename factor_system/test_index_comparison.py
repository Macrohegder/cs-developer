#!/usr/bin/env python3
"""
测试不同连续合约指数（99/888/889 vs 88/88A2）对 carry_ret 简化回测的影响

目的：找出与完整版回测（StrategyBacktester + 具体合约）结果最接近的指数组合

测试方案：
1. 主力=99, 次主力=889
2. 主力=888, 次主力=889
3. 主力=88, 次主力=88A2（原方案，对照）

完整版基准：年化 +6.00%, 夏普 0.59, 最大回撤 -15.33%
"""

import sys
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from vnpy.trader.constant import Interval
from vnpy.trader.database import get_database
from factor_system.factors.batch_factors import safe_div


# =============================================================================
# 配置
# =============================================================================

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
    'ZN88.SHFE'
]

START = "2020-01-01"
END = "2024-12-31"

# 完整版基准
BENCHMARK = {
    "annual_return": 0.0600,
    "sharpe_ratio": 0.59,
    "max_drawdown": -0.1533,
}


# =============================================================================
# 数据加载
# =============================================================================

def load_index_data(dominant_symbols: list, start: str, end: str, 
                     primary_suffix: str, secondary_suffix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载指定后缀的连续合约数据
    
    参数:
        primary_suffix: 主力连续后缀，如 "99", "888", "88"
        secondary_suffix: 次主力连续后缀，如 "889", "88A2"
    """
    db = get_database()
    
    primary_data = {}
    secondary_data = {}
    
    for ds in dominant_symbols:
        # 提取品种代码和交易所
        # ds 格式: "RB88.SHFE"
        parts = ds.split(".")
        exchange_str = parts[1]
        symbol_base = parts[0].replace("88", "")  # "RB88" -> "RB"
        
        primary_symbol = f"{symbol_base}{primary_suffix}"
        secondary_symbol = f"{symbol_base}{secondary_suffix}"
        
        # 动态获取 Exchange 枚举
        from vnpy.trader.constant import Exchange
        exchange = getattr(Exchange, exchange_str)
        
        # 加载主力数据
        try:
            bars = db.load_bar_data(primary_symbol, exchange, Interval.DAILY, 
                                    pd.Timestamp(start), pd.Timestamp(end))
            if bars:
                df = pd.DataFrame.from_records([{
                    'datetime': b.datetime,
                    'close': b.close_price,
                    'open': b.open_price,
                    'high': b.high_price,
                    'low': b.low_price,
                    'volume': b.volume,
                } for b in bars])
                df.set_index('datetime', inplace=True)
                primary_data[ds] = df['close']
        except Exception:
            pass
        
        # 加载次主力数据
        try:
            bars = db.load_bar_data(secondary_symbol, exchange, Interval.DAILY,
                                    pd.Timestamp(start), pd.Timestamp(end))
            if bars:
                df = pd.DataFrame.from_records([{
                    'datetime': b.datetime,
                    'close': b.close_price,
                } for b in bars])
                df.set_index('datetime', inplace=True)
                secondary_data[ds] = df['close']
        except Exception:
            pass
    
    primary_df = pd.DataFrame(primary_data)
    secondary_df = pd.DataFrame(secondary_data)
    
    return primary_df, secondary_df


# =============================================================================
# carry_ret 计算
# =============================================================================

def calc_carry_ret(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    """计算 carry_ret: (F1 - F2) / F1 / 60 * 365"""
    common_cols = primary_df.columns.intersection(secondary_df.columns)
    f1 = primary_df[common_cols]
    f2 = secondary_df[common_cols]
    
    carry = safe_div(f1 - f2, f1, fill=0.0) / 60.0 * 365.0
    
    # 扩展回原始列
    result = pd.DataFrame(index=primary_df.index, columns=primary_df.columns)
    result[common_cols] = carry
    return result


# =============================================================================
# 简化回测
# =============================================================================

def run_simplified_backtest(factor_df: pd.DataFrame, close_df: pd.DataFrame,
                            holding_period: int = 5, trading_signal: float = 0.2,
                            leverage: float = 2.0, commission: float = 0.0001) -> Dict:
    """向量化简化回测"""
    
    # 对齐
    factor_df, close_df = factor_df.align(close_df, join='inner', axis=0)
    returns_df = close_df.pct_change()
    
    # 生成信号（做多高 carry = Backwardation）
    signal_df = pd.DataFrame(0.0, index=factor_df.index, columns=factor_df.columns)
    
    for dt in factor_df.index:
        row = factor_df.loc[dt].dropna()
        if len(row) < 4:
            continue
        row_sorted = row.sort_values()
        choose = max(1, int(np.floor(trading_signal * len(row_sorted))))
        
        # long_low=False: 做多高 carry（Backwardation），做空低 carry（Contango）
        long_symbols = row_sorted.index[-choose:]
        short_symbols = row_sorted.index[:choose]
        
        signal_df.loc[dt, long_symbols] = 1.0
        signal_df.loc[dt, short_symbols] = -1.0
    
    # 滚动持仓
    position_df = signal_df.rolling(window=holding_period, min_periods=1).sum()
    
    # 组合收益
    position_shifted = position_df.shift(1)
    weight_abs_sum = position_shifted.abs().sum(axis=1).replace(0, np.nan)
    weight_df = position_shifted.div(weight_abs_sum, axis=0)
    
    gross_return = (weight_df * returns_df).sum(axis=1).fillna(0)
    
    # 手续费
    pos_change = position_df.diff().abs().sum(axis=1)
    total_pos = position_df.abs().sum(axis=1).replace(0, np.nan)
    turnover_ratio = (pos_change / total_pos).fillna(0)
    cost = turnover_ratio * commission * 2 * leverage
    
    net_return = gross_return * leverage - cost
    net_return = net_return.fillna(0)
    nav = (1 + net_return).cumprod()
    
    # 绩效
    valid = net_return.notna()
    net_return = net_return[valid]
    nav = nav[valid]
    
    annual_return = net_return.mean() * 240
    annual_vol = net_return.std() * np.sqrt(240)
    sharpe = annual_return / annual_vol if annual_vol > 1e-12 else 0.0
    
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()
    
    calmar = -annual_return / max_dd if max_dd < -1e-12 else np.inf
    
    return {
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "nav_series": nav,
        "net_return_series": net_return,
    }


def calc_distance(metrics: Dict, benchmark: Dict) -> float:
    """计算与基准的欧氏距离（标准化后）"""
    # 使用 annual_return, sharpe_ratio, max_drawdown 三个维度
    # 标准化：用基准值作为参考点
    diff_return = (metrics["annual_return"] - benchmark["annual_return"]) / abs(benchmark["annual_return"])
    diff_sharpe = (metrics["sharpe_ratio"] - benchmark["sharpe_ratio"]) / abs(benchmark["sharpe_ratio"])
    diff_dd = (metrics["max_drawdown"] - benchmark["max_drawdown"]) / abs(benchmark["max_drawdown"])
    
    return np.sqrt(diff_return**2 + diff_sharpe**2 + diff_dd**2)


# =============================================================================
# 主程序
# =============================================================================

def main():
    print("=" * 70)
    print("连续合约指数对比测试 — 寻找最接近完整版回测的指数组合")
    print("=" * 70)
    print(f"\n基准（完整版 StrategyBacktester, 做多 Backwardation）:")
    print(f"  年化收益: {BENCHMARK['annual_return']:+.2%}")
    print(f"  夏普比率: {BENCHMARK['sharpe_ratio']:+.2f}")
    print(f"  最大回撤: {BENCHMARK['max_drawdown']:.2%}")
    print()
    
    # 定义测试方案
    schemes = [
        ("主力=99, 次主力=889", "99", "889"),
        ("主力=888, 次主力=889", "888", "889"),
        ("主力=88, 次主力=88A2 (原方案)", "88", "88A2"),
    ]
    
    results = []
    
    for name, primary_suffix, secondary_suffix in schemes:
        print(f"\n[{name}]")
        print("-" * 50)
        
        # 加载数据
        primary_df, secondary_df = load_index_data(
            DOMINANT_SYMBOLS, START, END, primary_suffix, secondary_suffix
        )
        
        print(f"  主力数据: {primary_df.shape}")
        print(f"  次主力数据: {secondary_df.shape}")
        
        if primary_df.empty or secondary_df.empty:
            print("  [SKIP] 数据加载失败")
            continue
        
        # 计算 carry_ret
        carry_df = calc_carry_ret(primary_df, secondary_df)
        
        # 回测
        backtest_result = run_simplified_backtest(carry_df, primary_df)
        
        # 计算与基准的距离
        distance = calc_distance(backtest_result, BENCHMARK)
        
        print(f"  年化收益: {backtest_result['annual_return']:+.2%}")
        print(f"  夏普比率: {backtest_result['sharpe_ratio']:+.2f}")
        print(f"  最大回撤: {backtest_result['max_drawdown']:.2%}")
        print(f"  Calmar:   {backtest_result['calmar_ratio']:+.2f}")
        print(f"  与基准距离: {distance:.3f}")
        
        results.append({
            "name": name,
            "primary_suffix": primary_suffix,
            "secondary_suffix": secondary_suffix,
            "distance": distance,
            **backtest_result,
        })
    
    # 排序
    results.sort(key=lambda x: x["distance"])
    
    print("\n" + "=" * 70)
    print("对比总结（按与基准的距离排序）")
    print("=" * 70)
    print(f"{'排名':<4} {'方案':<30} {'年化收益':<10} {'夏普':<8} {'最大回撤':<10} {'距离':<8}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['name']:<30} {r['annual_return']:>+8.2%}   {r['sharpe_ratio']:>+5.2f}   {r['max_drawdown']:>7.2%}   {r['distance']:>6.3f}")
    
    print("\n" + "=" * 70)
    if results:
        best = results[0]
        print(f"【推荐】最接近完整版的方案: {best['name']}")
        print(f"  距离: {best['distance']:.3f}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
