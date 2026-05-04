#!/usr/bin/env python3
"""
指数敏感性对比测试 — 因子值基础 × 回测价格基础 全组合扫描

规则:
- Carry 类因子: 因子值固定 88+88A2, 回测价格对比 99/888/889
- 其他因子: 因子值对比 88/99/888/889, 回测价格对比 99/888/889
- 简化版回测绝对不能用 88 价格
- 完整版基准单独运行（本脚本只测简化版组合）

用法:
    python run_index_sensitivity.py --factor skew
    python run_index_sensitivity.py --batch --factors skew,momentum,kurtosis,carry_ret
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import traceback
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from vnpy.trader.constant import Interval

from factor_system.factor_engine import FactorEngine
from factor_system.backtest_engine import BatchBacktestEngine
from factor_system.factor_registry import get_registry


DEFAULT_SYMBOLS = [
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

BAD_SYMBOLS_888 = {'I88.DCE', 'LU88.INE', 'P88.DCE'}
BAD_SYMBOLS_889 = {'CJ88.CZCE', 'FU88.SHFE', 'RU88.SHFE'}
CARRY_FACTORS = {'carry_ret', 'carry_momentum', 'spread_zscore', 'spread_return'}


def parse_args():
    parser = argparse.ArgumentParser(description="指数敏感性对比测试")
    parser.add_argument("--factor", type=str, default="skew")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--factors", type=str, default=None)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--output", type=str, default="/root/cs_developer/factor_system/reports/index_sensitivity.csv")
    return parser.parse_args()


def get_valid_symbols(primary_suffix: str) -> List[str]:
    bad = set()
    if primary_suffix == '888':
        bad = BAD_SYMBOLS_888
    elif primary_suffix == '889':
        bad = BAD_SYMBOLS_889
    return [s for s in DEFAULT_SYMBOLS if s not in bad]


def compute_factor(factor_name: str, symbols: List[str], start: datetime, end: datetime,
                   primary_suffix: str, secondary_suffix: str) -> Optional[pd.DataFrame]:
    try:
        engine = FactorEngine(
            dominant_symbols=symbols, start=start, end=end,
            primary_suffix=primary_suffix, secondary_suffix=secondary_suffix, verbose=False
        )
        engine.load_all_data()
        df = engine.compute(factor_name)
        return df
    except Exception as e:
        print(f"  [ERROR] 计算失败: {e}")
        return None


def backtest(factor_df: pd.DataFrame, symbols: List[str], start: str, end: str,
             price_suffix: str, long_low: bool) -> Optional[Dict]:
    try:
        bt = BatchBacktestEngine(
            symbols, start, end,
            primary_suffix=price_suffix,
            secondary_suffix="889" if price_suffix != "88" else "88A2",
            verbose=False
        )
        result = bt.run_factor(
            "test", factor_df=factor_df, long_low=long_low, auto_direction=False,
            holding_period=5, trading_signal=0.2, leverage=2.0, commission=0.0001
        )
        if result:
            return {
                "annual_return": result.annual_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "calmar_ratio": result.calmar_ratio,
                "win_rate": result.win_rate,
            }
    except Exception as e:
        print(f"  失败: {e}")
    return None


def run_comparison(factor_name: str, start: datetime, end: datetime) -> List[Dict]:
    registry = get_registry()
    meta = registry.get(factor_name)
    long_low = (meta.ic_direction == -1) if meta else False
    is_carry = factor_name in CARRY_FACTORS

    if is_carry:
        factor_bases = [("88", "88A2")]
    else:
        factor_bases = [("88", "88A2"), ("99", "889"), ("888", "889"), ("889", "88A2")]

    price_bases = ["99", "888", "889"]
    results = []

    print(f"\n{'='*70}")
    print(f"因子: {factor_name} | long_low={long_low} | carry={is_carry}")
    print(f"{'='*70}")

    for fac_pri, fac_sec in factor_bases:
        fac_symbols = get_valid_symbols(fac_pri)
        print(f"\n[因子值: {fac_pri}+{fac_sec}] 计算中...", end=" ")
        factor_df = compute_factor(factor_name, fac_symbols, start, end, fac_pri, fac_sec)
        if factor_df is None or factor_df.empty:
            print("失败")
            continue
        print(f"OK shape={factor_df.shape}")

        for price_pri in price_bases:
            price_symbols = get_valid_symbols(price_pri)
            common = list(set(factor_df.columns) & set(price_symbols))
            if len(common) < 10:
                continue

            print(f"  [价格: {price_pri}] ", end="")
            r = backtest(factor_df[common], common,
                         start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                         price_pri, long_low)
            if r:
                print(f"年化{r['annual_return']:+.2%} 夏普{r['sharpe_ratio']:+.2f}")
                results.append({
                    "factor": factor_name, "factor_base": f"{fac_pri}+{fac_sec}",
                    "price_base": price_pri, **r
                })
            else:
                print("失败")

    return results


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    if args.batch and args.factors:
        factor_list = [f.strip() for f in args.factors.split(",")]
    else:
        factor_list = [args.factor]

    all_results = []
    for fname in factor_list:
        all_results.extend(run_comparison(fname, start, end))

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        print(f"\n[OK] 结果已保存: {args.output}")

        print("\n" + "="*70)
        print("最优组合（按夏普排序）")
        print("="*70)
        for factor in df["factor"].unique():
            sub = df[df["factor"] == factor]
            best = sub.loc[sub["sharpe_ratio"].idxmax()]
            print(f"  {factor:<20s} 因子值={best['factor_base']:<12s} 价格={best['price_base']:<4s} "
                  f"夏普={best['sharpe_ratio']:+.2f} 年化={best['annual_return']:+.2%}")


if __name__ == "__main__":
    main()
