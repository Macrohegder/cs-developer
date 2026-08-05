#!/usr/bin/env python3
"""
日历价差套利参数扫描

一次性加载数据，批量测试 EnhancedCalendarSpreadStrategy 参数组合，
找出 IC 盈利且 IM 夏普 >= 1.8 的最优参数。

用法：
    python scan_calendar_spread_params.py
"""

import sys
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# pandas 2.0+ 移除 iteritems
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# patch pytz
import pytz
_original_timezone = pytz.timezone

def _patched_timezone(zone):
    if zone == "Asia/Beijing":
        return _original_timezone("Asia/Shanghai")
    return _original_timezone(zone)

pytz.timezone = _patched_timezone

from clickhouse_driver import Client

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "factor_system"))

from vnpy.trader.constant import Interval
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df, load_history_df
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

from strategies.enhanced_calendar_spread_strategy import EnhancedCalendarSpreadStrategy


START = datetime(2022, 7, 22)
END = datetime(2026, 7, 14)
CAPITAL = 10_000_000
COMMISSION = 0.0001


def load_dominant_mapping(product: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = Client(host='localhost')
    keys = [f"{product}88.CFFEX@1", f"{product}88.CFFEX@2"]
    result = client.execute(
        'SELECT datetime, key, value FROM vnpy.vnpy_dominant_contract '
        'WHERE key IN %(keys)s AND datetime >= %(start)s AND datetime <= %(end)s '
        'ORDER BY datetime',
        {'keys': keys, 'start': start, 'end': end}
    )
    df = pd.DataFrame(result, columns=['datetime', 'key', 'value'])
    df['datetime'] = pd.to_datetime(df['datetime']).dt.normalize()
    df = df.drop_duplicates(subset=['datetime', 'key'])
    pivot = df.pivot(index='datetime', columns='key', values='value')
    return pivot


def prepare_product_data(product: str, contract_df: pd.DataFrame) -> Dict:
    """为单个品种准备数据"""
    mapping = load_dominant_mapping(product, START, END)
    idx_df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, END)
    if hasattr(idx_df.index, 'tz_localize'):
        idx_df.index = idx_df.index.tz_localize(None)

    k1 = f"{product}88.CFFEX@1"
    k2 = f"{product}88.CFFEX@2"
    contracts = set()
    if k1 in mapping.columns:
        contracts.update(mapping[k1].dropna().unique())
    if k2 in mapping.columns:
        contracts.update(mapping[k2].dropna().unique())
    contracts = sorted({c for c in contracts if isinstance(c, str) and '.' in c})

    # 预加载 history_df 和 backtester 公共结构
    history_df = load_history_df(contracts, Interval.DAILY, START, END)

    backtester = StrategyBacktester(
        vt_symbols=contracts,
        interval=Interval.DAILY,
        start=START,
        end=END,
        capital=CAPITAL,
    )
    backtester.history_df = history_df
    backtester.contract_df = contract_df

    symbol_ix = contract_df.index + "." + contract_df["exchange"]
    backtester.multiplier_series = pd.Series(contract_df["size"].values, index=symbol_ix)

    dominant_symbols = history_df.columns.get_level_values(0).drop_duplicates()
    intraday_change_list = []
    close_change_list = []
    for vt_symbol in dominant_symbols:
        open_series = history_df[(vt_symbol, "open_price")]
        close_series = history_df[(vt_symbol, "close_price")]
        intraday_change_list.append(close_series - open_series)
        close_change_list.append(close_series - close_series.shift(1))

    backtester.intraday_change_df = pd.concat(intraday_change_list, axis=1)
    backtester.intraday_change_df.columns = dominant_symbols
    backtester.close_change_df = pd.concat(close_change_list, axis=1)
    backtester.close_change_df.columns = dominant_symbols

    return {
        'product': product,
        'mapping': mapping,
        'idx_close': idx_df['close_price'],
        'contracts': contracts,
        'history_df': history_df,
        'backtester': backtester,
    }


def run_single(product: str, params: Dict, data_cache: Dict, contract_df: pd.DataFrame) -> Dict:
    """运行单个参数组合的回测"""
    mapping = data_cache[product]['mapping'].copy()
    idx_close = data_cache[product]['idx_close']
    history_df = data_cache[product]['history_df']
    backtester = data_cache[product]['backtester']

    if len(data_cache[product]['contracts']) == 0:
        return None

    mapping = mapping.reindex(history_df.index)

    strategy_setting = {
        "products": [product],
        "vol_threshold": params['vol_threshold'],
        "vol_window": params['vol_window'],
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "stop_loss_pct": params['stop_loss_pct'],
        "take_profit_pct": params['take_profit_pct'],
        "only_quarterly": params['only_quarterly'],
        "roll_before_days": params['roll_before_days'],
        "mapping": mapping,
        "idx_close": idx_close,
        "contract_df": contract_df,
        "history_df": history_df,
    }

    target_df = backtester.run_backtesting(EnhancedCalendarSpreadStrategy, strategy_setting)

    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=COMMISSION,
        capital=CAPITAL,
        plot_chart=False,
    )

    overall = result["overall"]
    net_pnl = overall["net_pnl"]
    total_pnl = net_pnl.sum()
    total_return = overall["balance"].iloc[-1] / overall["balance"].iloc[0] - 1
    total_days = len(overall)
    annual_return = total_return / total_days * 250
    daily_ret = overall["return"]
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(250) if daily_ret.std() > 0 else 0
    max_dd = overall["ddpercent"].min()
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0
    profit_days = int((net_pnl > 0).sum())
    loss_days = int((net_pnl < 0).sum())

    return {
        'product': product,
        'total_pnl': round(total_pnl, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'profit_days': profit_days,
        'loss_days': loss_days,
        **params,
    }


def main():
    print("="*70)
    print("日历价差套利参数扫描")
    print("="*70)

    # 加载公共数据
    print("\n[1/2] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/2] 加载品种数据...")
    data_cache = {
        'IC': prepare_product_data('IC', contract_df),
        'IM': prepare_product_data('IM', contract_df),
    }

    # 参数空间
    param_grid = {
        'vol_threshold': [0.10, 0.15, 0.20],
        'vol_window': [10, 20],
        'stop_loss_pct': [0.0, 0.01, 0.02],
        'take_profit_pct': [0.0, 0.02, 0.05],
        'only_quarterly': [False, True],
        'roll_before_days': [0, 5, 10],
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"\n总参数组合数: {len(combinations)}")

    results_ic = []
    results_im = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] 参数: {params}")

        res_ic = run_single('IC', params, data_cache, contract_df)
        if res_ic:
            results_ic.append(res_ic)
            print(f"  IC -> sharpe={res_ic['sharpe']:.3f}, pnl={res_ic['total_pnl']:,.0f}, max_dd={res_ic['max_dd']:.2%}")

        res_im = run_single('IM', params, data_cache, contract_df)
        if res_im:
            results_im.append(res_im)
            print(f"  IM -> sharpe={res_im['sharpe']:.3f}, pnl={res_im['total_pnl']:,.0f}, max_dd={res_im['max_dd']:.2%}")

    df_ic = pd.DataFrame(results_ic)
    df_im = pd.DataFrame(results_im)

    output_dir = Path('/root/quant/cs_developer/docs/ic_im_vol_scan')
    output_dir.mkdir(parents=True, exist_ok=True)
    df_ic.to_csv(f'{output_dir}/ic_scan_results.csv', index=False)
    df_im.to_csv(f'{output_dir}/im_scan_results.csv', index=False)

    # 筛选：IC 盈利且 IM 夏普 >= 1.8
    ic_profitable = df_ic[df_ic['total_pnl'] > 0].copy()
    im_good = df_im[df_im['sharpe'] >= 1.8].copy()

    print("\n" + "="*70)
    print("扫描结果汇总")
    print("="*70)
    print(f"IC 盈利组合数: {len(ic_profitable)} / {len(df_ic)}")
    print(f"IM 夏普>=1.8 组合数: {len(im_good)} / {len(df_im)}")

    if len(ic_profitable) > 0:
        print("\nIC 盈利组合 TOP 10（按夏普）:")
        print(ic_profitable.sort_values('sharpe', ascending=False).head(10).to_string(index=False))

    if len(im_good) > 0:
        print("\nIM 夏普>=1.8 组合 TOP 10（按夏普）:")
        print(im_good.sort_values('sharpe', ascending=False).head(10).to_string(index=False))

    # 找共同参数
    common_cols = keys + ['product']
    if len(ic_profitable) > 0 and len(im_good) > 0:
        merged = pd.merge(
            ic_profitable[keys + ['sharpe', 'total_pnl', 'max_dd']],
            im_good[keys + ['sharpe', 'total_pnl', 'max_dd']],
            on=keys,
            suffixes=('_ic', '_im')
        )
        print(f"\nIC 盈利 + IM 夏普>=1.8 的共同参数组合数: {len(merged)}")
        if len(merged) > 0:
            merged['combined_score'] = merged['sharpe_ic'] + merged['sharpe_im']
            print("\n共同参数 TOP 10（按 combined_score）:")
            print(merged.sort_values('combined_score', ascending=False).head(10).to_string(index=False))
            merged.to_csv(f'{output_dir}/common_params.csv', index=False)

    print(f"\n[OK] 结果保存到 {output_dir}")


if __name__ == "__main__":
    main()
