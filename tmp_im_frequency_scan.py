#!/usr/bin/env python3
"""临时脚本：扫描 IM 单品种不同过滤条件下的交易频率与绩效"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

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


def run_im_config(label: str, only_quarterly: bool, vol_threshold: float):
    product = "IM"
    latest = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, datetime.today()).index[-1]
    latest = latest.to_pydatetime() if hasattr(latest, 'to_pydatetime') else latest
    if hasattr(latest, 'tz_localize'):
        latest = latest.tz_localize(None)

    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    mapping = load_dominant_mapping(product, START, latest)
    idx_df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, latest)
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

    history_df = load_history_df(contracts, Interval.DAILY, START, latest)

    backtester = StrategyBacktester(
        vt_symbols=contracts,
        interval=Interval.DAILY,
        start=START,
        end=latest,
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

    strategy_setting = {
        "products": [product],
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mapping.copy().reindex(history_df.index),
        "idx_close": idx_df['close_price'],
        "contract_df": contract_df,
        "history_df": history_df,
        "vol_threshold": {product: vol_threshold},
        "vol_window": 20,
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "only_quarterly": only_quarterly,
        "roll_before_days": 5,
        "use_quarterly_pair": False,
        "dynamic_direction": False,
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
    total_pnl = overall["net_pnl"].sum()
    total_return = overall["balance"].iloc[-1] / overall["balance"].iloc[0] - 1
    annual_return = total_return / len(overall) * 250
    daily_ret = overall["return"]
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(250) if daily_ret.std() > 0 else 0
    max_dd = overall["ddpercent"].min()

    # 交易频率统计
    pos_days = (target_df.abs().sum(axis=1) > 0).sum()
    entries = ((target_df.abs().sum(axis=1).shift(1) == 0) & (target_df.abs().sum(axis=1) > 0)).sum()
    avg_hold = target_df.apply(lambda col: (col != 0).sum()).sum() / (target_df.abs().sum(axis=1) > 0).sum() if pos_days > 0 else 0

    print(f"{label:45s} | sharpe={sharpe:.4f} | return={total_return:.4%} | max_dd={max_dd:.4%} | "
          f"持仓天数={pos_days:3d}({pos_days/len(target_df):.1%}) | 开仓次数={entries:2d} | 日均合约持仓={avg_hold:.2f}")
    return {
        "label": label, "sharpe": sharpe, "total_return": total_return, "max_dd": max_dd,
        "pos_days": pos_days, "entries": entries, "avg_hold": avg_hold,
    }


def main():
    configs = [
        ("IM only_quarterly=True, vol=0.15", True, 0.15),
        ("IM only_quarterly=True, vol=0.10", True, 0.10),
        ("IM only_quarterly=True, vol=0.05", True, 0.05),
        ("IM only_quarterly=False, vol=0.15", False, 0.15),
        ("IM only_quarterly=False, vol=0.10", False, 0.10),
        ("IM only_quarterly=False, vol=0.05", False, 0.05),
    ]
    results = []
    for label, oq, vt in configs:
        results.append(run_im_config(label, oq, vt))


if __name__ == "__main__":
    main()
