#!/usr/bin/env python3
"""临时脚本：对比不同合约对/方向模式下的 IC/IM 日历价差绩效"""

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


def get_all_product_contracts(product: str, contract_df: pd.DataFrame, start: datetime) -> List[str]:
    start_ts = pd.Timestamp(start)
    mask = (
        (contract_df['exchange'] == 'CFFEX')
        & (contract_df.index.astype(str).str.startswith(product))
        & (contract_df['product'] == '期货')
    )
    syms = contract_df[mask].index.astype(str).tolist()
    syms = [
        s for s in syms
        if pd.to_datetime(contract_df.loc[s, 'expiry'], errors='coerce') > start_ts
    ]
    return sorted([f"{s}.CFFEX" for s in syms])


def get_latest_common_date() -> datetime:
    latest_dates = []
    for product in ["IC", "IM"]:
        df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, datetime.today())
        if hasattr(df.index, 'tz_localize'):
            df.index = df.index.tz_localize(None)
        latest_dates.append(df.index[-1])
    latest = min(latest_dates)
    return latest.to_pydatetime() if hasattr(latest, 'to_pydatetime') else latest


def prepare_contracts(products: List[str], contract_df: pd.DataFrame, end: datetime, use_quarterly_pair: bool):
    all_contracts = []
    mappings = {}
    idx_closes = {}
    for p in products:
        mapping = load_dominant_mapping(p, START, end)
        mappings[p] = mapping
        idx_df = load_bar_df(f"{p}88.CFFEX", Interval.DAILY, START, end)
        if hasattr(idx_df.index, 'tz_localize'):
            idx_df.index = idx_df.index.tz_localize(None)
        idx_closes[p] = idx_df['close_price']

        if use_quarterly_pair:
            contracts = get_all_product_contracts(p, contract_df, START)
        else:
            k1 = f"{p}88.CFFEX@1"
            k2 = f"{p}88.CFFEX@2"
            extra = set()
            if k1 in mapping.columns:
                extra.update(mapping[k1].dropna().unique())
            if k2 in mapping.columns:
                extra.update(mapping[k2].dropna().unique())
            contracts = sorted({c for c in extra if isinstance(c, str) and '.' in c})
        all_contracts.extend(contracts)
    return sorted(set(all_contracts)), mappings, idx_closes


def run_config(products: List[str], label: str, base_params: Dict, dynamic_params: Dict):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    latest = get_latest_common_date()
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    use_quarterly = base_params.get("use_quarterly_pair", False)
    contracts, mappings, idx_closes = prepare_contracts(products, contract_df, latest, use_quarterly)

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
        "products": products,
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": {p: mappings[p].copy().reindex(history_df.index) for p in products},
        "idx_close": idx_closes,
        "contract_df": contract_df,
        "history_df": history_df,
        "vol_threshold": {p: 0.15 for p in products},
        **base_params,
        **dynamic_params,
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
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0
    print(f"  products      : {products}")
    print(f"  total_pnl     : {total_pnl:,.2f}")
    print(f"  total_return  : {total_return:.4%}")
    print(f"  annual_return : {annual_return:.4%}")
    print(f"  max_dd        : {max_dd:.4%}")
    print(f"  sharpe        : {sharpe:.4f}")
    print(f"  calmar        : {calmar:.4f}")
    return {
        "label": label,
        "products": products,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def main():
    results = []
    # 1. 当前季度合约对 + 动态方向（即 run_latest_calendar_signal.py 默认）
    results.append(run_config(
        ["IC", "IM"], "季度合约对 + 动态方向（IC+IM 组合）",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": True},
        {"dynamic_direction": True, "spread_lookback": {"IC": 40, "IM": 20},
         "spread_z_threshold": {"IC": 1.0, "IM": 2.5}},
    ))

    # 2. 季度合约对 + 固定方向 +1
    results.append(run_config(
        ["IC", "IM"], "季度合约对 + 固定方向 +1（IC+IM 组合）",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": True},
        {"dynamic_direction": False},
    ))

    # 3. 原 dominant @1/@2 映射 + only_quarterly + 固定方向 +1（复现之前最优）
    results.append(run_config(
        ["IC", "IM"], "dominant映射 + only_quarterly + 固定方向 +1（IC+IM 组合）",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": False},
        {"dynamic_direction": False},
    ))

    # 单品种：IM 当前默认
    results.append(run_config(
        ["IM"], "IM 单品种：季度合约对 + 动态方向",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": True},
        {"dynamic_direction": True, "spread_lookback": {"IM": 20},
         "spread_z_threshold": {"IM": 2.5}},
    ))

    # 单品种：IM 固定方向
    results.append(run_config(
        ["IM"], "IM 单品种：季度合约对 + 固定方向 +1",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": True},
        {"dynamic_direction": False},
    ))

    # 单品种：IM dominant + 固定方向
    results.append(run_config(
        ["IM"], "IM 单品种：dominant映射 + only_quarterly + 固定方向 +1",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": False},
        {"dynamic_direction": False},
    ))

    # 单品种：IC 当前默认
    results.append(run_config(
        ["IC"], "IC 单品种：季度合约对 + 动态方向",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": True},
        {"dynamic_direction": True, "spread_lookback": {"IC": 40},
         "spread_z_threshold": {"IC": 1.0}},
    ))

    # 单品种：IC 固定方向
    results.append(run_config(
        ["IC"], "IC 单品种：季度合约对 + 固定方向 +1",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": True},
        {"dynamic_direction": False},
    ))

    # 单品种：IC dominant + 固定方向
    results.append(run_config(
        ["IC"], "IC 单品种：dominant映射 + only_quarterly + 固定方向 +1",
        {"vol_window": 20, "stop_loss_pct": 0.0, "take_profit_pct": 0.0,
         "only_quarterly": True, "roll_before_days": 5, "use_quarterly_pair": False},
        {"dynamic_direction": False},
    ))

    print("\n" + "="*70)
    print("对比汇总")
    print("="*70)
    for r in results:
        print(f"{r['label']:50s} | sharpe={r['sharpe']:.4f} | return={r['total_return']:.4%} | max_dd={r['max_dd']:.4%}")


if __name__ == "__main__":
    main()
