#!/usr/bin/env python3
"""
IC/IM 日历价差方向组合测试

测试以下四种方向配置：
1. IC=+1, IM=+1  （都多近空远，基线）
2. IC=-1, IM=-1  （都空近多远，Contango方向）
3. IC=+1, IM=-1  （反向）
4. IC=-1, IM=+1  （反向）

波动阈值：IC=15%, IM=15%
其他参数：vol_window=20, only_quarterly=True, roll_before_days=5
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

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
END = datetime(2026, 7, 14)
CAPITAL = 10_000_000
COMMISSION = 0.0001
OUTPUT_DIR = Path("/root/quant/cs_developer/docs/ic_im_vol_scan")

PRODUCTS = ["IC", "IM"]
VOL_THRESHOLD = {"IC": 0.15, "IM": 0.15}
BASE_PARAMS = {
    "vol_window": 20,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "only_quarterly": True,
    "roll_before_days": 5,
}

DIRECTION_CASES = [
    {"name": "IC+1_IM+1", "direction": {"IC": 1, "IM": 1}},
    {"name": "IC-1_IM-1", "direction": {"IC": -1, "IM": -1}},
    {"name": "IC+1_IM-1", "direction": {"IC": 1, "IM": -1}},
    {"name": "IC-1_IM+1", "direction": {"IC": -1, "IM": +1}},
]


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


def run_case(case: Dict, data_cache: Dict, contract_df: pd.DataFrame) -> Dict:
    name = case["name"]
    direction = case["direction"]
    print(f"\n>>> 运行 {name} ...")

    all_contracts = []
    mapping_dict = {}
    idx_close_dict = {}
    for product in PRODUCTS:
        all_contracts.extend(data_cache[product]['contracts'])
        mapping_dict[product] = data_cache[product]['mapping'].copy()
        idx_close_dict[product] = data_cache[product]['idx_close']

    all_contracts = sorted(set(all_contracts))
    history_df = load_history_df(all_contracts, Interval.DAILY, START, END)

    backtester = StrategyBacktester(
        vt_symbols=all_contracts,
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

    for product in PRODUCTS:
        mapping_dict[product] = mapping_dict[product].reindex(history_df.index)

    strategy_setting = {
        "products": PRODUCTS,
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mapping_dict,
        "idx_close": idx_close_dict,
        "contract_df": contract_df,
        "history_df": history_df,
        "vol_threshold": VOL_THRESHOLD,
        "direction": direction,
        **BASE_PARAMS,
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
    win_rate = profit_days / (profit_days + loss_days) if (profit_days + loss_days) > 0 else 0

    prefix = OUTPUT_DIR / f"direction_{name}"
    target_df.to_csv(f"{prefix}_target.csv")
    overall[["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(f"{prefix}_pnl.csv")

    print(f"  pnl={total_pnl:,.0f}, return={total_return:.2%}, sharpe={sharpe:.3f}, max_dd={max_dd:.2%}")

    return {
        'name': name,
        'direction': direction,
        'total_pnl': round(total_pnl, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_days': profit_days,
        'loss_days': loss_days,
    }


def main():
    print("="*70)
    print("IC/IM 日历价差方向组合测试")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/2] 加载品种数据...")
    data_cache = {}
    for product in PRODUCTS:
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

        data_cache[product] = {
            'mapping': mapping,
            'idx_close': idx_df['close_price'],
            'contracts': contracts,
        }

    results = []
    for case in DIRECTION_CASES:
        res = run_case(case, data_cache, contract_df)
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "direction_combinations_results.csv", index=False)

    print("\n" + "="*70)
    print("结果汇总")
    print("="*70)
    print(results_df[["name", "total_pnl", "total_return", "max_dd", "sharpe", "calmar", "win_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
