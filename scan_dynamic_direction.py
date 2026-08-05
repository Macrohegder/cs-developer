#!/usr/bin/env python3
"""
动态方向切换参数扫描（基于期限结构斜率 z-score）

遍历 spread_lookback 与 spread_z_threshold，分别测试 IC / IM / IC+IM 组合。
输出汇总表到 docs/ic_im_vol_scan/dynamic_direction_scan_summary.csv。

用法：
    python scan_dynamic_direction.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

LOOKBACKS = [20, 40, 60]
Z_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]


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


def calculate_metrics(overall: pd.DataFrame) -> Dict:
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
    return {
        'total_pnl': round(total_pnl, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_days': profit_days,
        'loss_days': loss_days,
        'days': total_days,
    }


def extract_trades(target_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    pos_cols = [c for c in target_df.columns if c != "datetime"]
    trades = []
    for i in range(1, len(target_df)):
        dt = target_df.index[i]
        prev = target_df.iloc[i - 1]
        curr = target_df.iloc[i]
        for vt_symbol in pos_cols:
            change = curr[vt_symbol] - prev[vt_symbol]
            if change == 0:
                continue
            try:
                price = history_df.loc[dt, (vt_symbol, "close_price")]
            except Exception:
                price = np.nan
            trades.append({
                "datetime": dt,
                "vt_symbol": vt_symbol,
                "prev_pos": int(prev[vt_symbol]),
                "curr_pos": int(curr[vt_symbol]),
                "change": int(change),
                "direction": "BUY" if change > 0 else "SELL",
                "price": price,
            })
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["datetime", "vt_symbol"]).reset_index(drop=True)
    return trades_df


def run_single_backtest(
    products: List[str],
    data_cache: Dict,
    contract_df: pd.DataFrame,
    lookback: int,
    z_threshold: float,
) -> Dict:
    mappings = {}
    idx_closes = {}
    all_contracts = []
    for product in products:
        data_cache[product]['backtester'].history_df = data_cache[product]['history_df']
        mappings[product] = data_cache[product]['mapping'].copy().reindex(
            data_cache[product]['history_df'].index
        )
        idx_closes[product] = data_cache[product]['idx_close']
        all_contracts.extend(data_cache[product]['contracts'])

    if len(products) == 1:
        backtester = data_cache[products[0]]['backtester']
        history_df = data_cache[products[0]]['history_df']
    else:
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

        for product in products:
            mappings[product] = mappings[product].reindex(history_df.index)

    strategy_setting = {
        "products": products,
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mappings,
        "idx_close": idx_closes,
        "contract_df": contract_df,
        "history_df": history_df,
        "vol_threshold": {p: VOL_THRESHOLD[p] for p in products},
        **BASE_PARAMS,
        "dynamic_direction": True,
        "spread_lookback": lookback,
        "spread_z_threshold": z_threshold,
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
    metrics = calculate_metrics(overall)
    metrics['lookback'] = lookback
    metrics['z_threshold'] = z_threshold
    metrics['products'] = "+".join(products)

    # 统计交易记录与方向变化
    trades_df = extract_trades(target_df, history_df)
    metrics['trades_count'] = len(trades_df)

    return {
        'metrics': metrics,
        'target_df': target_df,
        'overall': overall,
        'history_df': history_df,
        'trades_df': trades_df,
    }


def main():
    print("="*70)
    print("动态方向切换参数扫描")
    print(f"参数范围: lookback={LOOKBACKS}, z_threshold={Z_THRESHOLDS}")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/2] 加载品种数据...")
    data_cache = {p: prepare_product_data(p, contract_df) for p in PRODUCTS}

    results = []
    scan_tasks = []
    for products in ([["IC"], ["IM"], ["IC", "IM"]]):
        for lookback in LOOKBACKS:
            for z_threshold in Z_THRESHOLDS:
                scan_tasks.append((products, lookback, z_threshold))

    total = len(scan_tasks)
    for idx, (products, lookback, z_threshold) in enumerate(scan_tasks, 1):
        print(f"\n>>> [{idx}/{total}] products={'+'.join(products)}, "
              f"lookback={lookback}, z_threshold={z_threshold}")
        try:
            res = run_single_backtest(products, data_cache, contract_df, lookback, z_threshold)
            results.append(res['metrics'])
            print(f"  sharpe={res['metrics']['sharpe']:.4f}, "
                  f"total_pnl={res['metrics']['total_pnl']:,.2f}, "
                  f"trades={res['metrics']['trades_count']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    summary = pd.DataFrame(results)
    cols = ['products', 'lookback', 'z_threshold', 'total_pnl', 'total_return',
            'annual_return', 'max_dd', 'sharpe', 'calmar', 'win_rate',
            'profit_days', 'loss_days', 'days', 'trades_count']
    summary = summary[[c for c in cols if c in summary.columns]]
    summary = summary.sort_values(['products', 'sharpe'], ascending=[True, False])
    summary.to_csv(OUTPUT_DIR / "dynamic_direction_scan_summary.csv", index=False)

    print("\n" + "="*70)
    print("扫描完成，汇总如下：")
    print("="*70)
    print(summary.to_string(index=False))
    print(f"\n[OK] 结果保存到: {OUTPUT_DIR / 'dynamic_direction_scan_summary.csv'}")


if __name__ == "__main__":
    main()
