#!/usr/bin/env python3
"""
获取 IC / IM 日历价差策略的最新开仓信号（季度合约对 + 动态方向切换版本）

默认使用季度合约对模式：自动选择“最近季月 vs 下一季月”，
例如当前（2026-07）为 IC2609-IC2612、IM2609-IM2612。

使用各自最优动态参数：
- IC: lookback=40, z_threshold=1.0
- IM: lookback=20, z_threshold=2.5

输出最新交易日的：
- 近月/远月合约
- z-score
- 套利方向
- 目标手数
"""

import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

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

PRODUCTS = ["IM"]
VOL_THRESHOLD = {"IC": 0.15, "IM": 0.15}
# 使用历史绩效最好的模式：
# - dominant @1/@2 映射选择合约对
# - only_quarterly=True：只在近月主力是季月（3/6/9/12）时交易
# - 固定方向 +1（多近空远），不开启动态方向切换
# - 波动率 >=15% 才开仓
BASE_PARAMS = {
    "vol_window": 20,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "only_quarterly": True,
    "roll_before_days": 5,
    "use_quarterly_pair": False,
    "dynamic_direction": False,
}
DYNAMIC_PARAMS = {}


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
    """加载该品种所有未过期的 CFFEX 期货合约（用于季度合约对模式）"""
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
    """根据 88 指数数据确定最新可用交易日"""
    latest_dates = []
    for product in PRODUCTS:
        df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, datetime.today())
        if hasattr(df.index, 'tz_localize'):
            df.index = df.index.tz_localize(None)
        latest_dates.append(df.index[-1])
    latest = min(latest_dates)
    return latest.to_pydatetime() if hasattr(latest, 'to_pydatetime') else latest


def _get_expiry_date(vt_symbol: str, contract_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    symbol = vt_symbol.split(".")[0]
    try:
        expiry = contract_df.loc[symbol, "expiry"]
        if pd.notna(expiry):
            dt = pd.to_datetime(expiry, errors='coerce')
            if pd.notna(dt):
                return dt
    except Exception:
        pass
    return None


def _get_listed_date(vt_symbol: str, contract_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    symbol = vt_symbol.split(".")[0]
    try:
        listed = contract_df.loc[symbol, "listed"]
        if pd.notna(listed):
            dt = pd.to_datetime(listed, errors='coerce')
            if pd.notna(dt):
                return dt
    except Exception:
        pass
    return None


def _get_expiry_month(vt_symbol: str) -> Tuple[int, int]:
    code = vt_symbol.split(".")[0]
    if len(code) < 6:
        return (0, 0)
    year_short = int(code[-4:-2])
    month = int(code[-2:])
    return (2000 + year_short, month)


def _month_span(m1: Tuple[int, int], m2: Tuple[int, int]) -> int:
    return (m2[0] - m1[0]) * 12 + (m2[1] - m1[1])


def build_quarterly_pair_df(product: str, history_df: pd.DataFrame,
                            contract_df: pd.DataFrame) -> pd.DataFrame:
    """从 history_df 中自动挑选最近季月和下一季月合约对"""
    close_df = history_df.xs("close_price", axis=1, level=1)
    symbols = [s for s in close_df.columns
               if s.startswith(product) and "." in s and len(s.split(".")[0]) == 6]

    records = []
    for dt in close_df.index:
        candidates = []
        for s in symbols:
            price = close_df.loc[dt, s]
            if pd.isna(price):
                continue
            expiry = _get_expiry_date(s, contract_df)
            listed = _get_listed_date(s, contract_df)
            if expiry is None or expiry <= dt or listed is None or dt < listed:
                continue
            month = _get_expiry_month(s)[1]
            if month not in (3, 6, 9, 12):
                continue
            candidates.append((expiry, s))

        if len(candidates) < 2:
            records.append((dt, None, None))
            continue

        candidates.sort()
        near = candidates[0][1]
        far = candidates[1][1]

        near_month = _get_expiry_month(near)
        far_month = _get_expiry_month(far)
        span = _month_span(near_month, far_month)
        if span <= 0 or span > 3:
            records.append((dt, None, None))
        else:
            records.append((dt, near, far))

    return pd.DataFrame(records, columns=["datetime", "near", "far"]).set_index("datetime")


def prepare_product_data(product: str, contract_df: pd.DataFrame, end: datetime) -> Dict:
    mapping = load_dominant_mapping(product, START, end)
    idx_df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, end)
    if hasattr(idx_df.index, 'tz_localize'):
        idx_df.index = idx_df.index.tz_localize(None)

    k1 = f"{product}88.CFFEX@1"
    k2 = f"{product}88.CFFEX@2"

    if BASE_PARAMS.get("use_quarterly_pair", False):
        # 季度合约对模式：加载该品种所有 CFFEX 合约
        all_contracts = get_all_product_contracts(product, contract_df, START)
        extra = set()
        if k1 in mapping.columns:
            extra.update(mapping[k1].dropna().unique())
        if k2 in mapping.columns:
            extra.update(mapping[k2].dropna().unique())
        contracts = sorted(set(all_contracts) | {c for c in extra if isinstance(c, str) and '.' in c})
    else:
        # dominant 映射模式：只加载 @1/@2 映射中出现过的合约
        extra = set()
        if k1 in mapping.columns:
            extra.update(mapping[k1].dropna().unique())
        if k2 in mapping.columns:
            extra.update(mapping[k2].dropna().unique())
        contracts = sorted({c for c in extra if isinstance(c, str) and '.' in c})

    history_df = load_history_df(contracts, Interval.DAILY, START, end)

    backtester = StrategyBacktester(
        vt_symbols=contracts,
        interval=Interval.DAILY,
        start=START,
        end=end,
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
    return {
        'total_pnl': round(total_pnl, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
    }


def main():
    print("="*70)
    print("IC / IM 日历价差 — dominant 映射 + 季月过滤 + 固定方向")
    print("="*70)

    latest = get_latest_common_date()
    print(f"\n数据库最新可用交易日: {latest.date()}")

    print("\n[1/3] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/3] 加载品种数据...")
    data_cache = {p: prepare_product_data(p, contract_df, latest) for p in PRODUCTS}

    print("\n[3/3] 运行回测...")
    all_contracts = sorted(set(
        c for p in PRODUCTS for c in data_cache[p]['contracts']
    ))
    history_df = load_history_df(all_contracts, Interval.DAILY, START, latest)

    backtester = StrategyBacktester(
        vt_symbols=all_contracts,
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

    mappings = {}
    idx_closes = {}
    for p in PRODUCTS:
        mappings[p] = data_cache[p]['mapping'].copy().reindex(history_df.index)
        idx_closes[p] = data_cache[p]['idx_close']

    strategy_setting = {
        "products": PRODUCTS,
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mappings,
        "idx_close": idx_closes,
        "contract_df": contract_df,
        "history_df": history_df,
        "vol_threshold": VOL_THRESHOLD,
        **BASE_PARAMS,
        **DYNAMIC_PARAMS,
    }

    target_df = backtester.run_backtesting(EnhancedCalendarSpreadStrategy, strategy_setting)

    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=COMMISSION,
        capital=CAPITAL,
        plot_chart=False,
    )
    metrics = calculate_metrics(result["overall"])

    print("\n" + "="*70)
    print(f"回测绩效（截至 {latest.date()}）")
    print("="*70)
    print(f"  total_pnl     : {metrics['total_pnl']:,.2f}")
    print(f"  total_return  : {metrics['total_return']:.4%}")
    print(f"  annual_return : {metrics['annual_return']:.4%}")
    print(f"  max_dd        : {metrics['max_dd']:.4%}")
    print(f"  sharpe        : {metrics['sharpe']:.4f}")
    print(f"  calmar        : {metrics['calmar']:.4f}")

    # 提取最新交易日的信号
    last_dt = target_df.index[-1]
    print("\n" + "="*70)
    print(f"【最新交易日 {last_dt.date()} 开仓建议】")
    print("="*70)

    close_df = history_df.xs("close_price", axis=1, level=1)

    for product in PRODUCTS:
        if BASE_PARAMS.get("use_quarterly_pair", False):
            # 季度合约对模式：自动选择最近季月 vs 下一季月
            pair_df = build_quarterly_pair_df(product, history_df, contract_df)
            if last_dt not in pair_df.index:
                print(f"  {product}: 无季度合约对数据")
                continue
            near = pair_df.loc[last_dt, "near"]
            far = pair_df.loc[last_dt, "far"]
            if pd.isna(near) or pd.isna(far):
                print(f"  {product}: 未选出有效季度合约对")
                continue
        else:
            # dominant 映射模式：使用主力 / 次主力映射
            k1 = f"{product}88.CFFEX@1"
            k2 = f"{product}88.CFFEX@2"
            mapping = mappings[product]
            if last_dt not in mapping.index:
                print(f"  {product}: 无映射数据")
                continue
            near = mapping.loc[last_dt, k1]
            far = mapping.loc[last_dt, k2]
            if pd.isna(near) or pd.isna(far):
                print(f"  {product}: 数据不完整")
                continue

        near_lots = target_df.loc[last_dt, near] if near in target_df.columns else 0
        far_lots = target_df.loc[last_dt, far] if far in target_df.columns else 0
        direction = int(np.sign(near_lots)) if near_lots != 0 else 0

        near_price = close_df.loc[last_dt, near] if near in close_df.columns else np.nan
        far_price = close_df.loc[last_dt, far] if far in close_df.columns else np.nan
        spread_pct = (near_price - far_price) / near_price if near_price and not pd.isna(near_price) else np.nan

        print(f"\n  {product}:")
        print(f"    近月合约: {near}")
        print(f"    远月合约: {far}")
        print(f"    近月价格: {near_price:.2f}")
        print(f"    远月价格: {far_price:.2f}")
        print(f"    spread_pct: {spread_pct:.4%}")
        print(f"    方向: {direction} ({'+1 多近空远' if direction == 1 else ('-1 空近多远' if direction == -1 else '空仓')})")
        print(f"    近月目标手数: {int(near_lots)}")
        print(f"    远月目标手数: {int(far_lots)}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
