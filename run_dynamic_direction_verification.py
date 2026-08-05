#!/usr/bin/env python3
"""
动态方向切换最优参数验证与交易记录核对

分别运行：
- IC-only : lookback=40, z_threshold=1.0
- IM-only : lookback=20, z_threshold=2.5
- IC+IM   : lookback=60, z_threshold=1.5

输出：
- 绩效指标
- 每日方向分布（+1 多近空远 / -1 空近多远）
- 实际交易记录样本（重点检查 -1 方向是否 SELL 近月 / BUY 远月）
"""

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

BEST_CONFIGS = {
    "IC": {"products": ["IC"], "lookback": 40, "z_threshold": 1.0},
    "IM": {"products": ["IM"], "lookback": 20, "z_threshold": 2.5},
    "IC+IM": {"products": ["IC", "IM"], "lookback": 60, "z_threshold": 1.5},
    "IC+IM (per-product)": {
        "products": ["IC", "IM"],
        "lookback": {"IC": 40, "IM": 20},
        "z_threshold": {"IC": 1.0, "IM": 2.5},
    },
    "IC+IM (static)": {
        "products": ["IC", "IM"],
        "lookback": 60,
        "z_threshold": 1.0,
        "dynamic_direction": False,
    },
}


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


def extract_trades(target_df: pd.DataFrame, history_df: pd.DataFrame,
                   mapping_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
            product = vt_symbol[:2]
            k1 = f"{product}88.CFFEX@1"
            k2 = f"{product}88.CFFEX@2"
            mapping = mapping_dict.get(product)
            near = mapping.loc[dt, k1] if mapping is not None and k1 in mapping.columns else None
            far = mapping.loc[dt, k2] if mapping is not None and k2 in mapping.columns else None
            role = "unknown"
            if vt_symbol == near:
                role = "near"
            elif vt_symbol == far:
                role = "far"
            try:
                price = history_df.loc[dt, (vt_symbol, "close_price")]
            except Exception:
                price = np.nan
            trades.append({
                "datetime": dt,
                "product": product,
                "vt_symbol": vt_symbol,
                "role": role,
                "near": near,
                "far": far,
                "prev_pos": int(prev[vt_symbol]),
                "curr_pos": int(curr[vt_symbol]),
                "change": int(change),
                "direction": "BUY" if change > 0 else "SELL",
                "price": price,
            })
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["datetime", "product", "vt_symbol"]).reset_index(drop=True)
    return trades_df


def summarize_direction(target_df: pd.DataFrame, mapping_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """根据每日持仓符号推断每个品种的套利方向"""
    records = []
    for dt in target_df.index:
        for product, mapping in mapping_dict.items():
            k1 = f"{product}88.CFFEX@1"
            if k1 not in mapping.columns or dt not in mapping.index:
                continue
            near = mapping.loc[dt, k1]
            if pd.isna(near) or near not in target_df.columns:
                continue
            lots = target_df.loc[dt, near]
            if lots == 0:
                continue
            records.append({"datetime": dt, "product": product, "direction": int(np.sign(lots))})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df


def run_backtest(products: List[str], data_cache: Dict, contract_df: pd.DataFrame,
                 lookback: int, z_threshold: float,
                 dynamic_direction: bool = True) -> Dict:
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
        "dynamic_direction": dynamic_direction,
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
    trades_df = extract_trades(target_df, history_df, mappings)
    direction_df = summarize_direction(target_df, mappings)

    return {
        'products': products,
        'lookback': lookback,
        'z_threshold': z_threshold,
        'metrics': metrics,
        'target_df': target_df,
        'overall': overall,
        'trades_df': trades_df,
        'direction_df': direction_df,
    }


def print_verification(name: str, res: Dict):
    print("\n" + "="*70)
    print(f"【{name}】动态方向切换验证 | lookback={res['lookback']}, z_threshold={res['z_threshold']}")
    print("="*70)
    m = res['metrics']
    print(f"  total_pnl     : {m['total_pnl']:,.2f}")
    print(f"  total_return  : {m['total_return']:.4%}")
    print(f"  annual_return : {m['annual_return']:.4%}")
    print(f"  max_dd        : {m['max_dd']:.4%}")
    print(f"  sharpe        : {m['sharpe']:.4f}")
    print(f"  calmar        : {m['calmar']:.4f}")
    print(f"  win_rate      : {m['win_rate']:.2%}")
    print(f"  trades count  : {len(res['trades_df'])}")

    direction_df = res['direction_df']
    if not direction_df.empty:
        for product in direction_df['product'].unique():
            sub = direction_df[direction_df['product'] == product]
            plus1 = (sub['direction'] == 1).sum()
            minus1 = (sub['direction'] == -1).sum()
            print(f"  {product} 方向分布: +1(多近空远)={plus1}天, -1(空近多远)={minus1}天")

    trades = res['trades_df']
    if not trades.empty:
        # 打印所有 -1 方向的开仓记录（用于核对逻辑）
        open_trades = trades[trades['prev_pos'] == 0]
        minus1_opens = open_trades[open_trades.apply(
            lambda r: (r['product'] == r['vt_symbol'][:2]) and r['role'] == 'near' and r['change'] < 0,
            axis=1
        )]
        print("\n  --- -1 方向开仓记录样本（近月 SELL / 远月 BUY）---")
        print(minus1_opens.head(20).to_string(index=False))
        if len(minus1_opens) > 20:
            print(f"  ... 共 {len(minus1_opens)} 条 -1 方向开仓记录")

        # 保存完整交易记录
        prefix = OUTPUT_DIR / f"dynamic_{name.lower().replace('+', '_')}"
        trades.to_csv(f"{prefix}_trades.csv", index=False)
        res['target_df'].to_csv(f"{prefix}_target.csv")
        res['overall'][["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(f"{prefix}_pnl.csv")
        print(f"\n  [OK] 交易记录保存至 {prefix}_trades.csv")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("动态方向切换最优参数验证")
    print("="*70)

    print("\n[1/2] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/2] 加载品种数据...")
    data_cache = {p: prepare_product_data(p, contract_df) for p in PRODUCTS}

    for name, cfg in BEST_CONFIGS.items():
        products = cfg["products"]
        dd = cfg.get("dynamic_direction", True)
        res = run_backtest(products, data_cache, contract_df, cfg["lookback"], cfg["z_threshold"], dynamic_direction=dd)
        print_verification(name, res)

    print("\n" + "="*70)
    print("验证完成")
    print("="*70)


if __name__ == "__main__":
    main()
