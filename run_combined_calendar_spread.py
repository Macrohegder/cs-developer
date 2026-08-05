#!/usr/bin/env python3
"""
三品种组合日历价差套利完整版回测

按品种差异化波动阈值：
- IF: 25%
- IC: 15%
- IM: 15%

其他参数：
- vol_window = 20
- only_quarterly = True
- roll_before_days = 5
- stop_loss_pct = take_profit_pct = 0
- leverage = 2.0, capital_ratio = 1.0
- 资金按品种均分

输出：
- 组合绩效指标
- 每日目标仓位
- 交易记录
- 净值曲线
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Set

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

PRODUCTS = ["IF", "IC", "IM"]
VOL_THRESHOLD = {
    "IF": 0.25,
    "IC": 0.15,
    "IM": 0.15,
}
BASE_PARAMS = {
    "vol_window": 20,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "only_quarterly": True,
    "roll_before_days": 5,
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


def main():
    print("="*70)
    print("三品种组合日历价差套利完整版回测")
    print(f"品种: {PRODUCTS}")
    print(f"按品种波动阈值: {VOL_THRESHOLD}")
    print(f"其他参数: {BASE_PARAMS}")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/3] 加载品种数据...")
    all_contracts = set()
    mapping_dict = {}
    idx_close_dict = {}

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
        contracts = {c for c in contracts if isinstance(c, str) and '.' in c}
        all_contracts.update(contracts)

        mapping_dict[product] = mapping
        idx_close_dict[product] = idx_df['close_price']
        print(f"  {product}: {len(contracts)} 个合约")

    all_contracts = sorted(all_contracts)
    print(f"\n总合约数: {len(all_contracts)}")

    print("\n[3/3] 运行组合回测...")
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

    # 对齐 mapping
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

    # 保存结果
    prefix = OUTPUT_DIR / "combined_if25_ic15_im15"
    target_df.to_csv(f"{prefix}_target.csv")
    overall[["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(f"{prefix}_pnl.csv")

    metrics = pd.DataFrame([{
        'name': 'combined IF25%/IC15%/IM15%',
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
    }])
    metrics.to_csv(f"{prefix}_metrics.csv", index=False)

    # 推导交易记录
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
    trades_df.to_csv(f"{prefix}_trades.csv", index=False)

    # 绘制净值曲线
    plt.figure(figsize=(12, 6))
    nav = overall["balance"] / overall["balance"].iloc[0]
    plt.plot(nav.index, nav, label=f"Combined (Sharpe={sharpe:.2f})")
    plt.title("Combined Calendar Spread Strategy NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_nav.png", dpi=150)
    plt.close()

    print("\n" + "="*70)
    print("【三品种组合绩效】")
    print("="*70)
    print(f"  total_pnl     : {total_pnl:,.2f}")
    print(f"  total_return  : {total_return:.4%}")
    print(f"  annual_return : {annual_return:.4%}")
    print(f"  max_dd        : {max_dd:.4%}")
    print(f"  sharpe        : {sharpe:.4f}")
    print(f"  calmar        : {calmar:.4f}")
    print(f"  win_rate      : {win_rate:.2%}")
    print(f"  profit/loss   : {profit_days} / {loss_days}")
    print(f"  trades count  : {len(trades_df)}")
    print("="*70)
    print(f"\n[OK] 结果保存到: {prefix}_*")


if __name__ == "__main__":
    main()
