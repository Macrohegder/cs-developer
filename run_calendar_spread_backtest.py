#!/usr/bin/env python3
"""
日历价差套利完整版回测入口（基于 StrategyBacktester）

支持 IC / IM / IF 三个股指期货品种，与简化框架做交叉验证。

用法：
    python run_calendar_spread_backtest.py --product IM --vol-threshold 0.15
    python run_calendar_spread_backtest.py --product IC --vol-threshold 0.15
    python run_calendar_spread_backtest.py --product IF --vol-threshold 0.15
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Set

import pandas as pd
import numpy as np

# pandas 2.0+ 移除 iteritems，但 vnpy_alpharesearch 旧代码仍调用
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# patch pytz 对 Asia/Beijing 的支持
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
from vnpy_alpharesearch.utility import load_bar_df
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance
from vnpy_alpharesearch.utility import load_history_df

from strategies.calendar_spread_strategy import CalendarSpreadStrategy


CAPITAL = 10_000_000
COMMISSION = 0.0001


def parse_args():
    parser = argparse.ArgumentParser(description="日历价差套利完整版回测")
    parser.add_argument("--product", type=str, default="IM", choices=["IC", "IM", "IF"],
                        help="品种：IC、IM 或 IF")
    parser.add_argument("--vol-threshold", type=float, default=0.15,
                        help="20 日年化波动阈值（默认 0.15=15%%）")
    parser.add_argument("--start", type=str, default="2022-07-22")
    parser.add_argument("--end", type=str, default="2026-07-14")
    parser.add_argument("--capital", type=int, default=CAPITAL)
    parser.add_argument("--commission", type=float, default=COMMISSION)
    parser.add_argument("--output-dir", type=str,
                        default="/root/quant/cs_developer/docs/ic_im_vol_filtered_full")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def load_dominant_mapping(product: str, start: datetime, end: datetime) -> pd.DataFrame:
    """从 ClickHouse 加载 @1/@2 主力映射"""
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


def calculate_volatility(price_series: pd.Series, window: int = 20) -> pd.Series:
    """计算年化波动率"""
    ret = price_series.pct_change()
    vol = ret.rolling(window=window, min_periods=window//2).std() * np.sqrt(250)
    return vol


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    product = args.product

    print(f"\n{'='*70}")
    print(f"{product}-only 日历价差套利完整版回测 | vol >= {args.vol_threshold*100:.0f}%")
    print(f"{'='*70}")

    # 1. 加载 @1/@2 映射
    print("\n[1/5] 加载 dominant 映射...")
    mapping = load_dominant_mapping(product, start, end)
    k1 = f"{product}88.CFFEX@1"
    k2 = f"{product}88.CFFEX@2"
    print(f"  映射记录: {mapping.shape[0]} 天")

    # 2. 加载 88 指数并计算波动
    print("\n[2/5] 加载 {product}88 指数并计算波动...")
    idx_df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, start, end)
    if hasattr(idx_df.index, 'tz_localize'):
        idx_df.index = idx_df.index.tz_localize(None)
    idx_vol = calculate_volatility(idx_df['close_price'], window=20)
    print(f"  波动均值: {idx_vol.mean():.2%}, >=阈值天数: {(idx_vol>=args.vol_threshold).sum()}")

    # 3. 收集所有具体合约
    print("\n[3/5] 收集具体合约列表...")
    contracts: Set[str] = set()
    if k1 in mapping.columns:
        contracts.update(mapping[k1].dropna().unique())
    if k2 in mapping.columns:
        contracts.update(mapping[k2].dropna().unique())
    contracts = {c for c in contracts if isinstance(c, str) and '.' in c}
    print(f"  合约数: {len(contracts)}")

    # 4. 加载合约信息
    print("\n[4/5] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()

    # 5. 创建回测器并手动加载数据（绕过 DominantManager，因为 vt_symbols 是具体合约而非 88 指数）
    print("\n[5/5] 运行 StrategyBacktester...")
    backtester = StrategyBacktester(
        vt_symbols=sorted(contracts),
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=args.capital,
    )

    # 手动加载历史数据
    history_df = load_history_df(sorted(contracts), Interval.DAILY, start, end)
    backtester.history_df = history_df

    # 加载合约信息
    backtester.contract_df = contract_df

    # 提取合约乘数（数据库原始列名为 size）
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)
    symbol_ix = contract_df.index + "." + contract_df["exchange"]
    backtester.multiplier_series = pd.Series(
        contract_df["size"].values,
        index=symbol_ix
    )

    # 计算 intraday_change 和 close_change
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

    # 把 mapping 对齐到 history_df 的日期
    mapping = mapping.reindex(history_df.index)

    strategy_setting = {
        "product": product,
        "vol_threshold": args.vol_threshold,
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mapping,
        "idx_vol": idx_vol,
        "contract_df": contract_df,
        "history_df": backtester.history_df,
    }

    target_df = backtester.run_backtesting(CalendarSpreadStrategy, strategy_setting)

    # 6. 绩效分析
    result = calculate_portfolio_performance(
        target_df=target_df,
        interval=Interval.DAILY,
        commission=args.commission,
        capital=args.capital,
        plot_chart=args.plot,
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

    stats_mapping = {
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

    # 7. 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prefix = f"{args.output_dir}/{product.lower()}_only_vol{int(args.vol_threshold*100)}"
    target_df.to_csv(f"{prefix}_target.csv")
    overall[["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
    result["product"].to_csv(f"{prefix}_product.csv")

    metrics = pd.DataFrame([{
        'name': f"{product}-only vol>={int(args.vol_threshold*100)}% (full)",
        **stats_mapping
    }])
    metrics.to_csv(f"{prefix}_metrics.csv", index=False)

    # 8. 打印
    print("\n" + "="*70)
    print(f"【{product}-only vol>={args.vol_threshold*100:.0f}% 完整版绩效】")
    print("="*70)
    print(f"  total_pnl          : {stats_mapping['total_pnl']:,.2f}")
    print(f"  total_return       : {stats_mapping['total_return']:.4%}")
    print(f"  annual_return      : {stats_mapping['annual_return']:.4%}")
    print(f"  max_dd             : {stats_mapping['max_dd']:.4%}")
    print(f"  sharpe             : {stats_mapping['sharpe']:.4f}")
    print(f"  calmar             : {stats_mapping['calmar']:.4f}")
    print(f"  win_rate           : {stats_mapping['win_rate']:.2%}")
    print(f"  profit/loss days   : {stats_mapping['profit_days']} / {stats_mapping['loss_days']}")
    print(f"  days               : {stats_mapping['days']}")
    print("="*70)
    print(f"\n[OK] 结果保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
