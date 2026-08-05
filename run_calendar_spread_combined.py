#!/usr/bin/env python3
"""
日历价差套利多品种组合回测（基于 StrategyBacktester）

支持 IC / IM / IF 任意组合，支持 vol 过滤或不过滤，并输出实际交易记录。

用法：
    # 三品种组合，无波动过滤
    python run_calendar_spread_combined.py --products IC,IM,IF --vol-threshold 0

    # 三品种组合，vol>=15%
    python run_calendar_spread_combined.py --products IC,IM,IF --vol-threshold 0.15

    # 单品种
    python run_calendar_spread_combined.py --products IM --vol-threshold 0.15
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, List

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
    parser = argparse.ArgumentParser(description="日历价差套利多品种组合回测")
    parser.add_argument("--products", type=str, default="IC,IM,IF",
                        help="品种组合，逗号分隔（如 IC,IM,IF）")
    parser.add_argument("--vol-threshold", type=float, default=0,
                        help="20 日年化波动阈值；0 表示不过滤（默认不过滤）")
    parser.add_argument("--start", type=str, default="2022-07-22")
    parser.add_argument("--end", type=str, default="2026-07-14")
    parser.add_argument("--capital", type=int, default=CAPITAL)
    parser.add_argument("--commission", type=float, default=COMMISSION)
    parser.add_argument("--output-dir", type=str,
                        default="/root/quant/cs_developer/docs/ic_im_vol_combined_full")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--print-trades", action="store_true", default=True,
                        help="是否打印并保存实际交易记录")
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


def extract_trade_records(target_df: pd.DataFrame, mapping_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    从 target_df 中提取实际交易记录：只记录仓位发生变化的日子。
    """
    records = []
    prev_pos = pd.Series(dtype=float)

    for dt in target_df.index:
        curr_pos = target_df.loc[dt].replace(0, np.nan).dropna()
        if len(curr_pos) == 0 and len(prev_pos) == 0:
            prev_pos = curr_pos
            continue

        pos_change = curr_pos.sub(prev_pos, fill_value=0)
        trades = pos_change[pos_change != 0]

        if len(trades) > 0:
            # 判断交易类型：开新仓 / 平仓 / 换月
            # 根据当前持仓变化推断所属品种和近远月
            for vt_symbol, change in trades.items():
                product = vt_symbol[:2]
                k1 = f"{product}88.CFFEX@1"
                k2 = f"{product}88.CFFEX@2"
                mapping = mapping_dict.get(product)
                if mapping is None or dt not in mapping.index:
                    continue
                near = mapping.loc[dt, k1] if pd.notna(mapping.loc[dt, k1]) else None
                far = mapping.loc[dt, k2] if pd.notna(mapping.loc[dt, k2]) else None
                role = "near" if vt_symbol == near else ("far" if vt_symbol == far else "unknown")
                records.append({
                    'datetime': dt,
                    'product': product,
                    'vt_symbol': vt_symbol,
                    'role': role,
                    'near': near,
                    'far': far,
                    'prev_lots': prev_pos.get(vt_symbol, 0),
                    'curr_lots': curr_pos.get(vt_symbol, 0),
                    'change': change,
                })

        prev_pos = curr_pos

    return pd.DataFrame(records)


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    products = [p.strip().upper() for p in args.products.split(",")]
    vol_label = f"vol{int(args.vol_threshold*100)}" if args.vol_threshold > 0 else "no_vol_filter"

    print(f"\n{'='*70}")
    if args.vol_threshold > 0:
        print(f"组合回测 {','.join(products)} | vol >= {args.vol_threshold*100:.0f}%")
    else:
        print(f"组合回测 {','.join(products)} | 无波动过滤")
    print(f"{'='*70}")

    # 1. 加载每个品种的 @1/@2 映射和波动
    print("\n[1/4] 加载 dominant 映射与波动...")
    mapping_dict: Dict[str, pd.DataFrame] = {}
    vol_dict: Dict[str, pd.Series] = {}
    for product in products:
        mapping = load_dominant_mapping(product, start, end)
        mapping_dict[product] = mapping

        idx_df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, start, end)
        if hasattr(idx_df.index, 'tz_localize'):
            idx_df.index = idx_df.index.tz_localize(None)
        idx_vol = calculate_volatility(idx_df['close_price'], window=20)
        vol_dict[product] = idx_vol

        print(f"  {product}: 映射 {mapping.shape[0]} 天, 波动均值 {idx_vol.mean():.2%}")

    # 2. 收集所有具体合约
    print("\n[2/4] 收集具体合约列表...")
    contracts: Set[str] = set()
    for product in products:
        mapping = mapping_dict[product]
        k1 = f"{product}88.CFFEX@1"
        k2 = f"{product}88.CFFEX@2"
        if k1 in mapping.columns:
            contracts.update(mapping[k1].dropna().unique())
        if k2 in mapping.columns:
            contracts.update(mapping[k2].dropna().unique())
    contracts = {c for c in contracts if isinstance(c, str) and '.' in c}
    print(f"  合约数: {len(contracts)}")

    # 3. 加载合约信息
    print("\n[3/4] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    # 4. 创建回测器并手动加载数据
    print("\n[4/4] 运行 StrategyBacktester...")
    backtester = StrategyBacktester(
        vt_symbols=sorted(contracts),
        interval=Interval.DAILY,
        start=start,
        end=end,
        capital=args.capital,
    )

    history_df = load_history_df(sorted(contracts), Interval.DAILY, start, end)
    backtester.history_df = history_df
    backtester.contract_df = contract_df

    symbol_ix = contract_df.index + "." + contract_df["exchange"]
    backtester.multiplier_series = pd.Series(
        contract_df["size"].values,
        index=symbol_ix
    )

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
    for product in products:
        mapping_dict[product] = mapping_dict[product].reindex(history_df.index)

    strategy_setting = {
        "products": products,
        "vol_threshold": args.vol_threshold,
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mapping_dict,
        "idx_vol": vol_dict,
        "contract_df": contract_df,
        "history_df": backtester.history_df,
    }

    target_df = backtester.run_backtesting(CalendarSpreadStrategy, strategy_setting)

    # 5. 绩效分析
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

    # 6. 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    product_tag = "_".join([p.lower() for p in products])
    prefix = f"{args.output_dir}/{product_tag}_{vol_label}"
    target_df.to_csv(f"{prefix}_target.csv")
    overall[["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
    result["product"].to_csv(f"{prefix}_product.csv")

    metrics = pd.DataFrame([{
        'name': f"{'+'.join(products)} {vol_label} (full)",
        **stats_mapping
    }])
    metrics.to_csv(f"{prefix}_metrics.csv", index=False)

    # 7. 打印实际交易记录
    if args.print_trades:
        trade_df = extract_trade_records(target_df, mapping_dict)
        trade_df.to_csv(f"{prefix}_trades.csv", index=False)
        print("\n" + "="*70)
        print("【实际交易记录样本】（仓位变化）")
        print("="*70)
        print(trade_df.head(30).to_string(index=False))
        if len(trade_df) > 30:
            print(f"\n... 共 {len(trade_df)} 条交易记录，完整记录保存至 {prefix}_trades.csv")
        print("="*70)

    # 8. 打印绩效
    print("\n" + "="*70)
    print(f"【{'+'.join(products)} {vol_label} 完整版绩效】")
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
