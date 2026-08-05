#!/usr/bin/env python3
"""
商品期货日历价差套利回测（多近空远）

复用 EnhancedCalendarSpreadStrategy + StrategyBacktester + calculate_portfolio_performance
（与 run_optimal_calendar_spread.py 同一审计级口径），对商品主力对 @1/@2 回测：

- direction=+1 多近空远
- vol_threshold=0.15, vol_window=20（88 主连收盘价年化波动过滤）
- roll_before_days=5（近月到期前 5 天平仓）
- only_quarterly=False / use_quarterly_pair=False（直接交易 dominant @1/@2 映射）
- dynamic_direction=False，无止盈止损
- capital=10,000,000, leverage=2.0, capital_ratio=1.0, commission=0.0001
- max_month_span=12（商品主力 1/5/9 轮换，@1/@2 跨度常为 4 个月）

用法：
    python3 run_commodity_calendar_backtest.py              # 全部 5 个品种
    python3 run_commodity_calendar_backtest.py --product I  # 单跑一个品种

输出：docs/commodity_calendar_scan/ 下每品种 pnl/target/trades csv + COMMODITY_SCAN_REPORT.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# pandas 2.0+ 移除 iteritems
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# patch pytz：ClickHouse 服务器时区 'Asia/Beijing' 会导致 load_contract_df 报错
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


START = datetime(2020, 1, 2)
END = datetime(2026, 7, 22)
CAPITAL = 10_000_000
COMMISSION = 0.0001
TRADING_DAYS_PER_YEAR = 240  # 报告夏普口径：日频 × √240
OUTPUT_DIR = Path("/root/quant/cs_developer/docs/commodity_calendar_scan")

# 品种 -> 交易所
PRODUCT_EXCHANGES = {
    "I": "DCE",
    "UR": "CZCE",
    "P": "DCE",
    "RB": "SHFE",
    "JM": "DCE",
}

STRATEGY_PARAMS = {
    "direction": 1,
    "vol_threshold": 0.15,
    "vol_window": 20,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "only_quarterly": False,
    "use_quarterly_pair": False,
    "dynamic_direction": False,
    "roll_before_days": 5,
    "max_month_span": 12,
}


def load_dominant_mapping(product: str, exchange: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = Client(host='localhost')
    keys = [f"{product}88.{exchange}@1", f"{product}88.{exchange}@2"]
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


def filter_contracts_with_bars(contracts: List[str], start: datetime, end: datetime):
    """只保留在 vnpy.bar_data 中有日线数据的合约，返回 (保留列表, 剔除列表)"""
    if not contracts:
        return contracts, []
    client = Client(host='localhost')
    symbols = [c.split('.')[0] for c in contracts]
    rows = client.execute(
        "SELECT DISTINCT symbol, exchange FROM vnpy.bar_data "
        "WHERE symbol IN %(symbols)s AND interval = 'd' "
        "AND datetime >= %(start)s AND datetime <= %(end)s",
        {'symbols': symbols, 'start': start, 'end': end}
    )
    available = {f"{s}.{e}" for s, e in rows}
    kept = [c for c in contracts if c in available]
    dropped = sorted(set(contracts) - set(kept))
    if dropped:
        print(f"    [warn] 无日线数据，剔除合约: {dropped}")
    return kept, dropped


def prepare_product_data(product: str, exchange: str, contract_df: pd.DataFrame) -> Dict:
    mapping = load_dominant_mapping(product, exchange, START, END)
    idx_df = load_bar_df(f"{product}88.{exchange}", Interval.DAILY, START, END)
    if hasattr(idx_df.index, 'tz_localize'):
        idx_df.index = idx_df.index.tz_localize(None)

    k1 = f"{product}88.{exchange}@1"
    k2 = f"{product}88.{exchange}@2"
    contracts = set()
    if k1 in mapping.columns:
        contracts.update(mapping[k1].dropna().unique())
    if k2 in mapping.columns:
        contracts.update(mapping[k2].dropna().unique())
    contracts = sorted({c for c in contracts if isinstance(c, str) and '.' in c})

    # 过滤掉在 bar_data 中无日线数据的合约（如 I2202.DCE 这类映射中出现但未入库的合约），
    # 否则 load_bar_df 对空结果 set_index 会直接报错
    contracts, dropped = filter_contracts_with_bars(contracts, START, END)

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
        'dropped': dropped,
        'history_df': history_df,
        'backtester': backtester,
    }


def run_backtest(product: str, data_cache: Dict, contract_df: pd.DataFrame) -> Dict:
    mapping = data_cache[product]['mapping'].copy()
    idx_close = data_cache[product]['idx_close']
    history_df = data_cache[product]['history_df']
    backtester = data_cache[product]['backtester']

    mapping = mapping.reindex(history_df.index)

    strategy_setting = {
        "products": [product],
        "leverage": 2.0,
        "capital_ratio": 1.0,
        "mapping": mapping,
        "idx_close": idx_close,
        "contract_df": contract_df,
        "history_df": history_df,
        **STRATEGY_PARAMS,
    }

    target_df = backtester.run_backtesting(EnhancedCalendarSpreadStrategy, strategy_setting)

    if len(target_df.columns) == 0:
        # 全期无持仓：target_df 只剩索引，绩效函数无法处理，手工构造零盈亏结果
        overall = pd.DataFrame(index=target_df.index)
        overall["net_pnl"] = 0.0
        overall["pnl"] = 0.0
        overall["balance"] = float(CAPITAL)
        overall["highlevel"] = float(CAPITAL)
        overall["drawdown"] = 0.0
        overall["return"] = 0.0
        overall["ddpercent"] = 0.0
        return {
            'product': product,
            'target_df': target_df,
            'overall': overall,
            'result': {"overall": overall},
            'total_pnl': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_dd': 0.0,
            'sharpe': 0.0,
            'days': len(overall),
        }

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
    annual_return = total_return / total_days * TRADING_DAYS_PER_YEAR
    daily_ret = overall["return"]
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_ret.std() > 0 else 0
    max_dd = overall["ddpercent"].min()

    return {
        'product': product,
        'target_df': target_df,
        'overall': overall,
        'result': result,
        'total_pnl': round(total_pnl, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'days': total_days,
    }


def extract_trades(target_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """从每日目标仓位推导逐合约交易记录"""
    pos_cols = [c for c in target_df.columns if c != "datetime"]
    pos_df = target_df[pos_cols].copy()

    trades = []
    for i in range(1, len(pos_df)):
        dt = pos_df.index[i]
        prev = pos_df.iloc[i - 1]
        curr = pos_df.iloc[i]
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


def extract_spread_round_trips(target_df: pd.DataFrame, history_df: pd.DataFrame,
                               contract_df: pd.DataFrame) -> pd.DataFrame:
    """
    按"价差组合"层面汇总完整往返交易：
    仓位从全平变为非全平 = 开仓；从非全平变为全平或 pair 变化 = 平仓/移仓。
    盈亏用真实收盘价差 × 手数 × 乘数近似（不含日内成交与手续费，仅用于胜率统计）。
    """
    pos_cols = [c for c in target_df.columns if c != "datetime"]
    pos_df = target_df[pos_cols].copy()

    def state_of(row) -> Optional[tuple]:
        items = tuple(sorted((s, int(v)) for s, v in row.items() if int(v) != 0))
        return items if items else None

    def get_mult(vt_symbol: str) -> float:
        symbol = vt_symbol.split(".")[0]
        try:
            size = contract_df.loc[symbol, "size"]
            if pd.notna(size) and size > 0:
                return float(size)
        except Exception:
            pass
        return 1.0

    records = []
    open_state = None
    open_dt = None
    open_prices: Dict[str, float] = {}

    for i in range(len(pos_df)):
        dt = pos_df.index[i]
        state = state_of(pos_df.iloc[i])
        if state == open_state:
            continue
        # 状态变化：先结清旧仓
        if open_state is not None:
            pnl_approx = 0.0
            for vt_symbol, lots in open_state:
                try:
                    exit_price = history_df.loc[dt, (vt_symbol, "close_price")]
                except Exception:
                    exit_price = np.nan
                entry_price = open_prices.get(vt_symbol, np.nan)
                if pd.notna(exit_price) and pd.notna(entry_price):
                    pnl_approx += (exit_price - entry_price) * lots * get_mult(vt_symbol)
            records.append({
                "entry_dt": open_dt,
                "exit_dt": dt,
                "symbols": ",".join(s for s, _ in open_state),
                "holding_days": len(pos_df.loc[open_dt:dt]) - 1,
                "pnl_approx": pnl_approx,
                "rolled": state is not None,
            })
        # 开新仓
        if state is not None:
            open_prices = {}
            for vt_symbol, _ in state:
                try:
                    open_prices[vt_symbol] = history_df.loc[dt, (vt_symbol, "close_price")]
                except Exception:
                    open_prices[vt_symbol] = np.nan
        open_state = state
        open_dt = dt if state is not None else None

    # 期末仍持仓：记一条未平仓记录
    if open_state is not None:
        last_dt = pos_df.index[-1]
        pnl_approx = 0.0
        for vt_symbol, lots in open_state:
            try:
                exit_price = history_df.loc[last_dt, (vt_symbol, "close_price")]
            except Exception:
                exit_price = np.nan
            entry_price = open_prices.get(vt_symbol, np.nan)
            if pd.notna(exit_price) and pd.notna(entry_price):
                pnl_approx += (exit_price - entry_price) * lots * get_mult(vt_symbol)
        records.append({
            "entry_dt": open_dt,
            "exit_dt": None,
            "symbols": ",".join(s for s, _ in open_state),
            "holding_days": len(pos_df.loc[open_dt:]) - 1,
            "pnl_approx": pnl_approx,
            "rolled": False,
        })

    return pd.DataFrame(records)


def compute_position_stats(target_df: pd.DataFrame) -> Dict:
    """持仓天数占比、当前是否持仓、当前持仓明细"""
    pos_cols = [c for c in target_df.columns if c != "datetime"]
    pos_df = target_df[pos_cols]
    has_pos = (pos_df != 0).any(axis=1)
    current = pos_df.iloc[-1]
    current_pos = {s: int(v) for s, v in current.items() if int(v) != 0}
    return {
        "position_days": int(has_pos.sum()),
        "total_days": len(pos_df),
        "position_ratio": float(has_pos.mean()) if len(pos_df) else 0.0,
        "currently_holding": len(current_pos) > 0,
        "current_position": current_pos,
    }


def main():
    parser = argparse.ArgumentParser(description="商品期货日历价差套利回测")
    parser.add_argument("--product", type=str, default=None,
                        help="单跑一个品种（I/UR/P/RB/JM），不指定则跑全部")
    args = parser.parse_args()

    if args.product:
        product = args.product.upper()
        if product not in PRODUCT_EXCHANGES:
            print(f"未知品种: {args.product}，可选: {list(PRODUCT_EXCHANGES)}")
            sys.exit(1)
        products = [product]
    else:
        products = list(PRODUCT_EXCHANGES)

    print("=" * 70)
    print("商品期货日历价差套利回测（多近空远）")
    print(f"品种: {products}")
    print(f"区间: {START.date()} ~ {END.date()}")
    print(f"参数: {STRATEGY_PARAMS}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/2] 逐品种加载数据并回测...")
    summary_rows: List[Dict] = []

    for product in products:
        exchange = PRODUCT_EXCHANGES[product]
        print(f"\n>>> {product} ({exchange}) 加载数据...")
        data_cache = {product: prepare_product_data(product, exchange, contract_df)}
        n_contracts = len(data_cache[product]['contracts'])
        print(f"    合约数: {n_contracts}, 映射天数: {len(data_cache[product]['mapping'])}")

        print(f">>> {product} 运行回测...")
        res = run_backtest(product, data_cache, contract_df)

        history_df = data_cache[product]['history_df']
        prefix = OUTPUT_DIR / f"commodity_{product.lower()}"
        res['target_df'].to_csv(f"{prefix}_target.csv")
        res['overall'][["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(f"{prefix}_pnl.csv")

        trades_df = extract_trades(res['target_df'], history_df)
        trades_df.to_csv(f"{prefix}_trades.csv", index=False)

        round_trips = extract_spread_round_trips(res['target_df'], history_df, contract_df)
        round_trips.to_csv(f"{prefix}_trade_pairs.csv", index=False)

        pos_stats = compute_position_stats(res['target_df'])

        closed = round_trips[round_trips["exit_dt"].notna()] if not round_trips.empty else round_trips
        n_pairs = len(closed)
        win_rate = float((closed["pnl_approx"] > 0).mean()) if n_pairs > 0 else 0.0
        avg_holding = float(closed["holding_days"].mean()) if n_pairs > 0 else 0.0

        print(f"  total_pnl     : {res['total_pnl']:,.2f}")
        print(f"  total_return  : {res['total_return']:.4%}")
        print(f"  annual_return : {res['annual_return']:.4%}")
        print(f"  max_dd        : {res['max_dd']:.4%}")
        print(f"  sharpe(x√{TRADING_DAYS_PER_YEAR}) : {res['sharpe']:.4f}")
        print(f"  trade pairs   : {n_pairs}  win_rate: {win_rate:.2%}  avg_holding: {avg_holding:.1f}d")
        print(f"  position days : {pos_stats['position_days']}/{pos_stats['total_days']} "
              f"({pos_stats['position_ratio']:.1%})  holding_now: {pos_stats['currently_holding']}")

        summary_rows.append({
            "product": product,
            "exchange": exchange,
            "contracts": n_contracts,
            "total_pnl": res['total_pnl'],
            "total_return": res['total_return'],
            "annual_return": res['annual_return'],
            "sharpe": res['sharpe'],
            "max_dd": res['max_dd'],
            "trade_pairs": n_pairs,
            "win_rate": win_rate,
            "avg_holding_days": avg_holding,
            "position_days": pos_stats['position_days'],
            "total_days": pos_stats['total_days'],
            "position_ratio": pos_stats['position_ratio'],
            "currently_holding": pos_stats['currently_holding'],
            "current_position": str(pos_stats['current_position']),
            "dropped_contracts": ",".join(data_cache[product]['dropped']),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "commodity_scan_metrics.csv", index=False)

    write_report(summary_df)

    print("\n" + "=" * 70)
    print("回测完成")
    print(f"结果保存到: {OUTPUT_DIR}")
    print("=" * 70)


def write_report(summary_df: pd.DataFrame):
    lines = []
    lines.append("# 商品期货日历价差扫描报告（多近空远）")
    lines.append("")
    lines.append(f"- 回测区间：{START.date()} ~ {END.date()}")
    lines.append(f"- 策略：EnhancedCalendarSpreadStrategy，direction=+1 多近空远，"
                 f"vol_threshold={STRATEGY_PARAMS['vol_threshold']}（{STRATEGY_PARAMS['vol_window']} 日年化波动，88 主连收盘价），"
                 f"roll_before_days={STRATEGY_PARAMS['roll_before_days']}，only_quarterly=False，"
                 f"use_quarterly_pair=False，dynamic_direction=False，无止盈止损，"
                 f"max_month_span={STRATEGY_PARAMS['max_month_span']}")
    lines.append(f"- 资金：{CAPITAL:,}，leverage=2.0，capital_ratio=1.0，commission=万1（单边按成交额）")
    lines.append("- 口径：信号 T 日收盘计算，T+1 日开盘价成交；"
                 "日盈亏 = start_pos×(close−pre_close) + pos_change×(close−open) − 佣金；"
                 f"夏普 = 日频收益均值/标准差 × √{TRADING_DAYS_PER_YEAR}")
    lines.append("- 胜率/持有天数基于价差组合层面往返交易（pnl 为收盘价近似，不含日内成交价与手续费，仅用于统计）")
    lines.append("")
    lines.append("## 汇总指标")
    lines.append("")
    lines.append("| 品种 | 交易所 | 总盈亏 | 总收益率 | 年化 | 夏普 | 最大回撤 | 交易对数 | 胜率 | 平均持有天数 | 持仓天数占比 | 当前持仓 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['product']} | {r['exchange']} | {r['total_pnl']:,.0f} | "
            f"{r['total_return']:.2%} | {r['annual_return']:.2%} | {r['sharpe']:.2f} | "
            f"{r['max_dd']:.2%} | {int(r['trade_pairs'])} | {r['win_rate']:.1%} | "
            f"{r['avg_holding_days']:.1f} | {r['position_ratio']:.1%} | "
            f"{'是 ' + r['current_position'] if r['currently_holding'] else '否'} |"
        )
    lines.append("")
    lines.append("## 数据质量备注")
    lines.append("")
    lines.append("以下合约出现在主力映射 @1/@2 中，但 vnpy.bar_data 无日线数据，已从回测合约池剔除；"
                 "映射引用这些合约的交易日策略自动空仓（近/远腿无有效收盘价即不开仓）：")
    lines.append("")
    for _, r in summary_df.iterrows():
        dropped = r.get('dropped_contracts', '')
        lines.append(f"- {r['product']}：{dropped if dropped else '无'}")
    lines.append("")
    lines.append("## 文件清单")
    lines.append("")
    lines.append("每品种输出（`commodity_<product>_*.csv`）：")
    lines.append("- `_pnl.csv`：每日 balance/drawdown/net_pnl/return")
    lines.append("- `_target.csv`：每日目标仓位")
    lines.append("- `_trades.csv`：逐合约仓位变动明细")
    lines.append("- `_trade_pairs.csv`：价差组合层面往返交易（含移仓标记）")
    lines.append("")
    lines.append("注：`_trade_pairs.csv` 中 `rolled=True` 表示该笔以移仓（平旧对新开新对）结束，"
                 "策略在主力映射 @1/@2 切换或近月到期前 5 天时自动移仓。")
    lines.append("")

    (OUTPUT_DIR / "COMMODITY_SCAN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {OUTPUT_DIR / 'COMMODITY_SCAN_REPORT.md'}")


if __name__ == "__main__":
    main()
