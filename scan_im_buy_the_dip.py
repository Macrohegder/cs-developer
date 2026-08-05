#!/usr/bin/env python3
"""
IM 跨月价差 long-only buy-the-dip 参数扫描与基准对比。

对比三类策略：
1. static: 永远多近月、空远月
2. dynamic: 高 z-score 时允许反手
3. buy_the_dip: 仅在低 z-score 且正 carry regime 下做多近空远
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
PRODUCT = "IM"
OUTPUT_DIR = Path("/root/quant/cs_developer/docs/im_buy_the_dip_scan")

BASE_PARAMS = {
    "products": [PRODUCT],
    "leverage": 2.0,
    "capital_ratio": 1.0,
    "vol_threshold": {PRODUCT: 0.15},
    "vol_window": 20,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "only_quarterly": True,
    "roll_before_days": 5,
}

DYNAMIC_LOOKBACKS = [20, 40, 60]
DYNAMIC_Z_THRESHOLDS = [1.0, 1.5, 2.0, 2.5]

DIP_LOOKBACKS = [20, 40, 60]
DIP_ENTRY_ZS = [-1.0, -1.5, -2.0]
DIP_EXIT_ZS = [-0.5, -0.25, 0.0]
REGIME_LOOKBACKS = [60, 90]


def load_dominant_mapping(product: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = Client(host="localhost")
    keys = [f"{product}88.CFFEX@1", f"{product}88.CFFEX@2"]
    result = client.execute(
        "SELECT datetime, key, value FROM vnpy.vnpy_dominant_contract "
        "WHERE key IN %(keys)s AND datetime >= %(start)s AND datetime <= %(end)s "
        "ORDER BY datetime",
        {"keys": keys, "start": start, "end": end},
    )
    df = pd.DataFrame(result, columns=["datetime", "key", "value"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.drop_duplicates(subset=["datetime", "key"])
    return df.pivot(index="datetime", columns="key", values="value")


def prepare_product_data(product: str, contract_df: pd.DataFrame) -> Dict:
    mapping = load_dominant_mapping(product, START, END)
    idx_df = load_bar_df(f"{product}88.CFFEX", Interval.DAILY, START, END)
    if hasattr(idx_df.index, "tz_localize"):
        idx_df.index = idx_df.index.tz_localize(None)

    k1 = f"{product}88.CFFEX@1"
    k2 = f"{product}88.CFFEX@2"
    contracts = set()
    if k1 in mapping.columns:
        contracts.update(mapping[k1].dropna().unique())
    if k2 in mapping.columns:
        contracts.update(mapping[k2].dropna().unique())
    contracts = sorted({c for c in contracts if isinstance(c, str) and "." in c})

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
        "mapping": mapping.reindex(history_df.index),
        "idx_close": idx_df["close_price"],
        "contracts": contracts,
        "history_df": history_df,
        "backtester": backtester,
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
        "total_pnl": round(total_pnl, 2),
        "total_return": total_return,
        "annual_return": annual_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_days": profit_days,
        "loss_days": loss_days,
        "days": total_days,
    }


def extract_trades_count(target_df: pd.DataFrame) -> int:
    pos_cols = [c for c in target_df.columns if c != "datetime"]
    if not pos_cols:
        return 0
    changes = target_df[pos_cols].diff().fillna(0)
    return int((changes != 0).sum().sum())


def build_zero_overall(index: pd.Index) -> pd.DataFrame:
    """零仓位兜底绩效，避免无交易参数组合导致分析函数报错。"""
    if len(index) == 0:
        index = pd.DatetimeIndex([START])
    return pd.DataFrame(
        {
            "balance": CAPITAL,
            "drawdown": 0.0,
            "ddpercent": 0.0,
            "net_pnl": 0.0,
            "return": 0.0,
        },
        index=index,
    )


def run_strategy(data_cache: Dict, contract_df: pd.DataFrame, mode: str, **params) -> Dict:
    strategy_setting = {
        **BASE_PARAMS,
        "mapping": data_cache["mapping"],
        "idx_close": data_cache["idx_close"],
        "contract_df": contract_df,
        "history_df": data_cache["history_df"],
    }
    strategy_setting.update(params)

    target_df = data_cache["backtester"].run_backtesting(
        EnhancedCalendarSpreadStrategy,
        strategy_setting,
    )

    pos_cols = [c for c in target_df.columns if c != "datetime"]
    if not pos_cols:
        overall = build_zero_overall(target_df.index)
    else:
        result = calculate_portfolio_performance(
            target_df=target_df,
            interval=Interval.DAILY,
            commission=COMMISSION,
            capital=CAPITAL,
            plot_chart=False,
        )
        overall = result["overall"]

    metrics = calculate_metrics(overall)
    metrics["mode"] = mode
    metrics["trades_count"] = extract_trades_count(target_df)
    for key, value in params.items():
        metrics[key] = value

    return {
        "metrics": metrics,
        "target_df": target_df,
        "overall": overall,
    }


def save_best_artifacts(name: str, result: Dict):
    prefix = OUTPUT_DIR / f"best_{name}"
    result["target_df"].to_csv(f"{prefix}_target.csv")
    result["overall"][["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(
        f"{prefix}_pnl.csv"
    )


def build_report(best_df: pd.DataFrame) -> str:
    lines = [
        "# IM Buy-The-Dip 对比报告",
        "",
        f"- 回测区间：{START.date()} ~ {END.date()}",
        "- 框架：StrategyBacktester + EnhancedCalendarSpreadStrategy",
        "- 执行时序：当日收盘生成信号，下一根 bar 生效",
        "",
        "## 最优结果",
        "",
        "| 模式 | total_pnl | total_return | max_dd | sharpe | trades_count | 关键参数 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in best_df.iterrows():
        key_parts = []
        for key in ["spread_lookback", "spread_z_threshold", "dip_entry_z", "dip_exit_z", "regime_lookback"]:
            if key in row and pd.notna(row[key]):
                key_parts.append(f"{key}={row[key]}")
        key_text = ", ".join(key_parts) if key_parts else "-"
        lines.append(
            f"| {row['mode']} | {row['total_pnl']:.2f} | {row['total_return']:.2%} | "
            f"{row['max_dd']:.2%} | {row['sharpe']:.4f} | {int(row['trades_count'])} | {key_text} |"
        )

    return "\n".join(lines) + "\n"


def main():
    print("=" * 70)
    print("IM Buy-The-Dip 参数扫描与基准对比")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)
    data_cache = prepare_product_data(PRODUCT, contract_df)

    results = []
    best_results = {}

    print("\n[1/3] 运行静态基准...")
    static_res = run_strategy(
        data_cache,
        contract_df,
        mode="static",
        dynamic_direction=False,
        buy_the_dip_mode=False,
    )
    results.append(static_res["metrics"])
    best_results["static"] = static_res
    print(
        f"  static sharpe={static_res['metrics']['sharpe']:.4f}, "
        f"total_pnl={static_res['metrics']['total_pnl']:,.2f}"
    )

    print("\n[2/3] 扫描动态方向策略...")
    best_dynamic = None
    for lookback in DYNAMIC_LOOKBACKS:
        for z_threshold in DYNAMIC_Z_THRESHOLDS:
            res = run_strategy(
                data_cache,
                contract_df,
                mode="dynamic",
                dynamic_direction=True,
                buy_the_dip_mode=False,
                spread_lookback=lookback,
                spread_z_threshold=z_threshold,
            )
            results.append(res["metrics"])
            if best_dynamic is None or res["metrics"]["sharpe"] > best_dynamic["metrics"]["sharpe"]:
                best_dynamic = res
            print(
                f"  dynamic lookback={lookback}, z={z_threshold}: "
                f"sharpe={res['metrics']['sharpe']:.4f}"
            )
    best_results["dynamic"] = best_dynamic

    print("\n[3/3] 扫描 buy-the-dip 策略...")
    best_dip = None
    for lookback in DIP_LOOKBACKS:
        for entry_z in DIP_ENTRY_ZS:
            for exit_z in DIP_EXIT_ZS:
                if exit_z < entry_z:
                    continue
                for regime_lookback in REGIME_LOOKBACKS:
                    res = run_strategy(
                        data_cache,
                        contract_df,
                        mode="buy_the_dip",
                        dynamic_direction=False,
                        buy_the_dip_mode=True,
                        spread_lookback=lookback,
                        dip_entry_z=entry_z,
                        dip_exit_z=exit_z,
                        regime_lookback=regime_lookback,
                    )
                    results.append(res["metrics"])
                    if best_dip is None or res["metrics"]["sharpe"] > best_dip["metrics"]["sharpe"]:
                        best_dip = res
                    print(
                        f"  dip lookback={lookback}, entry={entry_z}, exit={exit_z}, regime={regime_lookback}: "
                        f"sharpe={res['metrics']['sharpe']:.4f}"
                    )
    best_results["buy_the_dip"] = best_dip

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    best_rows = []
    for mode, res in best_results.items():
        row = dict(res["metrics"])
        best_rows.append(row)
        save_best_artifacts(mode, res)

    best_df = pd.DataFrame(best_rows).sort_values("sharpe", ascending=False)
    best_df.to_csv(OUTPUT_DIR / "best_by_mode.csv", index=False)
    (OUTPUT_DIR / "REPORT.md").write_text(build_report(best_df), encoding="utf-8")

    print("\n" + "=" * 70)
    print("最优结果")
    print("=" * 70)
    print(best_df.to_string(index=False))
    print(f"\n[OK] 结果保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
