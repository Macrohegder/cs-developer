#!/usr/bin/env python3
"""
高波动阈值对比扫描

测试 vol_threshold ∈ [0.15, 0.20, 0.25, 0.30] 对 IF / IC / IM 的影响，
其他参数固定为最优值：
- vol_window=20
- only_quarterly=True
- roll_before_days=5
- stop_loss_pct=0
- take_profit_pct=0

输出：
- 各阈值下各品种绩效表格
- NAV 对比图
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

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

BASE_PARAMS = {
    "vol_window": 20,
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "only_quarterly": True,
    "roll_before_days": 5,
}

VOL_THRESHOLDS = [0.15, 0.20, 0.25, 0.30]
PRODUCTS = ["IF", "IC", "IM"]


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


def run_backtest(product: str, vol_threshold: float, data_cache: Dict, contract_df: pd.DataFrame) -> Dict:
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
        "vol_threshold": vol_threshold,
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

    return {
        'product': product,
        'vol_threshold': vol_threshold,
        'target_df': target_df,
        'overall': overall,
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


def plot_comparison(results_df: pd.DataFrame, overall_dict: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('sharpe', 'Sharpe Ratio'),
        ('total_return', 'Total Return'),
        ('max_dd', 'Max Drawdown'),
        ('calmar', 'Calmar Ratio'),
    ]

    for ax, (col, title) in zip(axes.flat, metrics):
        pivot = results_df.pivot(index='vol_threshold', columns='product', values=col)
        pivot.index = [f"{int(x*100)}%" for x in pivot.index]
        pivot.plot(kind='bar', ax=ax, rot=0)
        ax.set_title(title)
        ax.set_xlabel('Vol Threshold')
        ax.legend(title='Product')
        ax.grid(True, alpha=0.3)
        if col in ['total_return', 'max_dd']:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    plt.tight_layout()
    plt.savefig(output_dir / 'vol_threshold_comparison_metrics.png', dpi=150)
    plt.close()

    # NAV 对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, product in zip(axes, PRODUCTS):
        for vt in VOL_THRESHOLDS:
            key = (product, vt)
            if key not in overall_dict:
                continue
            overall = overall_dict[key]
            nav = overall["balance"] / overall["balance"].iloc[0]
            ax.plot(nav.index, nav, label=f"{int(vt*100)}%")
        ax.set_title(f'{product} NAV by Vol Threshold')
        ax.set_xlabel('Date')
        ax.set_ylabel('NAV')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'vol_threshold_comparison_nav.png', dpi=150)
    plt.close()


def main():
    print("="*70)
    print("高波动阈值对比扫描（IF / IC / IM）")
    print(f"基础参数: {BASE_PARAMS}")
    print(f"波动阈值: {[int(x*100) for x in VOL_THRESHOLDS]}%")
    print("="*70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] 加载合约信息...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    print("\n[2/2] 加载品种数据...")
    data_cache = {
        'IF': prepare_product_data('IF', contract_df),
        'IC': prepare_product_data('IC', contract_df),
        'IM': prepare_product_data('IM', contract_df),
    }

    results = []
    overall_dict = {}

    for product in PRODUCTS:
        for vt in VOL_THRESHOLDS:
            print(f"\n>>> 运行 {product} vol_threshold={vt:.2f}...")
            res = run_backtest(product, vt, data_cache, contract_df)
            results.append({
                'product': res['product'],
                'vol_threshold': res['vol_threshold'],
                'total_pnl': res['total_pnl'],
                'total_return': res['total_return'],
                'annual_return': res['annual_return'],
                'max_dd': res['max_dd'],
                'sharpe': res['sharpe'],
                'calmar': res['calmar'],
                'win_rate': res['win_rate'],
                'profit_days': res['profit_days'],
                'loss_days': res['loss_days'],
                'days': res['days'],
            })
            overall_dict[(product, vt)] = res['overall']

            # 保存每个组合的 pnl 曲线
            prefix = OUTPUT_DIR / f"vt{int(vt*100)}_{product.lower()}"
            res['overall'][["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(f"{prefix}_pnl.csv")
            res['target_df'].to_csv(f"{prefix}_target.csv")

            print(f"  pnl={res['total_pnl']:,.0f}, return={res['total_return']:.2%}, sharpe={res['sharpe']:.3f}, max_dd={res['max_dd']:.2%}")

    results_df = pd.DataFrame(results)
    results_df['vol_threshold_pct'] = (results_df['vol_threshold'] * 100).astype(int).astype(str) + '%'

    # 保存结果表
    results_df.to_csv(OUTPUT_DIR / 'vol_threshold_comparison.csv', index=False)

    # 绘制对比图
    plot_comparison(results_df, overall_dict, OUTPUT_DIR)

    print("\n" + "="*70)
    print("对比结果汇总")
    print("="*70)
    display_cols = ['product', 'vol_threshold_pct', 'total_pnl', 'total_return', 'max_dd', 'sharpe', 'calmar', 'win_rate']
    print(results_df[display_cols].to_string(index=False))

    print(f"\n[OK] 结果保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
