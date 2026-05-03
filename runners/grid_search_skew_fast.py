#!/usr/bin/env python3
"""
Skew因子参数网格搜索（快速版）
- 只加载一次K线数据，复用进行多组参数回测
- 参数组合: holding_period × trading_signal × leverage
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from itertools import product

import pandas as pd

from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.factor import FactorGenerator
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from factors.skew_factor import SkewFactor
from strategies.cross_sectional_strategy import CrossSectionalStrategy


# ============ 配置 ============
CONFIG = {
    "dominant_symbols": [
        'A88.DCE', 'AG88.SHFE', 'AL88.SHFE', 'AP88.CZCE', 'AU88.SHFE',
        'B88.DCE', 'BU88.SHFE', 'C88.DCE', 'CF88.CZCE', 'CJ88.CZCE',
        'CS88.DCE', 'CU88.SHFE', 'CY88.CZCE', 'EB88.DCE', 'EG88.DCE',
        'FG88.CZCE', 'FU88.SHFE', 'HC88.SHFE', 'I88.DCE', 'J88.DCE',
        'JD88.DCE', 'JM88.DCE', 'L88.DCE', 'LH88.DCE', 'LU88.INE',
        'M88.DCE', 'MA88.CZCE', 'NI88.SHFE', 'OI88.CZCE', 'P88.DCE',
        'PB88.SHFE', 'PF88.CZCE', 'PG88.DCE', 'PK88.CZCE', 'PP88.DCE',
        'RB88.SHFE', 'RM88.CZCE', 'RU88.SHFE', 'SA88.CZCE', 'SC88.INE',
        'SF88.CZCE', 'SI88.GFEX', 'SM88.CZCE', 'SN88.SHFE', 'SP88.SHFE',
        'SR88.CZCE', 'SS88.SHFE', 'UR88.CZCE', 'V88.DCE', 'Y88.DCE',
        'ZN88.SHFE'
    ],
    "start": datetime(2015, 1, 1),
    "end": datetime(2025, 12, 31),
    "capital": 10_000_000,
    "commission": 0.0001,
    "risk_free": 0,
    "plot_chart": False,
}

PARAM_GRID = {
    "holding_period": [3, 5, 10],
    "trading_signal": [0.1, 0.2, 0.3],
    "leverage": [1.0, 2.0, 3.0],
}


def check_factor_exists(config: dict) -> bool:
    """检查 Skew 因子是否已存在于数据库中"""
    try:
        dc = DataCenter()
        df = dc.load_factor_df(
            vt_symbols=config["dominant_symbols"][:2],
            name="skew",
            interval="d",
            parameter="lookback180",
            author="cross_sectional",
            start=config["start"],
            end=config["end"]
        )
        return len(df) > 0
    except Exception:
        return False


def prepare_factor(config: dict):
    """生成因子（只执行一次）"""
    print("=" * 70)
    print("Step 1: 准备 Skew 因子")
    print("=" * 70)

    if check_factor_exists(config):
        print("因子已存在于数据库中，跳过计算。")
        return

    dominant_symbols = config["dominant_symbols"]
    start = config["start"]
    end = config["end"]

    factor_setting = {
        "start": start,
        "end": end,
        "dominant_symbols": dominant_symbols,
        "lookback": 180,
    }

    fg = FactorGenerator(dominant_symbols, Interval.DAILY, start, end)
    print("加载历史价格数据...")
    price_df = fg.load_data()
    print(f"  价格数据加载完成，形状: {price_df.shape}")

    print("计算 Skew 因子...")
    factor_df = fg.generate_factor(SkewFactor, factor_setting)
    print(f"  因子计算完成，形状: {factor_df.shape}")

    print("保存因子到 DataCenter...")
    dc = DataCenter()
    dc.save_factor_df(
        df=factor_df,
        name="skew",
        interval="d",
        parameter="lookback180",
        author="cross_sectional"
    )
    print("  因子保存完成")


def load_data_once(bt: StrategyBacktester):
    """只加载一次数据"""
    print("\n" + "=" * 70)
    print("[GridSearch] 一次性加载历史行情数据...")
    print("=" * 70)
    bt.load_data()
    print("[GridSearch] 历史数据加载完成！")
    return bt


def run_single_backtest(bt: StrategyBacktester, holding_period, trading_signal, leverage):
    """使用已加载的数据运行单组参数回测"""
    setting = {
        "holding_period": holding_period,
        "trading_signal": trading_signal,
        "leverage": leverage,
        "long_low": True,
        "factor_name": "skew",
        "factor_parameter": "lookback180",
        "factor_author": "cross_sectional",
        "aggregation": "sum",
    }

    # 复用已加载的数据，直接调用run_backtesting
    target_df = bt.run_backtesting(CrossSectionalStrategy, setting)

    if target_df.empty:
        return None

    result = calculate_portfolio_performance(
        target_df,
        Interval.DAILY,
        commission=CONFIG["commission"],
        capital=CONFIG["capital"],
        risk_free=CONFIG["risk_free"],
        plot_chart=CONFIG["plot_chart"]
    )

    stats = result.get("statistics", {})

    def _f(key, default=0):
        v = stats.get(key, default)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    return {
        "holding_period": holding_period,
        "trading_signal": trading_signal,
        "leverage": leverage,
        "total_return": _f("total_return"),
        "annual_return": _f("annual_return"),
        "sharpe_ratio": _f("sharpe_ratio"),
        "max_drawdown": _f("max_drawdown"),
        "calmar_ratio": _f("calmar_ratio"),
        "win_rate": _f("win_rate"),
    }


def main():
    dominant_symbols = CONFIG["dominant_symbols"]

    # Step 1: 准备因子（只一次）
    prepare_factor(CONFIG)

    # Step 2: 创建Backtester并加载数据（只一次）
    bt = StrategyBacktester(
        vt_symbols=dominant_symbols,
        interval=Interval.DAILY,
        start=CONFIG["start"],
        end=CONFIG["end"],
        capital=CONFIG["capital"],
    )
    bt = load_data_once(bt)

    # Step 3: 参数网格搜索
    hp_list = PARAM_GRID["holding_period"]
    ts_list = PARAM_GRID["trading_signal"]
    lev_list = PARAM_GRID["leverage"]
    total = len(hp_list) * len(ts_list) * len(lev_list)

    print("\n" + "=" * 70)
    print(f"[GridSearch] 开始网格搜索: 共 {total} 组参数")
    print("=" * 70)

    results = []
    count = 0
    for holding_period, trading_signal, leverage in product(hp_list, ts_list, lev_list):
        count += 1
        print(f"\n[{count}/{total}] hp={holding_period}, ts={trading_signal}, lev={leverage}")

        result = run_single_backtest(bt, holding_period, trading_signal, leverage)
        if result:
            results.append(result)
            print(
                f"    total={result['total_return']:.2%}, "
                f"ann={result['annual_return']:.2%}, "
                f"sharpe={result['sharpe_ratio']:.2f}, "
                f"maxdd={result['max_drawdown']:.2%}"
            )
        else:
            print("    (无有效结果)")

    # 汇总输出
    print("\n" + "=" * 70)
    print("[GridSearch] 网格搜索完成！")
    print("=" * 70)

    if not results:
        print("无有效结果")
        return

    results_df = pd.DataFrame(results)
    results_df.sort_values("sharpe_ratio", ascending=False, inplace=True)

    print("\n按 Sharpe Ratio 排序的前10组参数:")
    for _, row in results_df.head(10).iterrows():
        print(
            f"  hp={int(row['holding_period'])}, ts={row['trading_signal']}, lev={row['leverage']} | "
            f"total={row['total_return']:.2%}, ann={row['annual_return']:.2%}, "
            f"sharpe={row['sharpe_ratio']:.2f}, maxdd={row['max_drawdown']:.2%}"
        )

    # 保存结果
    results_path = Path(__file__).parent / "grid_search_skew_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {results_path}")


if __name__ == "__main__":
    main()
