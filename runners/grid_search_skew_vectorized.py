#!/usr/bin/env python3
"""
Skew因子参数网格搜索 —— 向量化快速版
完全独立计算，不依赖 StrategyBacktester 框架
直接用 pandas 向量化操作，27组参数几秒完成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from math import floor, sqrt
from itertools import product

import numpy as np
import pandas as pd

from vnpy.trader.constant import Interval
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df


# ==================== 配置 ====================
DOMINANT_SYMBOLS = [
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
]

START = datetime(2015, 1, 1)
END = datetime(2025, 12, 31)
CAPITAL = 10_000_000
COMMISSION = 0.0001
RISK_FREE = 0

PARAM_GRID = {
    "holding_period": [3, 5, 10],
    "trading_signal": [0.1, 0.2, 0.3],
    "leverage": [1.0, 2.0, 3.0],
}


def load_factor_data():
    """加载 Skew 因子数据"""
    print("加载 Skew 因子数据...")
    dc = DataCenter()
    factor_df = dc.load_factor_df(
        vt_symbols=DOMINANT_SYMBOLS,
        name="skew",
        interval="d",
        parameter="lookback180",
        author="cross_sectional",
        start=START,
        end=END,
    )
    print(f"  因子数据: {factor_df.shape}")
    return factor_df


def load_price_data():
    """加载 88 连续合约的开盘/收盘价格"""
    print("加载 88 连续合约价格数据...")
    close_list = []
    open_list = []
    for sym in DOMINANT_SYMBOLS:
        df = load_bar_df(sym, Interval.DAILY, START, END)
        close_list.append(df["close_price"].rename(sym))
        open_list.append(df["open_price"].rename(sym))
    close_df = pd.concat(close_list, axis=1)
    open_df = pd.concat(open_list, axis=1)
    print(f"  价格数据: {close_df.shape}")
    return close_df, open_df


def load_multipliers():
    """从合约信息表获取每个品种的乘数"""
    print("加载合约乘数...")
    dc = DataCenter()
    contract_df = dc.load_contract_df()

    # 先读取主力映射，取第一天的主力合约来确定品种代码
    mapping_keys = [f"{s}@1" for s in DOMINANT_SYMBOLS]
    mapping_df = dc.load_reference_df(
        table="vnpy_dominant_contract",
        keys=mapping_keys,
        start=START,
        end=END,
    )

    multipliers = {}
    for ds in DOMINANT_SYMBOLS:
        key = f"{ds}@1"
        # 取第一个非空映射值
        first_contract = mapping_df[key].dropna().iloc[0]
        symbol = first_contract.split(".")[0]
        # symbol 如 "I2501"，contract_df 索引也是 "I2501"
        multiplier = contract_df.loc[symbol, "multiplier"]
        multipliers[ds] = multiplier

    multiplier_series = pd.Series(multipliers)
    print(f"  乘数加载完成，示例: {dict(multiplier_series.head(3))}")
    return multiplier_series


def calculate_stats(pnl_series, capital):
    """计算绩效指标（与 calculate_portfolio_performance 一致）"""
    balance = pnl_series.cumsum() + capital
    if len(balance) == 0 or balance.isna().all():
        return {}

    highlevel = balance.cummax()
    drawdown = balance - highlevel
    ret = balance.pct_change().fillna(0)
    ddpercent = drawdown / highlevel

    start_balance = balance.iloc[0]
    end_balance = balance.iloc[-1]
    total_days = len(balance)

    total_return = end_balance / start_balance - 1
    annual_return = total_return / total_days * 240
    daily_return = ret.mean()
    return_std = ret.std()
    sharpe_ratio = (daily_return - RISK_FREE / 240) / return_std * sqrt(240) if return_std > 0 else 0
    max_drawdown = drawdown.min()
    max_ddpercent = ddpercent.min()
    calmar_ratio = -pnl_series.sum() / max_drawdown if max_drawdown != 0 else 0
    win_rate = (pnl_series > 0).sum() / (pnl_series != 0).sum() if (pnl_series != 0).any() else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_ddpercent": max_ddpercent,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
    }


def backtest_vectorized(factor_df, close_df, open_df, multiplier_series,
                        holding_period, trading_signal, leverage):
    """
    向量化回测核心
    完全复现 CrossSectionalStrategy + calculate_symbol_pnl 逻辑
    """
    # 1. 计算每日原始信号（排名做多低Skew，做空高Skew）
    n = len(DOMINANT_SYMBOLS)
    choose = max(1, int(floor(trading_signal * n)))

    # 对每个交易日，根据因子值排序，生成 ±1 信号
    # 因子值越小（负偏度越大）→ 排名越前 → 做多
    rank_df = factor_df.rank(axis=1, method="first", ascending=True)
    signal_raw = pd.DataFrame(0, index=factor_df.index, columns=factor_df.columns)
    signal_raw[rank_df <= choose] = 1        # 多头
    signal_raw[rank_df > (n - choose)] = -1  # 空头

    # 2. 滚动聚合（holding_period 天求和）
    # 注意：这里用 rolling(holding_period).sum()，与策略的 rolling holding 一致
    # 但要处理前 holding_period-1 天的数据不够的问题
    if holding_period > 1:
        signal_rolling = signal_raw.rolling(window=holding_period, min_periods=1).sum()
    else:
        signal_rolling = signal_raw.copy()

    # 3. 计算每品种目标手数
    daily_capital = CAPITAL * leverage / holding_period
    target_capital_per_symbol = daily_capital / (2 * choose)

    # position = signal_rolling * target_capital / (close * multiplier)
    # 注意：用当日收盘价估算手数（策略是收盘后计算，次日开盘执行）
    position = signal_rolling.multiply(
        target_capital_per_symbol / (close_df.multiply(multiplier_series, axis=1))
    )
    position = position.fillna(0)

    # 4. 收益计算（复现 calculate_symbol_pnl 逻辑）
    # target[T] = position[T] （T日收盘后生成的目标）
    # end_pos[T] = target[T-1] = position[T-1]  （T日收盘仓位）
    # start_pos[T] = end_pos[T-1] = position[T-2]  （T日开盘仓位）
    end_pos = position.shift(1).fillna(0)
    start_pos = end_pos.shift(1).fillna(0)
    pos_change = end_pos - start_pos

    # trading_pnl = pos_change * (close - open) * multiplier
    trading_pnl = pos_change.multiply(
        (close_df - open_df).multiply(multiplier_series, axis=1)
    )

    # holding_pnl = start_pos * (close - pre_close) * multiplier
    holding_pnl = start_pos.multiply(
        (close_df - close_df.shift(1)).multiply(multiplier_series, axis=1)
    )

    total_pnl = trading_pnl + holding_pnl

    # 交易成本
    turnover = pos_change.abs().multiply(
        open_df.multiply(multiplier_series, axis=1)
    )
    trading_cost = turnover * COMMISSION
    net_pnl = total_pnl - trading_cost

    # 5. 组合汇总
    portfolio_pnl = net_pnl.sum(axis=1)

    # 去掉前 warmup 期（因子需要180天warmup + 2天position延迟）
    valid_pnl = portfolio_pnl.iloc[182:]

    stats = calculate_stats(valid_pnl, CAPITAL)
    return stats


def main():
    print("=" * 70)
    print("Skew 因子网格搜索 —— 向量化快速版")
    print("=" * 70)

    # 加载数据（只一次）
    factor_df = load_factor_data()
    close_df, open_df = load_price_data()
    multiplier_series = load_multipliers()

    # 对齐索引
    common_index = factor_df.index.intersection(close_df.index)
    factor_df = factor_df.loc[common_index]
    close_df = close_df.loc[common_index]
    open_df = open_df.loc[common_index]

    # 确保乘数顺序与列一致
    multiplier_series = multiplier_series.reindex(factor_df.columns)

    print(f"\n对齐后数据形状: {factor_df.shape}")
    print(f"交易日范围: {common_index[0]} ~ {common_index[-1]}")

    # 参数网格
    hp_list = PARAM_GRID["holding_period"]
    ts_list = PARAM_GRID["trading_signal"]
    lev_list = PARAM_GRID["leverage"]
    total = len(hp_list) * len(ts_list) * len(lev_list)

    print("\n" + "=" * 70)
    print(f"开始网格搜索: 共 {total} 组参数")
    print("=" * 70)

    results = []
    count = 0
    for hp, ts, lev in product(hp_list, ts_list, lev_list):
        count += 1
        stats = backtest_vectorized(
            factor_df, close_df, open_df, multiplier_series,
            holding_period=hp, trading_signal=ts, leverage=lev
        )
        result = {
            "holding_period": hp,
            "trading_signal": ts,
            "leverage": lev,
            **stats,
        }
        results.append(result)
        print(
            f"[{count:2d}/{total}] hp={hp}, ts={ts}, lev={lev} | "
            f"total={stats.get('total_return', 0):.2%}, "
            f"ann={stats.get('annual_return', 0):.2%}, "
            f"sharpe={stats.get('sharpe_ratio', 0):.2f}, "
            f"maxdd={stats.get('max_ddpercent', 0):.2%}"
        )

    # 汇总
    print("\n" + "=" * 70)
    print("网格搜索完成！")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df.sort_values("sharpe_ratio", ascending=False, inplace=True)

    print("\n按 Sharpe Ratio 排序的前10组参数:")
    for _, row in results_df.head(10).iterrows():
        print(
            f"  hp={int(row['holding_period'])}, ts={row['trading_signal']}, lev={row['leverage']} | "
            f"total={row['total_return']:.2%}, ann={row['annual_return']:.2%}, "
            f"sharpe={row['sharpe_ratio']:.2f}, maxdd={row['max_drawdown']:.2%}"
        )

    # 保存
    results_path = Path(__file__).parent / "grid_search_skew_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {results_path}")


if __name__ == "__main__":
    main()
