#!/usr/bin/env python3
"""
Skew 因子横截面多空策略回测 — 标准化执行脚本

严格遵循三步流程：
  Step 1: FactorGenerator 计算 Skew 因子 → DataCenter 保存
  Step 2: StrategyBacktester 执行回测 → 生成 target_df
  Step 3: calculate_portfolio_performance 计算绩效 → 输出指标

策略逻辑：
- 因子：SkewFactor（过去180天日收益率的偏度）
- 策略：CrossSectionalStrategy（横截面排序，做多负偏度，做空正偏度）
- 持仓：滚动 holding_period 天，每日新开一份仓位
- 交易标的：主力合约（通过 DominantManager 映射到具体合约）

回测区间：2015-01-01 起
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from vnpy.trader.constant import Interval

from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.factor import FactorGenerator
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from factors.skew_factor import SkewFactor
from strategies.cross_sectional_strategy import CrossSectionalStrategy


# =============================================================================
# 参数配置区
# =============================================================================
CONFIG = {
    # 品种池：活跃的商品期货（88 主力连续合约）
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

    # 回测时间范围
    "start": datetime(2015, 1, 1),
    "end": datetime(2025, 12, 31),

    # 初始资金
    "capital": 10_000_000,

    # 因子参数
    "factor_setting": {
        "lookback": 180          # Skew 回望期：180个交易日
    },

    # 策略参数
    "strategy_setting": {
        "holding_period": 5,         # 持仓天数
        "trading_signal": 0.2,       # 每端选股比例（20%）
        "factor_name": "skew",       # 因子名称
        "factor_parameter": "lookback180",  # 因子参数标识
        "factor_author": "cross_sectional",  # 因子作者
        "long_low": True,            # 做多负偏度（低Skew），做空正偏度（高Skew）
        "aggregation": "sum",        # 分仓滚动
        "leverage": 2.0,             # 2倍名义价值杠杆
    },

    # 绩效参数
    "commission": 0.0001,        # 手续费率（万1，双边）
    "risk_free": 0,              # 无风险利率
    "plot_chart": False,         # 是否绘制图表

    # 执行优化
    "skip_factor_if_exists": True,  # 若因子已存在于数据库则跳过 Step 1
}


def build_vt_symbols(dominant_symbols: list) -> list:
    """
    构建 FactorGenerator 所需的完整 vt_symbols 列表
    Skew 因子只需要 88（主力连续）的价格数据
    """
    return list(dominant_symbols)


def check_factor_exists(config: dict) -> bool:
    """检查 Skew 因子是否已存在于数据库中"""
    try:
        dc = DataCenter()
        df = dc.load_factor_df(
            vt_symbols=config["dominant_symbols"][:2],  # 抽查前两个品种
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


def step1_factor_calculation(config: dict) -> None:
    """Step 1: 因子计算"""
    print("=" * 70)
    print("Step 1: 计算 Skew 因子")
    print("=" * 70)

    if config.get("skip_factor_if_exists", False) and check_factor_exists(config):
        print("因子已存在于数据库中，跳过计算。")
        return

    dominant_symbols = config["dominant_symbols"]
    start = config["start"]
    end = config["end"]
    vt_symbols = build_vt_symbols(dominant_symbols)

    factor_setting = {
        "start": start,
        "end": end,
        "dominant_symbols": dominant_symbols,
        **config["factor_setting"]
    }

    fg = FactorGenerator(vt_symbols, Interval.DAILY, start, end)

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


def step2_strategy_backtest(config: dict):
    """Step 2: 策略回测"""
    print("\n" + "=" * 70)
    print("Step 2: 策略回测（CrossSectionalStrategy + Skew）")
    print("=" * 70)

    dominant_symbols = config["dominant_symbols"]
    start = config["start"]
    end = config["end"]
    capital = config["capital"]

    backtester = StrategyBacktester(
        dominant_symbols,
        Interval.DAILY,
        start,
        end,
        capital
    )

    print("加载历史行情数据...")
    backtester.load_data()

    print("运行回测...")
    target_df = backtester.run_backtesting(
        CrossSectionalStrategy,
        config["strategy_setting"]
    )
    print(f"  回测完成，目标仓位形状: {target_df.shape}")

    non_zero_days = (target_df != 0).any(axis=1).sum()
    print(f"  有仓位的交易日: {non_zero_days}/{len(target_df)}")
    print(f"  涉及的具体合约数: {len(target_df.columns)}")

    return target_df


def step3_performance_analysis(target_df, config: dict) -> dict:
    """Step 3: 绩效分析"""
    print("\n" + "=" * 70)
    print("Step 3: 绩效分析")
    print("=" * 70)

    result = calculate_portfolio_performance(
        target_df,
        Interval.DAILY,
        commission=config["commission"],
        capital=config["capital"],
        risk_free=config["risk_free"],
        plot_chart=config["plot_chart"]
    )

    stats = result["statistics"]
    print("\n【策略绩效指标】")
    for key, value in stats.items():
        print(f"  {key:<20s}: {value}")

    return result


def save_results(target_df, result: dict, config: dict) -> None:
    """保存回测结果到本地文件"""
    import pandas as pd

    prefix = "/root/cs_developer/result_skew"

    target_path = f"{prefix}_target.csv"
    target_df.to_csv(target_path)
    print(f"\n  目标仓位已保存: {target_path}")

    if "overall" in result:
        pnl_path = f"{prefix}_pnl.csv"
        result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(pnl_path)
        print(f"  净值曲线已保存: {pnl_path}")

    if "product" in result:
        product_path = f"{prefix}_product.csv"
        result["product"].to_csv(product_path)
        print(f"  分品种盈亏已保存: {product_path}")


def main():
    """主执行函数"""
    config = CONFIG

    step1_factor_calculation(config)
    target_df = step2_strategy_backtest(config)
    result = step3_performance_analysis(target_df, config)
    save_results(target_df, result, config)

    print("\n" + "=" * 70)
    print("Skew 因子回测执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()
