#!/usr/bin/env python3
"""
期货期限结构策略回测 — 价差涨跌幅因子（动量反转）

与 spread_zscore 的对比：
  - spread_zscore: 均值回归逻辑，Z-Score 衡量价差偏离均值程度
  - spread_return: 动量反转逻辑，过去10天价差涨跌幅衡量价差动量

执行流程（同样三步）：
  Step 1: FactorGenerator 计算因子 → DataCenter 保存
  Step 2: StrategyBacktester 执行回测 → 生成 target_df
  Step 3: calculate_portfolio_performance 计算绩效 → 输出指标与图表
"""

from datetime import datetime

from vnpy.trader.constant import Interval

from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.factor import FactorGenerator
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from spread_return_factor import SpreadReturnFactor
from term_structure_strategy import TermStructureStrategy


# =============================================================================
# 参数配置区（所有可调参数集中在此处）
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
    "start": datetime(2011, 1, 1),
    "end": datetime(2026, 4, 30),

    # 初始资金
    "capital": 10_000_000,

    # 因子参数
    "factor_setting": {
        "lookback": 10          # 价差历史回看周期（交易日），过去10天价差涨跌幅
    },

    # 策略参数
    "strategy_setting": {
        "holding_period": 10,          # 持仓天数
        "trading_signal": 0.1,         # 每端选股比例
        "factor_name": "spread_return",      # 使用价差涨跌幅因子
        "factor_parameter": "lookback10",    # 因子参数
        "factor_author": "futures_term_structure"
    },

    # 绩效参数
    "commission": 0.0001,        # 手续费率（万 1，双边）
    "risk_free": 0,              # 无风险利率
    "plot_chart": False,         # 是否绘制图表（关闭plotly，后续用matplotlib保存png）

    # 执行优化
    "skip_factor_if_exists": True,  # 若因子已存在于数据库则跳过 Step 1
}


def build_vt_symbols(dominant_symbols: list) -> list:
    """
    构建 FactorGenerator 所需的完整 vt_symbols 列表
    每个品种需要 88（主力连续）和 88A2（次主力连续）
    """
    vt_symbols = []
    for ds in dominant_symbols:
        vt_symbols.append(ds)
        vt_symbols.append(ds.replace("88.", "88A2."))
    return vt_symbols


def check_factor_exists(config: dict) -> bool:
    """检查因子是否已存在于数据库中"""
    try:
        dc = DataCenter()
        df = dc.load_factor_df(
            vt_symbols=config["dominant_symbols"][:2],  # 抽查前两个品种
            name="spread_return",
            interval="d",
            parameter=f"lookback{config['factor_setting']['lookback']}",
            author="futures_term_structure",
            start=config["start"],
            end=config["end"]
        )
        return len(df) > 0
    except Exception:
        return False


def step1_factor_calculation(config: dict) -> None:
    """
    Step 1: 因子计算
    使用 FactorGenerator 加载行情、计算因子、保存到 DataCenter
    """
    print("=" * 70)
    print("Step 1: 计算 SpreadReturn 因子（过去10天价差涨跌幅）")
    print("=" * 70)

    # 检查是否需要跳过
    if config.get("skip_factor_if_exists", False) and check_factor_exists(config):
        print("因子已存在于数据库中，跳过计算。")
        return

    dominant_symbols = config["dominant_symbols"]
    start = config["start"]
    end = config["end"]
    vt_symbols = build_vt_symbols(dominant_symbols)

    # 组装因子 setting（需要包含 dominant_symbols 以及起止时间）
    factor_setting = {
        "start": start,
        "end": end,
        "dominant_symbols": dominant_symbols,
        **config["factor_setting"]
    }

    # 创建因子生成器
    fg = FactorGenerator(vt_symbols, Interval.DAILY, start, end)

    # 加载历史价格数据
    print("加载历史价格数据...")
    price_df = fg.load_data()
    print(f"  价格数据加载完成，形状: {price_df.shape}")

    # 计算因子
    print("计算因子...")
    factor_df = fg.generate_factor(SpreadReturnFactor, factor_setting)
    print(f"  因子计算完成，形状: {factor_df.shape}")

    # 保存因子到 DataCenter（标准接口）
    print("保存因子到 DataCenter...")
    dc = DataCenter()
    dc.save_factor_df(
        df=factor_df,
        name="spread_return",
        interval="d",
        parameter=f"lookback{config['factor_setting']['lookback']}",
        author="futures_term_structure"
    )
    print("  因子保存完成")


def step2_strategy_backtest(config: dict):
    """
    Step 2: 策略回测
    使用 StrategyBacktester 加载数据、运行回测、生成 target_df
    """
    print("\n" + "=" * 70)
    print("Step 2: 策略回测（TermStructureStrategy + SpreadReturn）")
    print("=" * 70)

    dominant_symbols = config["dominant_symbols"]
    start = config["start"]
    end = config["end"]
    capital = config["capital"]

    # 创建策略回测器（标准接口：只传入 dominant_symbols）
    backtester = StrategyBacktester(
        dominant_symbols,
        Interval.DAILY,
        start,
        end,
        capital
    )

    # 加载历史行情数据
    print("加载历史行情数据...")
    backtester.load_data()

    # 运行回测
    print("运行回测...")
    target_df = backtester.run_backtesting(
        TermStructureStrategy,
        config["strategy_setting"]
    )
    print(f"  回测完成，目标仓位形状: {target_df.shape}")

    # 简单检查
    non_zero_days = (target_df != 0).any(axis=1).sum()
    print(f"  有仓位的交易日: {non_zero_days}/{len(target_df)}")
    print(f"  涉及的具体合约数: {len(target_df.columns)}")

    return target_df


def step3_performance_analysis(target_df, config: dict) -> dict:
    """
    Step 3: 绩效分析
    使用框架内置的 calculate_portfolio_performance 计算完整绩效
    """
    print("\n" + "=" * 70)
    print("Step 3: 绩效分析")
    print("=" * 70)

    # 使用框架标准绩效计算（不自定义）
    result = calculate_portfolio_performance(
        target_df,
        Interval.DAILY,
        commission=config["commission"],
        capital=config["capital"],
        risk_free=config["risk_free"],
        plot_chart=config["plot_chart"]
    )

    # 打印核心统计指标
    stats = result["statistics"]
    print("\n【策略绩效指标】")
    for key, value in stats.items():
        print(f"  {key:<20s}: {value}")

    return result


def save_results(target_df, result: dict, config: dict) -> None:
    """保存回测结果到本地文件（使用 spread_return 前缀以区分）"""
    import pandas as pd

    prefix = "/root/futures_term_structure_strategies/result_spread_return"

    # 保存目标仓位
    target_path = f"{prefix}_target.csv"
    target_df.to_csv(target_path)
    print(f"\n  目标仓位已保存: {target_path}")

    # 保存累计盈亏曲线
    if "overall" in result:
        pnl_path = f"{prefix}_pnl.csv"
        result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(pnl_path)
        print(f"  净值曲线已保存: {pnl_path}")

    # 保存分品种盈亏
    if "product" in result:
        product_path = f"{prefix}_product.csv"
        result["product"].to_csv(product_path)
        print(f"  分品种盈亏已保存: {product_path}")


def main():
    """主执行函数：严格按 Step 1 → Step 2 → Step 3 执行"""
    config = CONFIG

    # Step 1: 因子计算
    step1_factor_calculation(config)

    # Step 2: 策略回测
    target_df = step2_strategy_backtest(config)

    # Step 3: 绩效分析
    result = step3_performance_analysis(target_df, config)

    # 保存结果
    save_results(target_df, result, config)

    print("\n" + "=" * 70)
    print("SpreadReturn 因子回测全部流程执行完毕！")
    print("=" * 70)


if __name__ == "__main__":
    main()
