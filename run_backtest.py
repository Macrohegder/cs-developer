#!/usr/bin/env python3
"""
期货期限结构价差反转策略回测入口
基于 vnpy_alpharesearch 框架
"""

from datetime import datetime
from vnpy.trader.constant import Interval

from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.factor import FactorGenerator
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from spread_return_factor import SpreadReturnFactor
from overreact_strategy import OverreactStrategy


def main():
    # ========== 参数配置 ==========
    
    # 品种池：活跃的商品期货（排除金融期货、近月数据短的品种）
    # 数据长度 > 1000 天的品种
    dominant_symbols = [
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
    
    # 构建 vt_symbols：每个品种需要 88（主力）和 889（次主力）
    vt_symbols = []
    for dominant_symbol in dominant_symbols:
        vt_symbols.append(dominant_symbol)                    # 主力连续
        vt_symbols.append(dominant_symbol.replace("88.", "889."))  # 次主力连续
    
    # 回测时间范围
    start = datetime(2011, 1, 1)
    end = datetime(2026, 4, 30)
    
    # 初始资金
    capital = 10_000_000
    
    # 因子参数
    factor_setting = {
        "start": start,
        "end": end,
        "dominant_symbols": dominant_symbols,
        "lookback": 5  # 5个交易日 ≈ 1周
    }
    
    # 策略参数
    strategy_setting = {
        "holding_period": 5,      # 持仓5个交易日
        "trading_signal": 0.1,   # 每端10%品种
        "factor_name": "spread_return"
    }
    
    # ========== Step 1: 因子计算 ==========
    print("=" * 60)
    print("Step 1: 计算 SpreadReturn 因子")
    print("=" * 60)
    
    dc = DataCenter()
    fg = FactorGenerator(vt_symbols, Interval.DAILY, start, end)
    
    # 加载历史价格数据
    print("加载历史价格数据...")
    price_df = fg.load_data()
    print(f"价格数据加载完成，形状: {price_df.shape}")
    
    # 实例化因子并计算
    print("计算因子...")
    factor_df = fg.generate_factor(SpreadReturnFactor, factor_setting)
    print(f"因子计算完成，形状: {factor_df.shape}")
    
    # 保存因子到数据库
    print("保存因子到 DataCenter...")
    dc.create_reference_table(["factor"])
    for col in factor_df.columns:
        series = factor_df[col].dropna()
        if len(series) > 0:
            dc.save_reference_series("factor", f"spread_return_{col}", series)
    print("因子保存完成")
    
    # ========== Step 2: 策略回测 ==========
    print("\n" + "=" * 60)
    print("Step 2: 策略回测")
    print("=" * 60)
    
    sb = StrategyBacktester(dominant_symbols, Interval.DAILY, start, end, capital)
    
    # 加载回测所需数据
    print("加载回测数据...")
    sb.load_data()
    print("回测数据加载完成")
    
    # 运行回测
    print("运行回测...")
    result_df = sb.run_backtesting(OverreactStrategy, strategy_setting)
    print(f"回测完成，结果形状: {result_df.shape}")
    
    # ========== Step 3: 绩效分析 ==========
    print("\n" + "=" * 60)
    print("Step 3: 绩效分析")
    print("=" * 60)
    
    performance = calculate_portfolio_performance(result_df)
    
    print("\n策略绩效:")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    result_path = "/root/futures_term_structure_strategies/backtest_result.csv"
    result_df.to_csv(result_path)
    print(f"\n回测结果已保存到: {result_path}")
    
    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
