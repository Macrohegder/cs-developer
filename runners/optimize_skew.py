#!/usr/bin/env python3
"""
Skew 因子参数优化入口

搜索空间:
  - lookback: [60, 90, 120, 180, 240] 天

策略参数固定（因子搜索阶段杠杆不影响 Sharpe 排序）:
  - holding_period: 5
  - trading_signal: 0.2
  - leverage: 1.0

使用方法:
  python runners/optimize_skew.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from vnpy_alpharesearch import DataCenter

from runners.factor_optimizer import FactorParameterOptimizer
from factors.skew_factor import SkewFactor


# ============ 配置 ============
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

# 因子参数网格
PARAM_GRID = {
    "lookback": [60, 90, 120, 180, 240],
}

# 策略参数（因子搜索阶段固定）
STRATEGY_PARAMS = {
    "holding_period": 5,
    "trading_signal": 0.2,
    "leverage": 1.0,
}


def save_factor_to_datacenter(factor_df, lookback):
    """将因子保存到 DataCenter"""
    print(f"\n[Save] 将最优因子 (lookback={lookback}) 保存到 DataCenter...")
    dc = DataCenter()
    dc.save_factor_df(
        df=factor_df,
        name="skew",
        interval="d",
        parameter=f"lookback{lookback}",
        author="cross_sectional"
    )
    print("  保存完成")


def main():
    print("=" * 70)
    print("Skew 因子参数优化")
    print("=" * 70)
    print(f"参数网格: {PARAM_GRID}")
    print(f"策略参数: {STRATEGY_PARAMS}")
    print(f"品种数: {len(DOMINANT_SYMBOLS)}")
    print("=" * 70)

    optimizer = FactorParameterOptimizer(
        dominant_symbols=DOMINANT_SYMBOLS,
        start=START,
        end=END,
        capital=10_000_000,
        commission=0.0001,
    )

    results_df, factor_dict = optimizer.optimize(
        factor_class=SkewFactor,
        param_grid=PARAM_GRID,
        strategy_params=STRATEGY_PARAMS,
    )

    # 汇总输出
    print("\n" + "=" * 70)
    print("优化完成！按 Sharpe Ratio 排序的结果:")
    print("=" * 70)

    for _, row in results_df.iterrows():
        print(
            f"  lookback={int(row['lookback'])} | "
            f"total={row['total_return']:.2%}, "
            f"ann={row['annual_return']:.2%}, "
            f"sharpe={row['sharpe_ratio']:.2f}, "
            f"maxdd={row['max_ddpercent']:.2%}, "
            f"win_rate={row['win_rate']:.2%}"
        )

    # 保存结果
    results_path = Path(__file__).parent / "optimize_skew_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {results_path}")

    # 最优参数
    best = results_df.iloc[0]
    best_lookback = int(best['lookback'])
    print("\n" + "=" * 70)
    print("最优参数:")
    print(f"  lookback = {best_lookback}")
    print(f"  Sharpe   = {best['sharpe_ratio']:.2f}")
    print(f"  年化收益  = {best['annual_return']:.2%}")
    print(f"  最大回撤  = {best['max_ddpercent']:.2%}")
    print("=" * 70)

    # 保存最优因子到 DataCenter
    best_param_key = f"lookback{best_lookback}"
    best_factor_df = factor_dict[best_param_key]
    save_factor_to_datacenter(best_factor_df, best_lookback)


if __name__ == "__main__":
    main()
