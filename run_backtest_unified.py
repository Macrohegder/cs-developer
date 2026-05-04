#!/usr/bin/env python3
"""
统一回测入口 — 完整版策略回测的标准化执行脚本

支持：
1. 单因子回测：指定因子名称和参数，自动检查数据、执行回测
2. 批量回测：--batch 模式，一次回测多个因子
3. 自动计算：--auto-compute 自动计算缺失因子并入库
4. 标准化输出：结果保存到 result_<factor_name>_*.csv

用法示例：
    # 单因子回测（自动推断参数和方向）
    python run_backtest_unified.py --factor carry_ret

    # 指定参数和方向
    python run_backtest_unified.py --factor skew --parameter lookback180 --long-low True

    # 批量回测所有活跃因子
    python run_backtest_unified.py --batch

    # 批量回测指定因子列表
    python run_backtest_unified.py --batch --factors carry_ret,skew,spread_return

    # 自动计算缺失因子后回测
    python run_backtest_unified.py --factor carry_ret --auto-compute

    # 只检查数据准备情况（不执行回测）
    python run_backtest_unified.py --factor carry_ret --check-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "factor_system"))

import argparse
import traceback
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.strategy import StrategyBacktester, calculate_portfolio_performance

from strategies.cross_sectional_strategy import CrossSectionalStrategy
from factor_system.factor_registry import get_registry
from factor_system.factor_engine import FactorEngine


DEFAULT_SYMBOLS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一回测入口 — 完整版策略回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_backtest_unified.py --factor carry_ret
  python run_backtest_unified.py --factor skew --long-low True
  python run_backtest_unified.py --batch --factors carry_ret,skew,kurtosis
  python run_backtest_unified.py --factor carry_ret --auto-compute
  python run_backtest_unified.py --factor carry_ret --check-only
        """
    )

    # 因子配置
    parser.add_argument("--factor", type=str, default="carry_ret",
                        help="因子名称（如 carry_ret, skew, spread_return）")
    parser.add_argument("--parameter", type=str, default=None,
                        help="因子参数标识（如 cycle=5, lookback=180）")
    parser.add_argument("--factor-author", type=str, default="factor_system",
                        help="因子作者标识（默认 factor_system）")

    # 策略参数
    parser.add_argument("--holding-period", type=int, default=5,
                        help="持仓周期（默认5天）")
    parser.add_argument("--trading-signal", type=float, default=0.2,
                        help="每端交易比例（默认0.2=20%）")
    parser.add_argument("--long-low", type=str, default=None,
                        help="True=做多低因子值，False=做多高因子值，None=自动推断")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean"],
                        help="仓位汇总模式（默认 sum）")
    parser.add_argument("--leverage", type=float, default=2.0,
                        help="名义杠杆倍数（默认2.0）")

    # 回测参数
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="回测开始日期（默认 2020-01-01）")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="回测结束日期（默认 2024-12-31）")
    parser.add_argument("--capital", type=int, default=10_000_000,
                        help="初始资金（默认 1000万）")
    parser.add_argument("--commission", type=float, default=0.0001,
                        help="单边手续费率（默认 0.0001=万1）")
    parser.add_argument("--risk-free", type=float, default=0,
                        help="无风险利率（默认 0）")
    parser.add_argument("--plot-chart", action="store_true",
                        help="是否绘制图表")

    # 批量与自动化
    parser.add_argument("--batch", action="store_true",
                        help="批量模式（配合 --factors 使用，或回测所有活跃因子）")
    parser.add_argument("--factors", type=str, default=None,
                        help="批量回测的因子列表，逗号分隔（如 carry_ret,skew,spread_return）")
    parser.add_argument("--auto-compute", action="store_true",
                        help="因子缺失时自动计算并入库")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查数据准备情况，不执行回测")
    parser.add_argument("--primary-suffix", type=str, default="99",
                        help="自动计算时使用的主力连续后缀（默认 99）")
    parser.add_argument("--secondary-suffix", type=str, default="889",
                        help="自动计算时使用的次主力连续后缀（默认 889）")

    return parser.parse_args()


def infer_parameter(factor_name: str) -> str:
    """从 FactorRegistry 推断默认参数"""
    registry = get_registry()
    meta = registry.get(factor_name)
    if meta and meta.params:
        return "&".join([f"{k}={v}" for k, v in meta.params.items()])
    return ""


def infer_long_low(factor_name: str) -> bool:
    """从 FactorRegistry 推断 long_low 方向"""
    registry = get_registry()
    meta = registry.get(factor_name)
    if meta:
        # ic_direction=-1 → 做多低值（long_low=True）
        # ic_direction=+1 → 做多高值（long_low=False）
        return meta.ic_direction == -1
    return False


def check_factor_exists(
    factor_name: str,
    parameter: str,
    author: str,
    symbols: List[str],
    start: datetime,
    end: datetime
) -> bool:
    """检查 DataCenter 中是否存在指定因子"""
    dc = DataCenter()
    # 尝试多种 parameter 格式
    param_candidates = [parameter]
    if parameter != "":
        param_candidates.append("")

    for param in param_candidates:
        try:
            df = dc.load_factor_df(
                vt_symbols=symbols[:3],  # 抽查前3个品种
                name=factor_name,
                interval="d",
                parameter=param,
                author=author,
                start=start,
                end=end
            )
            if len(df) > 0:
                return True
        except Exception:
            continue
    return False


def auto_compute_factor(
    factor_name: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    primary_suffix: str = "99",
    secondary_suffix: str = "889",
    verbose: bool = True
) -> bool:
    """自动计算缺失因子并保存到 DataCenter"""
    if verbose:
        print(f"\n[Auto-Compute] 自动计算因子 '{factor_name}'...")

    registry = get_registry()
    meta = registry.get(factor_name)
    if meta is None:
        print(f"  [ERROR] 因子 '{factor_name}' 未在注册表中定义，无法自动计算")
        return False

    try:
        engine = FactorEngine(
            dominant_symbols=symbols,
            start=start,
            end=end,
            primary_suffix=primary_suffix,
            secondary_suffix=secondary_suffix,
            verbose=verbose
        )
        engine.load_all_data()

        factor_df = engine.compute(factor_name)
        if factor_df is None or factor_df.empty:
            print(f"  [ERROR] 因子 '{factor_name}' 计算结果为空")
            return False

        param_str = "&".join([f"{k}={v}" for k, v in meta.params.items()])
        engine.save_factor(factor_df, factor_name, param_str, author="factor_system")

        if verbose:
            print(f"  [OK] 因子 '{factor_name}' 已计算并保存 (parameter='{param_str}')")
        return True

    except Exception as e:
        print(f"  [ERROR] 因子 '{factor_name}' 自动计算失败: {e}")
        traceback.print_exc()
        return False


def run_single_backtest(
    factor_name: str,
    parameter: str,
    factor_author: str,
    long_low: bool,
    holding_period: int,
    trading_signal: float,
    aggregation: str,
    leverage: float,
    start: datetime,
    end: datetime,
    capital: int,
    commission: float,
    risk_free: float,
    plot_chart: bool,
    symbols: List[str],
    verbose: bool = True
) -> Optional[Dict]:
    """执行单个因子的完整回测"""

    if verbose:
        print("\n" + "=" * 70)
        print(f"因子回测: {factor_name} | parameter={parameter} | long_low={long_low}")
        print("=" * 70)

    # 1. 检查因子是否存在
    exists = check_factor_exists(factor_name, parameter, factor_author, symbols, start, end)
    if not exists:
        print(f"\n[ERROR] 因子 '{factor_name}' 在 DataCenter 中不存在")
        print(f"\n解决方法（二选一）：")
        print(f"  1. 手动计算: python factor_system/run_factor_pipeline.py --mode compute --factors {factor_name}")
        print(f"  2. 自动计算: 添加 --auto-compute 参数")
        return None

    # 2. 创建回测器
    backtester = StrategyBacktester(symbols, Interval.DAILY, start, end, capital)

    # 3. 加载数据
    if verbose:
        print("\n[Step 1] 加载历史行情数据...")
    backtester.load_data()

    # 4. 配置策略
    strategy_setting = {
        "holding_period": holding_period,
        "trading_signal": trading_signal,
        "factor_name": factor_name,
        "factor_parameter": parameter,
        "factor_author": factor_author,
        "long_low": long_low,
        "aggregation": aggregation,
        "leverage": leverage,
    }

    # 5. 运行回测
    if verbose:
        print("\n[Step 2] 运行回测...")
    target_df = backtester.run_backtesting(CrossSectionalStrategy, strategy_setting)

    non_zero_days = (target_df != 0).any(axis=1).sum()
    if verbose:
        print(f"  有仓位的交易日: {non_zero_days}/{len(target_df)}")
        print(f"  涉及的具体合约数: {len(target_df.columns)}")

    # 6. 绩效分析
    if verbose:
        print("\n[Step 3] 绩效分析...")
    result = calculate_portfolio_performance(
        target_df,
        Interval.DAILY,
        commission=commission,
        capital=capital,
        risk_free=risk_free,
        plot_chart=plot_chart
    )

    stats = result["statistics"]
    if verbose:
        print("\n【完整回测绩效指标】")
        for key, value in stats.items():
            print(f"  {key:<20s}: {value}")

    # 7. 保存结果
    prefix = f"/root/cs_developer/result_{factor_name}"
    target_df.to_csv(f"{prefix}_target.csv")
    result["overall"][["balance", "drawdown", "ddpercent"]].to_csv(f"{prefix}_pnl.csv")
    result["product"].to_csv(f"{prefix}_product.csv")
    if verbose:
        print(f"\n[OK] 结果已保存到 {prefix}_*.csv")

    return {
        "factor_name": factor_name,
        "parameter": parameter,
        "long_low": long_low,
        "stats": stats,
        "target_df": target_df,
        "result": result,
    }


def main():
    args = parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # 确定回测的因子列表
    if args.batch:
        if args.factors:
            factor_list = [f.strip() for f in args.factors.split(",")]
        else:
            registry = get_registry()
            factor_list = [m.name for m in registry.get_active()]
            print(f"批量模式: 回测所有 {len(factor_list)} 个活跃因子")
    else:
        factor_list = [args.factor]

    # 解析 long_low
    long_low = None
    if args.long_low is not None:
        long_low = args.long_low.lower() in ("true", "1", "yes", "y")

    # ===== check-only 模式 =====
    if args.check_only:
        print("=" * 70)
        print("数据准备情况检查")
        print("=" * 70)
        all_ready = True
        for factor_name in factor_list:
            parameter = args.parameter
            if parameter is None:
                parameter = infer_parameter(factor_name)
            exists = check_factor_exists(
                factor_name, parameter, args.factor_author,
                DEFAULT_SYMBOLS, start, end
            )
            status = "✓ 已就绪" if exists else "✗ 缺失"
            print(f"  {factor_name:<25s} {status}  (parameter='{parameter}')")
            if not exists:
                all_ready = False
        print()
        if all_ready:
            print("[OK] 所有因子数据已准备就绪，可以执行回测")
        else:
            print("[WARN] 部分因子数据缺失，请运行以下命令计算：")
            missing = [f for f in factor_list if not check_factor_exists(
                f, infer_parameter(f) if args.parameter is None else args.parameter,
                args.factor_author, DEFAULT_SYMBOLS, start, end
            )]
            print(f"  python factor_system/run_factor_pipeline.py --mode compute --factors {','.join(missing)}")
        return

    # ===== 回测模式 =====
    results = []
    skipped = []

    for factor_name in factor_list:
        # 推断参数
        parameter = args.parameter
        if parameter is None:
            parameter = infer_parameter(factor_name)

        # 推断方向
        factor_long_low = long_low
        if factor_long_low is None:
            factor_long_low = infer_long_low(factor_name)

        # 检查因子存在性
        exists = check_factor_exists(
            factor_name, parameter, args.factor_author,
            DEFAULT_SYMBOLS, start, end
        )

        # 自动计算缺失因子
        if not exists and args.auto_compute:
            success = auto_compute_factor(
                factor_name, DEFAULT_SYMBOLS, start, end,
                args.primary_suffix, args.secondary_suffix
            )
            if success:
                exists = True

        if not exists:
            print(f"\n{'='*70}")
            print(f"[SKIP] 因子 '{factor_name}' 数据未准备，跳过回测")
            print(f"{'='*70}")
            skipped.append(factor_name)
            continue

        # 执行回测
        result = run_single_backtest(
            factor_name=factor_name,
            parameter=parameter,
            factor_author=args.factor_author,
            long_low=factor_long_low,
            holding_period=args.holding_period,
            trading_signal=args.trading_signal,
            aggregation=args.aggregation,
            leverage=args.leverage,
            start=start,
            end=end,
            capital=args.capital,
            commission=args.commission,
            risk_free=args.risk_free,
            plot_chart=args.plot_chart,
            symbols=DEFAULT_SYMBOLS,
            verbose=True
        )

        if result:
            results.append(result)
        else:
            skipped.append(factor_name)

    # 批量汇总
    if len(results) > 1 or (len(results) == 1 and args.batch):
        print("\n" + "=" * 70)
        print("回测结果汇总")
        print("=" * 70)
        summary_data = []
        for r in results:
            s = r["stats"]
            summary_data.append({
                "factor": r["factor_name"],
                "parameter": r["parameter"],
                "long_low": r["long_low"],
                "annual_return": s.get("annual_return", "N/A"),
                "sharpe_ratio": s.get("sharpe_ratio", "N/A"),
                "max_drawdown": s.get("max_drawdown", "N/A"),
                "calmar_ratio": s.get("calmar_ratio", "N/A"),
            })
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        summary_path = "/root/cs_developer/result_unified_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[OK] 汇总表已保存: {summary_path}")

    print("\n" + "=" * 70)
    print(f"统一回测执行完毕！成功 {len(results)}/{len(factor_list)} 个因子")
    if skipped:
        print(f"跳过: {', '.join(skipped)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
