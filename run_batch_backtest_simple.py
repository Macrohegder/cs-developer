#!/usr/bin/env python3
"""
批量简化回测 — 向量化批量回测所有因子

用法：
    python run_batch_backtest_simple.py --start 2020-01-01 --end 2024-12-31
    python run_batch_backtest_simple.py --factors carry_ret,skew,momentum

后台运行（tmux）：
    tmux new-session -d -s batch_simple 'python run_batch_backtest_simple.py'
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
from factor_system.backtest_engine import BatchBacktestEngine
from factor_system.factor_registry import get_registry


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
    parser = argparse.ArgumentParser(description="批量简化回测")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--factors", type=str, default=None,
                        help="指定因子，逗号分隔。默认所有")
    parser.add_argument("--data-dir", type=str,
                        default="/root/cs_developer/factor_system/data")
    parser.add_argument("--output-dir", type=str,
                        default="/root/cs_developer/factor_system/reports")
    parser.add_argument("--holding-period", type=int, default=5)
    parser.add_argument("--trading-signal", type=float, default=0.2)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--commission", type=float, default=0.0001)
    return parser.parse_args()


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定因子列表
    if args.factors:
        factor_names = args.factors.split(",")
    else:
        # 从注册表获取所有活跃因子
        registry = get_registry()
        factor_names = sorted([m.name for m in registry.get_active()])
    
    print(f"批量简化回测 — 共 {len(factor_names)} 个因子")
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"数据目录: {data_dir}")
    
    # 创建回测引擎
    engine = BatchBacktestEngine(
        dominant_symbols=DEFAULT_SYMBOLS,
        start=args.start,
        end=args.end,
        data_dir=data_dir,
        verbose=True
    )
    
    # 批量回测
    results = engine.run_all(
        factor_names=factor_names,
        holding_period=args.holding_period,
        trading_signal=args.trading_signal,
        leverage=args.leverage,
        commission=args.commission,
    )
    
    # 汇总对比表
    if results:
        df = engine.compare_results(results, sort_by="sharpe_ratio", ascending=False)
        
        # 保存 CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"batch_backtest_simple_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] 结果已保存: {csv_path}")
        
        # 打印前10
        print("\n【Top 10 因子】")
        print(df.head(10).to_string(index=False))
    else:
        print("\n[WARN] 无有效回测结果")


if __name__ == "__main__":
    main()
