#!/usr/bin/env python3
"""
Factor Pipeline — 因子批量计算与分析的统一执行入口

功能：
1. 加载数据（一次加载，全局缓存）
2. 批量计算所有注册因子（或指定因子子集）
3. 保存因子到 DataCenter
4. 执行 IC 分析（多期 forward returns）
5. 生成相关性矩阵
6. 输出图表与 Markdown 报告

命令行用法：
    # 计算所有活跃因子
    python run_factor_pipeline.py --mode compute --start 2020-01-01 --end 2024-12-31

    # 只计算指定类别
    python run_factor_pipeline.py --mode compute --categories momentum volatility

    # 执行 IC 分析（已有因子数据）
    python run_factor_pipeline.py --mode analyze --start 2020-01-01 --end 2024-12-31

    # 全流程：计算 + 分析
    python run_factor_pipeline.py --mode full --start 2020-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

# 路径设置
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from factor_registry import FactorRegistry, FactorStatus, get_registry
from factor_engine import get_engine, FactorEngine
from factor_monitor import FactorMonitor


# =============================================================================
# 默认配置
# =============================================================================

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

REPORT_DIR = "/root/cs_developer/factor_system/reports"
DATA_DIR = "/root/cs_developer/factor_system/data"


# =============================================================================
# 命令行解析
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Factor Pipeline — 批量因子计算与分析")
    
    parser.add_argument("--mode", type=str, default="full",
                        choices=["compute", "analyze", "full", "registry", "report"],
                        help="执行模式")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="品种列表文件路径，默认使用内置50个活跃品种")
    parser.add_argument("--factors", type=str, default=None,
                        help="指定因子列表，逗号分隔，默认所有活跃因子")
    parser.add_argument("--categories", type=str, default=None,
                        help="按类别筛选因子，逗号分隔")
    parser.add_argument("--output", type=str, default=REPORT_DIR,
                        help="报告输出目录")
    parser.add_argument("--save", action="store_true",
                        help="保存因子到 DataCenter")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在于数据库的因子")
    parser.add_argument("--forward-periods", type=str, default="1,5,10,20",
                        help="IC分析的forward periods，逗号分隔")
    
    return parser.parse_args()


def load_symbols(path: Optional[str]) -> List[str]:
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            return lines
    return DEFAULT_SYMBOLS


# =============================================================================
# 执行逻辑
# =============================================================================

def run_compute(engine: FactorEngine, factor_names: List[str], save: bool = False) -> dict:
    """批量计算因子"""
    print("\n" + "=" * 70)
    print("Phase 1: 批量因子计算")
    print("=" * 70)
    
    results = engine.compute_all(factor_names=factor_names)
    
    if save:
        print("\n" + "-" * 70)
        print("保存因子到 DataCenter...")
        for name, df in results.items():
            meta = engine.registry.get(name)
            if meta is None:
                continue
            param_str = "&".join([f"{k}={v}" for k, v in meta.params.items()])
            try:
                engine.save_factor(df, name, param_str, author="factor_system")
            except Exception as e:
                print(f"  [WARN] {name} 保存失败: {e}")
    
    return results


def run_analyze(engine: FactorEngine, factor_results: dict, forward_periods: List[int], output_dir: str):
    """执行 IC 分析并生成报告"""
    print("\n" + "=" * 70)
    print("Phase 2: IC 分析与监控")
    print("=" * 70)
    
    monitor = FactorMonitor(
        close_df=engine.close,
        forward_periods=forward_periods,
        output_dir=output_dir
    )
    
    # 批量 IC 分析
    analysis_results = monitor.batch_analyze(factor_results)
    
    # 相关性矩阵
    print("\n计算因子相关性矩阵...")
    corr_df = monitor.correlation_matrix(factor_results)
    if not corr_df.empty:
        corr_path = monitor.plot_correlation_matrix(corr_df)
        print(f"  相关性矩阵图: {corr_path}")
    
    # 图表
    print("\n生成图表...")
    for period in forward_periods:
        if any(period in res for res in analysis_results.values()):
            path = monitor.plot_ic_timeseries(analysis_results, period=period)
            print(f"  IC时间序列 ({period}D): {path}")
    
    heatmap_path = monitor.plot_ic_heatmap(analysis_results)
    print(f"  IC热力图: {heatmap_path}")
    
    # Markdown 报告
    print("\n生成 Markdown 报告...")
    report_path = monitor.generate_markdown_report(analysis_results, corr_df)
    print(f"  报告路径: {report_path}")
    
    # 保存汇总 CSV
    summary = monitor.generate_summary_table(analysis_results, period=5)
    if not summary.empty:
        csv_path = f"{output_dir}/factor_summary_5d.csv"
        summary.to_csv(csv_path, index=False)
        print(f"  汇总CSV: {csv_path}")
    
    return analysis_results


def run_registry_summary():
    """打印注册表摘要"""
    reg = get_registry()
    print(reg.generate_summary())
    
    # 详细列表
    print("\n" + "-" * 70)
    print("Active Factor List:")
    print("-" * 70)
    
    for cat in sorted(reg.get_categories()):
        factors = reg.get_by_category(cat)
        print(f"\n[{cat}] ({len(factors)})")
        for f in factors:
            status_mark = "✓" if f.status == FactorStatus.ACTIVE else "✗"
            print(f"  {status_mark} {f.name:<25s} | {f.description[:50]}...")


def main():
    args = parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    symbols = load_symbols(args.symbols)
    forward_periods = [int(p) for p in args.forward_periods.split(",")]
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 模式分支
    if args.mode == "registry":
        run_registry_summary()
        return
    
    # 初始化引擎
    print("=" * 70)
    print("Factor Pipeline — 初始化")
    print("=" * 70)
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"品种数量: {len(symbols)}")
    print(f"Forward Periods: {forward_periods}")
    
    engine = get_engine(symbols, start, end, verbose=True)
    engine.load_all_data()
    
    # 确定要计算的因子
    registry = get_registry()
    if args.factors:
        factor_names = [f.strip() for f in args.factors.split(",")]
    elif args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        factor_names = [m.name for m in registry.get_active() if m.category in cats]
    else:
        factor_names = [m.name for m in registry.get_active()]
    
    print(f"\n待计算因子: {len(factor_names)} 个")
    print(f"  {', '.join(factor_names[:10])}{'...' if len(factor_names) > 10 else ''}")
    
    # 执行
    if args.mode in ("compute", "full"):
        factor_results = run_compute(engine, factor_names, save=args.save)
        
        # 保存到本地缓存（用于 analyze 模式直接加载）
        cache_file = f"{DATA_DIR}/factor_results_{args.start}_{args.end}.json"
        # 保存因子名称列表用于后续分析
        with open(cache_file.replace('.json', '_meta.json'), 'w') as f:
            json.dump({
                "factor_names": list(factor_results.keys()),
                "start": args.start,
                "end": args.end,
                "symbols": symbols,
            }, f)
        
        # 保存因子 DataFrame 为 parquet（本地缓存）
        for name, df in factor_results.items():
            pq_path = f"{DATA_DIR}/{name}_{args.start}_{args.end}.parquet"
            df.to_parquet(pq_path)
    else:
        # analyze 模式：从本地缓存加载
        factor_results = {}
        for name in factor_names:
            pq_path = f"{DATA_DIR}/{name}_{args.start}_{args.end}.parquet"
            if os.path.exists(pq_path):
                factor_results[name] = pd.read_parquet(pq_path)
            else:
                print(f"[WARN] {name} 本地缓存不存在，尝试重新计算...")
                df = engine.compute(name)
                factor_results[name] = df
    
    if args.mode in ("analyze", "full"):
        run_analyze(engine, factor_results, forward_periods, args.output)
    
    print("\n" + "=" * 70)
    print("Factor Pipeline — 执行完毕")
    print("=" * 70)


if __name__ == "__main__":
    main()
