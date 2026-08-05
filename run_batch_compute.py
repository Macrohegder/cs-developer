#!/usr/bin/env python3
"""
批量因子计算 — 基于 88/88A2 的因子值计算

执行逻辑：
1. 创建 FactorEngine(88) 计算所有活跃因子
2. 保存到 DataCenter

用法：
    python run_batch_compute.py --start 2020-01-01 --end 2024-12-31
    python run_batch_compute.py --start 2020-01-01 --end 2024-12-31 --categories momentum carry

后台运行（tmux）：
    tmux new-session -d -s batch_compute 'python run_batch_compute.py'
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
from factor_system.factor_registry import get_registry, FactorStatus
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
    parser = argparse.ArgumentParser(description="批量因子计算 — 80 个因子")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--categories", type=str, default=None,
                        help="指定类别，逗号分隔（如 momentum,carry）")
    parser.add_argument("--factors", type=str, default=None,
                        help="指定因子，逗号分隔")
    parser.add_argument("--output-dir", type=str,
                        default="/root/cs_developer/factor_system/data",
                        help="因子数据输出目录")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批计算因子数，控制内存")
    return parser.parse_args()


def compute_factors_batch(suffix, factor_names, symbols, start, end, output_dir):
    """用指定 suffix 的引擎批量计算因子并保存"""
    
    is_carry = lambda n: n.startswith(("carry_", "spread_"))
    
    sec_suffix = "88A2"  # 次主力始终用 88A2
    
    print(f"\n{'='*70}")
    print(f"批量计算: suffix={suffix}, 因子数={len(factor_names)}")
    print(f"{'='*70}")
    
    engine = FactorEngine(
        dominant_symbols=symbols,
        start=start,
        end=end,
        primary_suffix=suffix,
        secondary_suffix=sec_suffix,
        verbose=True
    )
    engine.load_all_data()
    
    results = {}
    registry = get_registry()
    
    for idx, name in enumerate(factor_names, 1):
        print(f"\n[{idx}/{len(factor_names)}] {name} ...", end=" ")
        try:
            df = engine.compute(name)
            results[name] = df
            
            # 保存到本地 parquet
            meta = registry.get(name)
            param = meta.params.get("cycle", meta.params.get("lookback", "default")) if meta else "default"
            fname = f"{name}_{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.parquet"
            fpath = Path(output_dir) / fname
            df.to_parquet(fpath)
            
            # 同时保存到 DataCenter（供完整回测使用）
            try:
                param_str = "&".join([f"{k}={v}" for k, v in meta.params.items()]) if meta else "default"
                engine.save_factor(df, name, param_str, author="factor_system")
            except Exception as e2:
                print(f"[WARN] DataCenter 保存失败: {e2}")
            
            print(f"OK shape={df.shape} -> {fname}")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
    
    return results


def main():
    args = parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registry = get_registry()
    all_factors = registry.get_active()
    
    # 筛选
    if args.factors:
        target_names = set(args.factors.split(","))
        all_factors = [m for m in all_factors if m.name in target_names]
    elif args.categories:
        cats = set(args.categories.split(","))
        all_factors = [m for m in all_factors if m.category in cats]
    
    factor_names = [m.name for m in all_factors]
    
    print(f"总因子数: {len(all_factors)}")
    
    # 计算所有因子
    if factor_names:
        compute_factors_batch("88", factor_names, DEFAULT_SYMBOLS, start, end, output_dir)
    
    print(f"\n{'='*70}")
    print("[OK] 全部因子计算完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
