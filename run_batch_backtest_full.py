#!/usr/bin/env python3
"""
批量完整回测 — 使用 StrategyBacktester 对所有因子执行完整回测

用法：
    python run_batch_backtest_full.py --start 2020-01-01 --end 2024-12-31
    python run_batch_backtest_full.py --categories carry,momentum
    python run_batch_backtest_full.py --factors carry_ret,skew,momentum

后台运行（tmux）：
    tmux new-session -d -s batch_full 'python run_batch_backtest_full.py'
"""

import os
import sys
import argparse
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from factor_system.factor_registry import get_registry


def parse_args():
    parser = argparse.ArgumentParser(description="批量完整回测")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--categories", type=str, default=None,
                        help="指定类别，逗号分隔")
    parser.add_argument("--factors", type=str, default=None,
                        help="指定因子，逗号分隔")
    parser.add_argument("--holding-period", type=int, default=5)
    parser.add_argument("--trading-signal", type=float, default=0.2)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--commission", type=float, default=0.0001)
    return parser.parse_args()


def run_unified_backtest(factor_name, start, end, holding_period, trading_signal, leverage, commission):
    """调用 run_backtest_unified.py 执行单个因子的完整回测"""
    cmd = [
        "python3", "run_backtest_unified.py",
        "--factor", factor_name,
        "--start", start,
        "--end", end,
        "--holding-period", str(holding_period),
        "--trading-signal", str(trading_signal),
        "--leverage", str(leverage),
        "--commission", str(commission),
        "--factor-author", "factor_system",
    ]
    
    print(f"\n{'='*70}")
    print(f"完整回测: {factor_name}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, cwd="/root/cs_developer", capture_output=False, text=True, timeout=600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {factor_name} 回测超时 (>600s)")
        return False
    except Exception as e:
        print(f"[ERROR] {factor_name}: {e}")
        return False


def main():
    args = parse_args()
    
    registry = get_registry()
    all_factors = registry.get_active()
    
    # 筛选
    if args.factors:
        target_names = set(args.factors.split(","))
        factor_names = [m.name for m in all_factors if m.name in target_names]
    elif args.categories:
        cats = set(args.categories.split(","))
        factor_names = [m.name for m in all_factors if m.category in cats]
    else:
        factor_names = [m.name for m in all_factors]
    
    factor_names = sorted(factor_names)
    
    print(f"批量完整回测 — 共 {len(factor_names)} 个因子")
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"预计总耗时: {len(factor_names) * 3}~{len(factor_names) * 5} 分钟")
    print(f"\n按 Ctrl+C 中断，或用 tmux attach 查看进度")
    print(f"{'='*70}\n")
    
    success_count = 0
    skip_count = 0
    
    for idx, name in enumerate(factor_names, 1):
        print(f"\n[{idx}/{len(factor_names)}] 开始回测: {name}")
        success = run_unified_backtest(
            name, args.start, args.end,
            args.holding_period, args.trading_signal,
            args.leverage, args.commission
        )
        if success:
            success_count += 1
        else:
            skip_count += 1
    
    print(f"\n{'='*70}")
    print("批量完整回测执行完毕")
    print(f"成功: {success_count}/{len(factor_names)}")
    print(f"失败/跳过: {skip_count}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
