#!/bin/bash
# 批量因子计算 + 简化回测 + 完整回测 流水线
# 用法: tmux new-session -d -s pipeline 'bash run_all_pipeline.sh'

set -e
cd /root/cs_developer

START="2020-01-01"
END="2024-12-31"

# 确保输出目录存在
mkdir -p factor_system/data
mkdir -p factor_system/reports

echo "========================================"
echo "Phase 1/3: 批量因子计算 (80 个因子)"
echo "时间范围: $START ~ $END"
echo "预计耗时: 30-60 分钟"
echo "========================================"
python3 run_batch_compute.py --start $START --end $END

echo ""
echo "========================================"
echo "Phase 2/3: 批量简化回测"
echo "预计耗时: 10-20 分钟"
echo "========================================"
python3 run_batch_backtest_simple.py --start $START --end $END

echo ""
echo "========================================"
echo "Phase 3/3: 批量完整回测"
echo "预计耗时: 4-6 小时"
echo "========================================"
python3 run_batch_backtest_full.py --start $START --end $END

echo ""
echo "========================================"
echo "全部任务执行完毕"
echo "========================================"
