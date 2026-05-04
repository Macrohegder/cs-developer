#!/usr/bin/env python3
"""
Top 10 + Carry 因子完整回测净值对比 + 策略相关性分析

输出:
    factor_system/reports/top_factors_nav_comparison.png
    factor_system/reports/top_factors_correlation_heatmap.png
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPORT_DIR = Path("/root/cs_developer/factor_system/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 因子列表
# =====================================================================
TOP10_FACTORS = [
    "skew_180", "kurtosis", "sharpe_ratio", "sharpe_ratio_889",
    "skew", "coef_of_variation", "duvol", "max_drawdown",
    "ulcer_index", "rsi_momentum_889"
]

CARRY_FACTORS = [
    "carry_ret", "carry_momentum", "carry_ret_889", "carry_momentum_889",
    "spread_zscore", "spread_zscore_889", "spread_return", "spread_return_889"
]

ALL_FACTORS = TOP10_FACTORS + CARRY_FACTORS

# =====================================================================
# 读取净值数据
# =====================================================================
nav_data = {}
returns_data = {}

print("=" * 60)
print("读取净值数据...")
print("=" * 60)

for factor in ALL_FACTORS:
    fpath = Path(f"/root/cs_developer/result_{factor}_pnl.csv")
    if not fpath.exists():
        print(f"[SKIP] {factor}: 文件不存在")
        continue
    
    df = pd.read_csv(fpath, parse_dates=["datetime"], index_col="datetime")
    if df.empty or "balance" not in df.columns:
        print(f"[SKIP] {factor}: 数据为空")
        continue
    
    # 归一化净值 (初始资金 1000万)
    nav = df["balance"] / 10_000_000.0
    nav_data[factor] = nav
    
    # 计算日收益率
    ret = nav.pct_change().dropna()
    returns_data[factor] = ret
    
    print(f"[OK] {factor:25s} | 交易日={len(nav):4d} | 最终净值={nav.iloc[-1]:.4f}")

# =====================================================================
# 对齐日期
# =====================================================================
common_index = None
for nav in nav_data.values():
    if common_index is None:
        common_index = nav.index
    else:
        common_index = common_index.intersection(nav.index)

print(f"\n对齐后共同交易日: {len(common_index)}")

nav_aligned = pd.DataFrame({k: v.reindex(common_index) for k, v in nav_data.items()})
returns_aligned = pd.DataFrame({k: v.reindex(common_index) for k, v in returns_data.items()})

# =====================================================================
# 图 1: 净值走势对比
# =====================================================================
print("\n绘制净值走势对比图...")

fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# 上图: Top 10
ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(TOP10_FACTORS)))
for i, factor in enumerate(TOP10_FACTORS):
    if factor in nav_aligned.columns:
        ax1.plot(nav_aligned.index, nav_aligned[factor], label=factor, color=colors[i], linewidth=1.2)

ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_title("Top 10 Factors — Full Backtest NAV Comparison", fontsize=14, fontweight="bold")
ax1.set_xlabel("Date")
ax1.set_ylabel("Normalized NAV (Start=1.0)")
ax1.legend(loc="upper left", fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# 下图: Carry 类
ax2 = axes[1]
colors2 = plt.cm.tab10(np.linspace(0, 1, len(CARRY_FACTORS)))
for i, factor in enumerate(CARRY_FACTORS):
    if factor in nav_aligned.columns:
        ax2.plot(nav_aligned.index, nav_aligned[factor], label=factor, color=colors2[i], linewidth=1.2)

ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_title("Carry Factors — Full Backtest NAV Comparison", fontsize=14, fontweight="bold")
ax2.set_xlabel("Date")
ax2.set_ylabel("Normalized NAV (Start=1.0)")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(REPORT_DIR / "top_factors_nav_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 净值对比图已保存: {REPORT_DIR / 'top_factors_nav_comparison.png'}")

# =====================================================================
# 图 2: 策略相关性热力图
# =====================================================================
print("\n计算策略相关性...")

# 计算收益率相关性矩阵
corr_matrix = returns_aligned.corr()

# 绘制热力图
fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

# 设置刻度
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr_matrix.columns, fontsize=8)

# 添加数值标注
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        if i >= j:  # 只标注下三角
            text = ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=6)

ax.set_title("Strategy Daily Return Correlation Matrix (Top 10 + Carry)", fontsize=14, fontweight="bold", pad=20)
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

plt.tight_layout()
fig.savefig(REPORT_DIR / "top_factors_correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 相关性热力图已保存: {REPORT_DIR / 'top_factors_correlation_heatmap.png'}")

# =====================================================================
# 相关性统计输出
# =====================================================================
print("\n" + "=" * 60)
print("相关性分析")
print("=" * 60)

# 提取上三角（排除对角线）
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append((
            corr_matrix.columns[i],
            corr_matrix.columns[j],
            corr_matrix.values[i, j]
        ))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\n相关性最高的 10 对策略:")
for f1, f2, c in corr_pairs[:10]:
    print(f"  {f1:25s} <-> {f2:25s} | corr={c:+.3f}")

print("\n相关性最低的 10 对策略:")
for f1, f2, c in corr_pairs[-10:]:
    print(f"  {f1:25s} <-> {f2:25s} | corr={c:+.3f}")

# =====================================================================
# 因子组合建议
# =====================================================================
print("\n" + "=" * 60)
print("低相关性组合建议 (|corr| < 0.3)")
print("=" * 60)

# 以 skew_180 为锚点，找与其低相关的其他 Top 因子
if "skew_180" in corr_matrix.columns:
    skew_corr = corr_matrix["skew_180"].drop("skew_180").sort_values(key=lambda x: abs(x))
    print(f"\n与 skew_180 相关性最低的因子:")
    for factor, corr in skew_corr.head(5).items():
        print(f"  {factor:25s} | corr={corr:+.3f}")

print("\n[OK] 全部可视化完成!")
