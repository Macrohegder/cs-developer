#!/usr/bin/env python3
"""
Top 10 因子完整回测可视化（按完整回测 Sharpe 排序）

输出:
    docs/top10_full_backtest_nav.png          — NAV 叠加图
    docs/top10_full_backtest_metrics.png      — 指标对比图
    docs/top10_full_backtest_correlation.png  — 相关性热力图
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DOCS_DIR = Path("/root/cs_developer/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Top 10 因子（按完整回测 Sharpe 排序，去重）
# =====================================================================
TOP10_FACTORS = [
    "carry_momentum_889",   # 1.177
    "oi_change_889",        # 1.056
    "kurtosis_889",         # 1.038
    "kurtosis",             # 0.834
    "skew_180",             # 0.654
    "skew",                 # 0.636
    "carry_ret",            # 0.602
    "timevol",              # 0.584
    "trend_coeff_889",      # 0.559
    "carry_momentum",       # 0.510
]

# =====================================================================
# 读取净值数据并计算指标
# =====================================================================
nav_data = {}
returns_data = {}
metrics = []

print("=" * 70)
print("Top 10 因子完整回测 — 读取净值数据")
print("=" * 70)

for factor in TOP10_FACTORS:
    fpath = Path(f"/root/cs_developer/result_{factor}_pnl.csv")
    if not fpath.exists():
        print(f"[SKIP] {factor}: 文件不存在")
        continue

    df = pd.read_csv(fpath, parse_dates=["datetime"], index_col="datetime")
    if df.empty or "balance" not in df.columns:
        print(f"[SKIP] {factor}: 数据为空")
        continue

    nav = df["balance"] / 10_000_000.0
    nav_data[factor] = nav

    ret = nav.pct_change().dropna()
    returns_data[factor] = ret

    # 计算指标
    annual_return = (nav.iloc[-1] ** (240 / len(nav))) - 1
    annual_vol = ret.std() * np.sqrt(240)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = (ret > 0).mean()

    metrics.append({
        "factor": factor,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "final_nav": nav.iloc[-1],
    })

    print(f"[OK] {factor:25s} | 年化={annual_return:+.2%} | 夏普={sharpe:+.3f} | 最大回撤={max_dd:+.2%}")

metrics_df = pd.DataFrame(metrics).sort_values("sharpe", ascending=False)

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
# 图 1: NAV 叠加图
# =====================================================================
print("\n绘制 NAV 叠加图...")

fig, ax = plt.subplots(figsize=(16, 9))

colors = plt.cm.tab10(np.linspace(0, 1, len(nav_aligned.columns)))
for i, factor in enumerate(nav_aligned.columns):
    label = f"{factor} (SR={metrics_df[metrics_df['factor']==factor]['sharpe'].values[0]:.2f})"
    ax.plot(nav_aligned.index, nav_aligned[factor], label=label,
            color=colors[i], linewidth=1.5, alpha=0.85)

ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_title("Top 10 Factors — Full Backtest NAV (Ranked by Sharpe)", fontsize=15, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Normalized NAV (Start = 1.0)", fontsize=12)
ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0.5)

plt.tight_layout()
fig.savefig(DOCS_DIR / "top10_full_backtest_nav.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] NAV 图: {DOCS_DIR / 'top10_full_backtest_nav.png'}")

# =====================================================================
# 图 2: 指标对比柱状图
# =====================================================================
print("\n绘制指标对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Top 10 Factors — Performance Metrics Comparison (Full Backtest)", fontsize=14, fontweight="bold")

factors = metrics_df["factor"].tolist()
x = np.arange(len(factors))
width = 0.6
colors_bar = plt.cm.Spectral(np.linspace(0, 1, len(factors)))

# 年化收益
ax = axes[0, 0]
vals = metrics_df["annual_return"] * 100
bars = ax.bar(x, vals, width, color=colors_bar, edgecolor="black", linewidth=0.3)
ax.set_ylabel("Annual Return (%)", fontsize=11)
ax.set_title("Annual Return", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=8)

# 夏普
ax = axes[0, 1]
vals = metrics_df["sharpe"]
bars = ax.bar(x, vals, width, color=colors_bar, edgecolor="black", linewidth=0.3)
ax.set_ylabel("Sharpe Ratio", fontsize=11)
ax.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"{v:.2f}",
            ha="center", va="bottom", fontsize=8)

# 最大回撤
ax = axes[1, 0]
vals = metrics_df["max_drawdown"] * 100
bars = ax.bar(x, vals, width, color=colors_bar, edgecolor="black", linewidth=0.3)
ax.set_ylabel("Max Drawdown (%)", fontsize=11)
ax.set_title("Max Drawdown", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5, f"{v:.1f}%",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold")

# Calmar
ax = axes[1, 1]
vals = metrics_df["calmar"]
bars = ax.bar(x, vals, width, color=colors_bar, edgecolor="black", linewidth=0.3)
ax.set_ylabel("Calmar Ratio", fontsize=11)
ax.set_title("Calmar Ratio", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.1f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(DOCS_DIR / "top10_full_backtest_metrics.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 指标图: {DOCS_DIR / 'top10_full_backtest_metrics.png'}")

# =====================================================================
# 图 3: 相关性热力图
# =====================================================================
print("\n绘制相关性热力图...")

corr_matrix = returns_aligned.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(corr_matrix.columns, fontsize=10)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        if i >= j:
            text = ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9)

ax.set_title("Daily Return Correlation — Top 10 Factors (Full Backtest)", fontsize=14, fontweight="bold", pad=20)
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

plt.tight_layout()
fig.savefig(DOCS_DIR / "top10_full_backtest_correlation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 相关性图: {DOCS_DIR / 'top10_full_backtest_correlation.png'}")

# =====================================================================
# 统计输出
# =====================================================================
print("\n" + "=" * 70)
print("Top 10 因子完整回测指标汇总")
print("=" * 70)
print(metrics_df[["factor", "annual_return", "sharpe", "max_drawdown", "calmar", "win_rate", "final_nav"]]
      .to_string(index=False, formatters={
          "annual_return": lambda x: f"{x:+.2%}",
          "sharpe": lambda x: f"{x:+.3f}",
          "max_drawdown": lambda x: f"{x:+.2%}",
          "calmar": lambda x: f"{x:.1f}",
          "win_rate": lambda x: f"{x:.1%}",
          "final_nav": lambda x: f"{x:.3f}",
      }))

# 低相关性组合
print("\n" + "=" * 70)
print("低相关性组合建议 (|corr| < 0.5)")
print("=" * 70)

corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.values[i, j]))

corr_pairs.sort(key=lambda x: abs(x[2]))
print("\n相关性最低的 5 对:")
for f1, f2, c in corr_pairs[:5]:
    print(f"  {f1:25s} <-> {f2:25s} | corr={c:+.3f}")

print("\n[OK] 全部可视化完成!")
