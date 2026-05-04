#!/usr/bin/env python3
"""
Top 3 等权组合构建 — carry_momentum_889 + oi_change_889 + kurtosis_889

输出:
    docs/top3_portfolio_nav.png      — 组合 vs 单因子 NAV 对比
    docs/top3_portfolio_report.txt   — 组合绩效报告
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

TOP3 = ["carry_momentum_889", "oi_change_889", "kurtosis_889"]

# =====================================================================
# 读取数据
# =====================================================================
print("=" * 70)
print("Top 3 等权组合构建")
print("=" * 70)

nav_data = {}
returns_data = {}

for factor in TOP3:
    fpath = Path(f"/root/cs_developer/result_{factor}_pnl.csv")
    df = pd.read_csv(fpath, parse_dates=["datetime"], index_col="datetime")
    nav = df["balance"] / 10_000_000.0
    nav_data[factor] = nav
    returns_data[factor] = nav.pct_change().dropna()
    print(f"[OK] {factor:25s} | 交易日={len(nav):4d} | 最终净值={nav.iloc[-1]:.4f}")

# =====================================================================
# 对齐日期并构建组合
# =====================================================================
common_index = None
for nav in nav_data.values():
    common_index = nav.index if common_index is None else common_index.intersection(nav.index)

print(f"\n对齐后共同交易日: {len(common_index)}")

# 对齐收益率
rets_aligned = pd.DataFrame({k: v.reindex(common_index) for k, v in returns_data.items()})

# 等权组合日收益
portfolio_ret = rets_aligned.mean(axis=1)

# 组合 NAV（从 1.0 开始）
portfolio_nav = (1 + portfolio_ret).cumprod()

# =====================================================================
# 计算组合指标
# =====================================================================
def calc_metrics(nav, ret):
    annual_return = (nav.iloc[-1] ** (240 / len(nav))) - 1
    annual_vol = ret.std() * np.sqrt(240)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = (ret > 0).mean()
    return annual_return, sharpe, max_dd, calmar, win_rate

# 单因子指标
single_metrics = []
for factor in TOP3:
    nav = nav_data[factor].reindex(common_index)
    ret = rets_aligned[factor]
    ar, sr, mdd, cm, wr = calc_metrics(nav, ret)
    single_metrics.append({
        "name": factor,
        "annual_return": ar,
        "sharpe": sr,
        "max_drawdown": mdd,
        "calmar": cm,
        "win_rate": wr,
        "final_nav": nav.iloc[-1],
    })

# 组合指标
ar_p, sr_p, mdd_p, cm_p, wr_p = calc_metrics(portfolio_nav, portfolio_ret)
portfolio_metrics = {
    "name": "Portfolio (Equal-Weight)",
    "annual_return": ar_p,
    "sharpe": sr_p,
    "max_drawdown": mdd_p,
    "calmar": cm_p,
    "win_rate": wr_p,
    "final_nav": portfolio_nav.iloc[-1],
}

# =====================================================================
# 打印报告
# =====================================================================
print("\n" + "=" * 70)
print("绩效对比")
print("=" * 70)
print(f"{'名称':<30s} {'年化收益':>10s} {'夏普':>8s} {'最大回撤':>10s} {'Calmar':>8s} {'胜率':>8s} {'最终净值':>10s}")
print("-" * 90)
for m in single_metrics:
    print(f"{m['name']:<30s} {m['annual_return']:>+9.2%} {m['sharpe']:>+8.3f} {m['max_drawdown']:>+9.2%} {m['calmar']:>8.1f} {m['win_rate']:>7.1%} {m['final_nav']:>10.3f}")
print("-" * 90)
print(f"{portfolio_metrics['name']:<30s} {portfolio_metrics['annual_return']:>+9.2%} {portfolio_metrics['sharpe']:>+8.3f} {portfolio_metrics['max_drawdown']:>+9.2%} {portfolio_metrics['calmar']:>8.1f} {portfolio_metrics['win_rate']:>7.1%} {portfolio_metrics['final_nav']:>10.3f}")

# 组合优势计算
best_single = max(single_metrics, key=lambda x: x["sharpe"])
print(f"\n组合夏普 = {sr_p:.3f} vs 最佳单因子 ({best_single['name']}) 夏普 = {best_single['sharpe']:.3f}")
print(f"组合最大回撤 = {mdd_p:.2%} vs 最佳单因子最大回撤 = {best_single['max_drawdown']:.2%}")

# 风险调整收益提升
risk_adj_improvement = (sr_p / best_single["sharpe"] - 1) * 100 if best_single["sharpe"] > 0 else 0
print(f"风险调整收益提升: {risk_adj_improvement:+.1f}%")

# =====================================================================
# 可视化
# =====================================================================
print("\n绘制组合 NAV 对比图...")

fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})

# 上图: NAV
ax1 = axes[0]
colors = ["#2E86AB", "#A23B72", "#F18F01", "#1B1B1E"]
for i, factor in enumerate(TOP3):
    nav = nav_data[factor].reindex(common_index)
    ax1.plot(nav.index, nav.values, label=f"{factor} (SR={single_metrics[i]['sharpe']:.2f})",
             color=colors[i], linewidth=1.5, alpha=0.7)

ax1.plot(portfolio_nav.index, portfolio_nav.values,
         label=f"Portfolio Equal-Weight (SR={sr_p:.2f})",
         color=colors[3], linewidth=2.5, linestyle="--")

ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_title("Top 3 Equal-Weight Portfolio vs Individual Factors (Full Backtest)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Normalized NAV (Start = 1.0)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0.85)

# 下图: 组合回撤
ax2 = axes[1]
running_max = portfolio_nav.cummax()
drawdown = (portfolio_nav - running_max) / running_max
ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="#C73E1D", alpha=0.4)
ax2.plot(drawdown.index, drawdown.values * 100, color="#C73E1D", linewidth=1.2)
ax2.set_ylabel("Drawdown (%)", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_title(f"Portfolio Drawdown (Max = {mdd_p:.2%})", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(DOCS_DIR / "top3_portfolio_nav.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] NAV 图: {DOCS_DIR / 'top3_portfolio_nav.png'}")

# =====================================================================
# 保存报告
# =====================================================================
report_path = DOCS_DIR / "top3_portfolio_report.txt"
with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("Top 3 等权组合完整回测报告\n")
    f.write("=" * 70 + "\n\n")
    f.write("组合构成:\n")
    for factor in TOP3:
        f.write(f"  - {factor}\n")
    f.write(f"\n权重: 各 1/3 (等权再平衡)\n")
    f.write(f"回测区间: {common_index[0].strftime('%Y-%m-%d')} ~ {common_index[-1].strftime('%Y-%m-%d')}\n")
    f.write(f"交易日数: {len(common_index)}\n\n")
    
    f.write("-" * 70 + "\n")
    f.write("单因子绩效\n")
    f.write("-" * 70 + "\n")
    for m in single_metrics:
        f.write(f"\n{m['name']}:\n")
        f.write(f"  年化收益:    {m['annual_return']:+.2%}\n")
        f.write(f"  夏普比率:    {m['sharpe']:+.3f}\n")
        f.write(f"  最大回撤:    {m['max_drawdown']:+.2%}\n")
        f.write(f"  Calmar:      {m['calmar']:.1f}\n")
        f.write(f"  胜率:        {m['win_rate']:.1%}\n")
        f.write(f"  最终净值:    {m['final_nav']:.3f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("组合绩效\n")
    f.write("=" * 70 + "\n")
    f.write(f"  年化收益:    {ar_p:+.2%}\n")
    f.write(f"  夏普比率:    {sr_p:+.3f}\n")
    f.write(f"  最大回撤:    {mdd_p:+.2%}\n")
    f.write(f"  Calmar:      {cm_p:.1f}\n")
    f.write(f"  胜率:        {wr_p:.1%}\n")
    f.write(f"  最终净值:    {portfolio_nav.iloc[-1]:.3f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("组合优势分析\n")
    f.write("=" * 70 + "\n")
    f.write(f"  组合夏普 vs 最佳单因子夏普:  {sr_p:.3f} vs {best_single['sharpe']:.3f}\n")
    f.write(f"  风险调整收益提升:            {risk_adj_improvement:+.1f}%\n")
    f.write(f"  组合最大回撤 vs 最佳单因子:  {mdd_p:.2%} vs {best_single['max_drawdown']:.2%}\n")
    f.write(f"  回撤改善:                    {(mdd_p - best_single['max_drawdown'])*100:+.1f}pp\n")

print(f"[OK] 报告已保存: {report_path}")
print("\n[OK] 全部完成!")
