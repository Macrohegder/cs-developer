#!/usr/bin/env python3
"""
完整回测 vs 简化回测 对比分析

读取：
- 完整回测: result_{factor}_pnl.csv (balance, drawdown, ddpercent)
- 简化回测: factor_system/reports/batch_backtest_simple_*.csv

输出：
- docs/full_vs_simple_comparison.png
- docs/full_vs_simple_metrics.csv
"""

import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPORT_DIR = Path("/root/cs_developer/docs")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# =====================================================================
# 1. 读取简化回测结果
# =====================================================================
simple_files = sorted(glob.glob("/root/cs_developer/factor_system/reports/batch_backtest_simple_*.csv"))
simple_df = pd.read_csv(simple_files[-1])

# 清洗 inf
for col in ["annual_return", "sharpe_ratio", "calmar_ratio"]:
    simple_df[col] = simple_df[col].replace([np.inf, -np.inf], np.nan)

simple_map = simple_df.set_index("factor_name")[["annual_return", "sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate"]]

# =====================================================================
# 2. 从完整回测 pnl 文件提取绩效指标
# =====================================================================
print("=" * 60)
print("提取完整回测绩效指标...")
print("=" * 60)

full_metrics = []

for fpath in sorted(Path("/root/cs_developer").glob("result_*_pnl.csv")):
    factor_name = fpath.stem.replace("result_", "").replace("_pnl", "")
    
    df = pd.read_csv(fpath, parse_dates=["datetime"])
    if df.empty or "balance" not in df.columns:
        continue
    
    balance = df["balance"]
    initial = balance.iloc[0]
    final = balance.iloc[-1]
    total_days = len(balance)
    
    # 年化收益
    ann_return = (final / initial) ** (TRADING_DAYS / total_days) - 1
    
    # 日收益率
    daily_ret = balance.pct_change().dropna()
    
    # 夏普
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS)) if daily_ret.std() > 1e-12 else 0
    
    # 最大回撤
    max_dd = df["ddpercent"].min() / 100.0 if "ddpercent" in df.columns else 0
    
    # Calmar
    calmar = -ann_return / max_dd if max_dd < -1e-12 else np.inf
    
    # 胜率
    win_rate = (daily_ret > 0).mean()
    
    full_metrics.append({
        "factor_name": factor_name,
        "full_annual_return": ann_return,
        "full_sharpe_ratio": sharpe,
        "full_max_drawdown": max_dd,
        "full_calmar_ratio": calmar,
        "full_win_rate": win_rate,
        "full_final_nav": final / initial,
    })
    
    print(f"[OK] {factor_name:25s} | ann={ann_return:+.2%} | sharpe={sharpe:+.3f} | dd={max_dd:.2%} | nav={final/initial:.3f}")

full_df = pd.DataFrame(full_metrics)

# =====================================================================
# 3. 合并对比
# =====================================================================
print("\n" + "=" * 60)
print("合并对比...")
print("=" * 60)

merged = pd.merge(full_df, simple_map, left_on="factor_name", right_index=True, how="inner")

# 计算偏差
merged["return_diff"] = merged["full_annual_return"] - merged["annual_return"]
merged["sharpe_diff"] = merged["full_sharpe_ratio"] - merged["sharpe_ratio"]
merged["dd_diff"] = merged["full_max_drawdown"] - merged["max_drawdown"]

# 保存 CSV
merged.to_csv(REPORT_DIR / "full_vs_simple_metrics.csv", index=False)
print(f"[OK] 对比数据已保存: {REPORT_DIR / 'full_vs_simple_metrics.csv'}")

# =====================================================================
# 4. 绘制对比图
# =====================================================================
print("\n绘制对比图...")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))

# --- 4.1 年化收益散点图 ---
ax = axes[0, 0]
valid = merged[(merged["annual_return"].notna()) & (merged["full_annual_return"].notna())]
ax.scatter(valid["annual_return"] * 100, valid["full_annual_return"] * 100, alpha=0.6, s=50)
# 对角线
lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
ax.set_xlabel("Simplified Annual Return (%)")
ax.set_ylabel("Full Annual Return (%)")
ax.set_title("Annual Return: Full vs Simplified")
ax.grid(True, alpha=0.3)

# 标注极端点
for _, row in valid.iterrows():
    if abs(row["return_diff"]) > 0.5:  # 偏差 > 50%
        ax.annotate(row["factor_name"], 
                   (row["annual_return"]*100, row["full_annual_return"]*100),
                   fontsize=6, alpha=0.7)

# --- 4.2 夏普散点图 ---
ax = axes[0, 1]
valid_s = merged[(merged["sharpe_ratio"].notna()) & (merged["full_sharpe_ratio"].notna())]
ax.scatter(valid_s["sharpe_ratio"], valid_s["full_sharpe_ratio"], alpha=0.6, s=50, color="green")
lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
ax.set_xlabel("Simplified Sharpe Ratio")
ax.set_ylabel("Full Sharpe Ratio")
ax.set_title("Sharpe Ratio: Full vs Simplified")
ax.grid(True, alpha=0.3)

# --- 4.3 最大回撤散点图 ---
ax = axes[0, 2]
ax.scatter(merged["max_drawdown"] * 100, merged["full_max_drawdown"] * 100, alpha=0.6, s=50, color="red")
lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
ax.set_xlabel("Simplified Max Drawdown (%)")
ax.set_ylabel("Full Max Drawdown (%)")
ax.set_title("Max Drawdown: Full vs Simplified")
ax.grid(True, alpha=0.3)

# --- 4.4 年化收益偏差分布 ---
ax = axes[1, 0]
valid_rd = valid["return_diff"].dropna() * 100
ax.hist(valid_rd, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Diff")
ax.axvline(valid_rd.median(), color="orange", linestyle="-", linewidth=2, label=f"Median={valid_rd.median():.1f}%")
ax.set_xlabel("Return Diff (Full - Simple) %")
ax.set_ylabel("Count")
ax.set_title("Annual Return Deviation Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

# --- 4.5 夏普偏差分布 ---
ax = axes[1, 1]
valid_sd = valid_s["sharpe_diff"].dropna()
ax.hist(valid_sd, bins=30, edgecolor="black", alpha=0.7, color="forestgreen")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Diff")
ax.axvline(valid_sd.median(), color="orange", linestyle="-", linewidth=2, label=f"Median={valid_sd.median():.3f}")
ax.set_xlabel("Sharpe Diff (Full - Simple)")
ax.set_ylabel("Count")
ax.set_title("Sharpe Ratio Deviation Distribution")
ax.legend()
ax.grid(True, alpha=0.3)

# --- 4.6 偏差最大的因子列表 ---
ax = axes[1, 2]
ax.axis("off")

# 按年化收益偏差排序
top_diff = valid.reindex(valid["return_diff"].abs().sort_values(ascending=False).index).head(15)

table_data = []
for _, row in top_diff.iterrows():
    table_data.append([
        row["factor_name"],
        f"{row['full_annual_return']:.1%}",
        f"{row['annual_return']:.1%}",
        f"{row['return_diff']:.1%}",
    ])

table = ax.table(cellText=table_data,
                colLabels=["Factor", "Full", "Simple", "Diff"],
                cellLoc="left",
                loc="center",
                colWidths=[0.45, 0.18, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
ax.set_title("Top 15 Factors by Return Deviation", fontsize=12, fontweight="bold", pad=20)

plt.tight_layout()
fig.savefig(REPORT_DIR / "full_vs_simple_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 对比图已保存: {REPORT_DIR / 'full_vs_simple_comparison.png'}")

# =====================================================================
# 5. 统计摘要
# =====================================================================
print("\n" + "=" * 60)
print("统计摘要")
print("=" * 60)

print(f"\n可对齐因子数: {len(merged)}")
print(f"\n年化收益偏差:")
print(f"  均值: {valid['return_diff'].mean():.2%}")
print(f"  中位数: {valid['return_diff'].median():.2%}")
print(f"  标准差: {valid['return_diff'].std():.2%}")
print(f"  最大正向偏差: {valid['return_diff'].max():.2%} ({valid.loc[valid['return_diff'].idxmax(), 'factor_name']})")
print(f"  最大负向偏差: {valid['return_diff'].min():.2%} ({valid.loc[valid['return_diff'].idxmin(), 'factor_name']})")

print(f"\n夏普偏差:")
print(f"  均值: {valid_s['sharpe_diff'].mean():.3f}")
print(f"  中位数: {valid_s['sharpe_diff'].median():.3f}")
print(f"  标准差: {valid_s['sharpe_diff'].std():.3f}")

print(f"\n完整回测显著优于简化回测 (return_diff > 30%):")
better = valid[valid["return_diff"] > 0.30].sort_values("return_diff", ascending=False)
for _, row in better.iterrows():
    print(f"  {row['factor_name']:25s} | full={row['full_annual_return']:.1%} | simple={row['annual_return']:.1%} | diff={row['return_diff']:.1%}")

print(f"\n简化回测显著优于完整回测 (return_diff < -30%):")
worse = valid[valid["return_diff"] < -0.30].sort_values("return_diff", ascending=True)
for _, row in worse.iterrows():
    print(f"  {row['factor_name']:25s} | full={row['full_annual_return']:.1%} | simple={row['annual_return']:.1%} | diff={row['return_diff']:.1%}")

print("\n[OK] 全部对比分析完成!")
