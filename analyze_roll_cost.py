#!/usr/bin/env python3
"""
分析 888/889 后复权导致的换月成本偏差

核心问题：
- 888/889 是后复权连续合约，消除了换月跳空
- 真实交易中，换月跳空是实际损益（近月 vs 远月价差）
- 交易费用/滑点无法弥补这种结构性偏差

对比对象：
- 88（原始主力合约，有换月跳空）
- 888（后复权，无跳空）
- 99（成交量加权，保留部分跳空）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factor_system.factor_engine import FactorEngine
from datetime import datetime

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPORT_DIR = Path("/root/cs_developer/docs")

# =====================================================================
# 1. 加载三种指数的数据
# =====================================================================
symbols = ['RB88.SHFE', 'HC88.SHFE', 'I88.DCE', 'CU88.SHFE', 'AL88.SHFE']
start = datetime(2020, 1, 1)
end = datetime(2024, 12, 31)

print("=" * 60)
print("加载三种指数数据...")
print("=" * 60)

data = {}
for suffix in ["88", "99", "888", "889"]:
    engine = FactorEngine(symbols, start, end, primary_suffix=suffix, verbose=False)
    engine.load_all_data()
    data[suffix] = engine.close.copy()
    print(f"{suffix}: shape={engine.close.shape}")

# =====================================================================
# 2. 计算换月跳空（价格突变）
# =====================================================================
print("\n" + "=" * 60)
print("换月跳空分析（以 RB 为例）")
print("=" * 60)

symbol = "RB88.SHFE"
results = []

for suffix in ["88", "99", "888", "889"]:
    prices = data[suffix][symbol].dropna()
    
    # 日收益率
    returns = prices.pct_change().dropna()
    
    # 识别大跳空（超过 3% 的单日波动）
    big_jumps = returns[abs(returns) > 0.03]
    
    # 价格连续性：计算相邻日价格比率的标准差
    price_ratios = prices.shift(1) / prices
    ratio_std = price_ratios.dropna().std()
    
    results.append({
        "suffix": suffix,
        "big_jumps": len(big_jumps),
        "big_jump_pct": len(big_jumps) / len(returns) * 100,
        "return_std": returns.std(),
        "ratio_std": ratio_std,
        "min_price": prices.min(),
        "neg_days": (prices < 0).sum(),
    })
    
    print(f"\n{suffix}:")
    print(f"  大跳空(>3%)次数: {len(big_jumps)} ({len(big_jumps)/len(returns)*100:.1f}%)")
    print(f"  日收益率标准差: {returns.std():.4f}")
    print(f"  负价格天数: {(prices < 0).sum()}")

results_df = pd.DataFrame(results)

# =====================================================================
# 3. 可视化换月跳空差异
# =====================================================================
print("\n绘制换月跳空对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 上图：RB 价格走势对比
ax = axes[0, 0]
for suffix, color in zip(["88", "99", "888", "889"], ["blue", "green", "red", "purple"]):
    prices = data[suffix][symbol].dropna()
    # 归一化到起点
    norm_prices = prices / prices.iloc[0]
    ax.plot(norm_prices.index, norm_prices.values, label=f"{suffix}", alpha=0.7, linewidth=1)

ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
ax.set_title(f"{symbol} — Normalized Price Comparison")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price")
ax.legend()
ax.grid(True, alpha=0.3)

# 日收益率分布
ax = axes[0, 1]
for suffix, color in zip(["88", "99", "888", "889"], ["blue", "green", "red", "purple"]):
    prices = data[suffix][symbol].dropna()
    returns = prices.pct_change().dropna()
    ax.hist(returns, bins=100, alpha=0.3, label=f"{suffix} (std={returns.std():.4f})", color=color, density=True)

ax.set_xlim(-0.1, 0.1)
ax.set_title(f"{symbol} — Daily Return Distribution")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

# 大跳空次数对比（所有品种平均）
ax = axes[1, 0]
all_jump_stats = []
for suffix in ["88", "99", "888", "889"]:
    jump_counts = []
    for sym in symbols:
        prices = data[suffix][sym].dropna()
        returns = prices.pct_change().dropna()
        big_jumps = len(returns[abs(returns) > 0.03])
        jump_counts.append(big_jumps / len(returns) * 100)
    all_jump_stats.append(np.mean(jump_counts))

bars = ax.bar(["88", "99", "888", "889"], all_jump_stats, color=["blue", "green", "red", "purple"], alpha=0.7)
ax.set_ylabel("Big Jump (>3%) %")
ax.set_title("Average Big Jump Frequency (5 Symbols)")
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, all_jump_stats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.1f}%", ha="center", fontsize=10)

# 负价格天数对比
ax = axes[1, 1]
neg_stats = []
for suffix in ["88", "99", "888", "889"]:
    neg_counts = []
    for sym in symbols:
        prices = data[suffix][sym].dropna()
        neg_counts.append((prices < 0).sum())
    neg_stats.append(np.mean(neg_counts))

bars = ax.bar(["88", "99", "888", "889"], neg_stats, color=["blue", "green", "red", "purple"], alpha=0.7)
ax.set_ylabel("Average Negative Price Days")
ax.set_title("Average Negative Price Days (5 Symbols)")
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, neg_stats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}", ha="center", fontsize=10)

plt.tight_layout()
fig.savefig(REPORT_DIR / "roll_cost_analysis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[OK] 换月成本分析图已保存: {REPORT_DIR / 'roll_cost_analysis.png'}")

# =====================================================================
# 4. 量化换月成本
# =====================================================================
print("\n" + "=" * 60)
print("换月成本量化估算")
print("=" * 60)

print("""
假设：每年换月 12 次（每月一次），每次换月成本 = 近远月价差 / 近月价格

真实换月成本来源：
1. 价差成本：远月 - 近月（可正可负）
2. 交易费用：双边手续费 + 滑点
3. 流动性成本：远月合约滑点更大

888/889 后复权的问题：
- 通过复权因子消除换月跳空
- 使得历史价格序列过于平滑
- 导致：做多策略虚增收益，做空策略虚减收益

举例（螺纹钢 RB）：
- 近月合约 4000 元，远月合约 4100 元
- 真实换月：卖出近月 4000，买入远月 4100，净亏 100（2.5%）
- 888 连续合约：历史价格统一乘以 1.025，价格显示为 4100
- 简化回测看不到这 2.5% 的成本！

交易费用/滑点能弥补吗？
- 交易费用：通常 0.01%~0.05% 每边
- 滑点：通常 0.01%~0.1% 每边
- 换月成本：通常 0.5%~3%（远大于交易费用）
- 结论：交易费用和滑点无法弥补换月成本的缺失
""")

# 实际估算：88 vs 888 的累积收益差异
print("\n实际测算：88 vs 888 的累积收益差异（RB 2020-2024）")
for suffix in ["88", "888"]:
    prices = data[suffix][symbol].dropna()
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    print(f"  {suffix}: 总收益 = {total_return:.2%}")

diff = data["88"][symbol].dropna().iloc[-1] / data["88"][symbol].dropna().iloc[0] - \
       data["888"][symbol].dropna().iloc[-1] / data["888"][symbol].dropna().iloc[0]
print(f"\n  88 比 888 多赚: {diff:.2%}")
print(f"  这就是 888 后复权'隐藏'的换月成本！")

print("\n[OK] 分析完成!")
