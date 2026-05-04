#!/usr/bin/env python3
"""
批量回测汇总报告生成器

功能：
1. 读取简化回测结果 (batch_backtest_simple_*.csv)
2. 读取完整回测结果 (result_unified_summary.csv)
3. 对比 88 vs 889 计算版本
4. 对比简化 vs 完整回测
5. 生成 Markdown 汇总报告

用法：
    python generate_summary_report.py --simple-csv factor_system/reports/batch_backtest_simple_*.csv --full-csv result_unified_summary.csv --output factor_system/reports/SUMMARY_REPORT.md
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

import pandas as pd
import numpy as np
from factor_system.factor_registry import get_registry


def parse_args():
    parser = argparse.ArgumentParser(description="生成批量回测汇总报告")
    parser.add_argument("--simple-csv", type=str, default=None,
                        help="简化回测结果 CSV 路径（支持通配符）")
    parser.add_argument("--full-csv", type=str, default="/root/cs_developer/result_unified_summary.csv",
                        help="完整回测汇总 CSV 路径")
    parser.add_argument("--output", type=str, default="/root/cs_developer/factor_system/reports/SUMMARY_REPORT.md",
                        help="输出 Markdown 报告路径")
    return parser.parse_args()


def load_simple_backtest(csv_path: str) -> pd.DataFrame:
    """加载简化回测结果"""
    if csv_path is None:
        # 自动查找最新的简化回测结果
        candidates = sorted(glob.glob("/root/cs_developer/factor_system/reports/batch_backtest_simple_*.csv"))
        if not candidates:
            return pd.DataFrame()
        csv_path = candidates[-1]
    
    df = pd.read_csv(csv_path)
    # 标记计算版本
    df["compute_version"] = df["factor_name"].apply(lambda x: "889" if x.endswith("_889") else "88")
    df["base_name"] = df["factor_name"].str.replace("_889", "", regex=False)
    # 处理 inf
    df["annual_return_clean"] = df["annual_return"].replace([np.inf, -np.inf], np.nan)
    df["sharpe_ratio_clean"] = df["sharpe_ratio"].replace([np.inf, -np.inf], np.nan)
    df["calmar_ratio_clean"] = df["calmar_ratio"].replace([np.inf, -np.inf], np.nan)
    return df


def load_full_backtest(csv_path: str) -> pd.DataFrame:
    """加载完整回测结果"""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["compute_version"] = df["factor"].apply(lambda x: "889" if str(x).endswith("_889") else "88")
    df["base_name"] = df["factor"].str.replace("_889", "", regex=False)
    return df


def generate_report(simple_df: pd.DataFrame, full_df: pd.DataFrame, output_path: str):
    """生成 Markdown 汇总报告"""
    
    registry = get_registry()
    lines = []
    
    # ================================================================
    # 标题
    # ================================================================
    lines.append("# 批量因子回测汇总报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n数据范围: 2020-01-01 ~ 2024-12-31")
    lines.append(f"品种数: 51 个商品期货")
    lines.append(f"总因子数: {len(registry.get_active())}")
    
    # ================================================================
    # 1. 因子设计总览
    # ================================================================
    lines.append("\n---\n")
    lines.append("\n## 1. 因子设计总览\n")
    
    overview = []
    for meta in sorted(registry.get_active(), key=lambda x: x.category):
        compute_data = "close_88 + close_88A2" if meta.category == "carry" else "close_88 / returns_88"
        if meta.name.endswith("_889"):
            compute_data = compute_data.replace("88", "889")
        direction = "做多低值 (IC-)" if meta.ic_direction == -1 else "做多高值 (IC+)"
        overview.append({
            "因子名": meta.name,
            "类别": meta.category,
            "因子计算数据": compute_data,
            "回测价格": meta.backtest_price_suffix,
            "交易方向": direction,
            "参数": str(meta.params),
        })
    
    overview_df = pd.DataFrame(overview)
    lines.append(overview_df.to_markdown(index=False))
    
    # ================================================================
    # 2. 简化回测结果汇总
    # ================================================================
    if not simple_df.empty:
        lines.append("\n---\n")
        lines.append("\n## 2. 简化回测结果（BatchBacktestEngine）\n")
        
        # 有效数据过滤（排除 inf）
        valid_simple = simple_df[simple_df["annual_return_clean"].notna() & simple_df["sharpe_ratio_clean"].notna()]
        invalid_simple = simple_df[simple_df["annual_return_clean"].isna() | simple_df["sharpe_ratio_clean"].isna()]
        
        lines.append(f"\n有效因子: {len(valid_simple)}/{len(simple_df)}")
        if len(invalid_simple) > 0:
            lines.append(f"\n**异常因子（计算出现 inf）**: {', '.join(invalid_simple['factor_name'].tolist())}")
            lines.append("\n> 异常原因: 888/889 数据中部分品种（I、LU、P）存在负价格，导致收益率计算出现 inf。")
        
        # Top 15 夏普
        lines.append("\n### 2.1 Top 15 因子（按夏普比率）\n")
        top_sharpe = valid_simple.nlargest(15, "sharpe_ratio_clean")[[
            "factor_name", "category", "annual_return_clean", "sharpe_ratio_clean", 
            "max_drawdown", "calmar_ratio_clean", "win_rate", "compute_version"
        ]]
        lines.append(top_sharpe.to_markdown(index=False))
        
        # Top 15 年化收益
        lines.append("\n### 2.2 Top 15 因子（按年化收益）\n")
        top_return = valid_simple.nlargest(15, "annual_return_clean")[[
            "factor_name", "category", "annual_return_clean", "sharpe_ratio_clean",
            "max_drawdown", "calmar_ratio_clean", "win_rate", "compute_version"
        ]]
        lines.append(top_return.to_markdown(index=False))
        
        # Top 15 Calmar
        lines.append("\n### 2.3 Top 15 因子（按 Calmar 比率）\n")
        top_calmar = valid_simple.nlargest(15, "calmar_ratio_clean")[[
            "factor_name", "category", "annual_return_clean", "sharpe_ratio_clean",
            "max_drawdown", "calmar_ratio_clean", "win_rate", "compute_version"
        ]]
        lines.append(top_calmar.to_markdown(index=False))
        
        # ================================================================
        # 3. 88 vs 889 版本对比
        # ================================================================
        lines.append("\n---\n")
        lines.append("\n## 3. 88 vs 889 计算版本对比\n")
        
        # 找出有双版本的因子
        paired = valid_simple.groupby("base_name").filter(lambda x: len(x) == 2 and set(x["compute_version"]) == {"88", "889"})
        if not paired.empty:
            comparison = []
            for base_name, group in paired.groupby("base_name"):
                row_88 = group[group["compute_version"] == "88"].iloc[0]
                row_889 = group[group["compute_version"] == "889"].iloc[0]
                comparison.append({
                    "因子": base_name,
                    "类别": row_88["category"],
                    "88_夏普": round(row_88["sharpe_ratio_clean"], 3),
                    "889_夏普": round(row_889["sharpe_ratio_clean"], 3),
                    "夏普差异": round(row_889["sharpe_ratio_clean"] - row_88["sharpe_ratio_clean"], 3),
                    "88_回撤": round(row_88["max_drawdown"], 3),
                    "889_回撤": round(row_889["max_drawdown"], 3),
                    "回撤差异": round(row_889["max_drawdown"] - row_88["max_drawdown"], 3),
                    "胜出": "889" if row_889["sharpe_ratio_clean"] > row_88["sharpe_ratio_clean"] else "88",
                })
            
            comp_df = pd.DataFrame(comparison)
            lines.append(f"\n双版本因子数: {len(comp_df)}")
            lines.append(f"\n889 胜出: {(comp_df['胜出'] == '889').sum()}")
            lines.append(f"\n88 胜出: {(comp_df['胜出'] == '88').sum()}")
            lines.append("\n### 3.1 889 显著优于 88 的因子（夏普差异 > 0.1）\n")
            better_889 = comp_df[comp_df["夏普差异"] > 0.1].sort_values("夏普差异", ascending=False)
            lines.append(better_889.to_markdown(index=False) if not better_889.empty else "\n无\n")
            
            lines.append("\n### 3.2 88 显著优于 889 的因子（夏普差异 < -0.1）\n")
            better_88 = comp_df[comp_df["夏普差异"] < -0.1].sort_values("夏普差异", ascending=True)
            lines.append(better_88.to_markdown(index=False) if not better_88.empty else "\n无\n")
            
            lines.append("\n### 3.3 全部双版本对比\n")
            lines.append(comp_df.sort_values("夏普差异", ascending=False).to_markdown(index=False))
    
    # ================================================================
    # 4. 完整回测结果
    # ================================================================
    if not full_df.empty:
        lines.append("\n---\n")
        lines.append("\n## 4. 完整回测结果（StrategyBacktester）\n")
        lines.append(f"\n完整回测因子数: {len(full_df)}")
        
        lines.append("\n### 4.1 Top 15 因子（按夏普比率）\n")
        if "sharpe_ratio" in full_df.columns:
            top_full = full_df.nlargest(15, "sharpe_ratio")[[
                "factor", "annual_return", "sharpe_ratio", "max_drawdown", "calmar_ratio"
            ]]
            lines.append(top_full.to_markdown(index=False))
        
        # ================================================================
        # 5. 简化 vs 完整回测对比
        # ================================================================
        if not simple_df.empty and not full_df.empty:
            lines.append("\n---\n")
            lines.append("\n## 5. 简化回测 vs 完整回测对比\n")
            
            merged = pd.merge(
                valid_simple[["factor_name", "annual_return_clean", "sharpe_ratio_clean", "max_drawdown"]],
                full_df[["factor", "annual_return", "sharpe_ratio", "max_drawdown"]],
                left_on="factor_name", right_on="factor", how="inner"
            )
            
            if not merged.empty:
                merged["年化收益偏差"] = merged["annual_return"] - merged["annual_return_clean"]
                merged["夏普偏差"] = merged["sharpe_ratio"] - merged["sharpe_ratio_clean"]
                
                lines.append(f"\n可对齐因子数: {len(merged)}")
                lines.append("\n### 5.1 偏差最大的因子（简化回测高估）\n")
                overestimate = merged.nlargest(10, "年化收益偏差")[[
                    "factor_name", "annual_return_clean", "annual_return", "年化收益偏差", "夏普偏差"
                ]]
                lines.append(overestimate.to_markdown(index=False))
                
                lines.append("\n### 5.2 偏差最小的因子（两者最接近）\n")
                closest = merged.nsmallest(10, "年化收益偏差").sort_values("年化收益偏差", ascending=True)[[
                    "factor_name", "annual_return_clean", "annual_return", "年化收益偏差", "夏普偏差"
                ]]
                lines.append(closest.to_markdown(index=False))
    
    # ================================================================
    # 6. 数据质量问题
    # ================================================================
    lines.append("\n---\n")
    lines.append("\n## 6. 数据质量问题\n")
    lines.append("\n### 6.1 888/889 负价格品种\n")
    lines.append("""
| 品种 | 888 最小价格 | 889 最小价格 | 影响 |
|------|-------------|-------------|------|
| I88.DCE (铁矿石) | -393.50 | 正常 | 收益率 inf |
| LU88.INE (低硫燃料油) | -683.00 | 正常 | 收益率 inf |
| P88.DCE (棕榈油) | -2618.00 | 正常 | 收益率 inf |
| FU88.SHFE (燃料油) | 正常 | -718.00 | 收益率 inf |
| CJ88.CZCE (红枣) | 正常 | -181.00 | 收益率 inf |
| RU88.SHFE (橡胶) | 正常 | 全负 | 收益率 inf |

> **根因**: 米筐复权算法在极端行情下产生负价格，导致 `pct_change()` 出现 inf。
> 
> **修复方案**: 在 `BatchBacktestEngine._load_price_data()` 和 `FactorEngine._build_derived_data()` 中增加负价格过滤逻辑，剔除当日收盘价 <= 0 的品种。
""")
    
    # ================================================================
    # 7. 结论与建议
    # ================================================================
    lines.append("\n---\n")
    lines.append("\n## 7. 结论与建议\n")
    
    if not simple_df.empty:
        best = valid_simple.nlargest(1, "sharpe_ratio_clean").iloc[0]
        lines.append(f"""
### 7.1 最优单因子

| 指标 | 值 |
|------|-----|
| 因子名 | {best['factor_name']} |
| 类别 | {best['category']} |
| 计算数据 | {'889' if best['factor_name'].endswith('_889') else '88'} |
| 年化收益 | {best['annual_return_clean']:.2%} |
| 夏普比率 | {best['sharpe_ratio_clean']:.3f} |
| 最大回撤 | {best['max_drawdown']:.2%} |
| Calmar | {best['calmar_ratio_clean']:.3f} |

**原因分析**:
- {best['factor_name']} 基于{'889' if best['factor_name'].endswith('_889') else '88'}数据计算，{'消除了换月跳空' if best['factor_name'].endswith('_889') else '保留了原始价格信息'}
- 该因子捕捉了商品期货截面{'收益率分布不对称性' if 'skew' in best['factor_name'] else '期限结构' if 'carry' in best['factor_name'] else '动量效应' if 'momentum' in best['factor_name'] else '波动率特征'}
- 在 2020-2024 年期间，商品市场{'波动较大' if best['category'] in ['volatility', 'skewness', 'statistical'] else '趋势明显' if best['category'] == 'momentum' else '期限结构稳定'}
""")
    
    lines.append("""
### 7.2 最优数据基础选择

| 因子类别 | 推荐计算数据 | 推荐回测价格 | 原因 |
|---------|-------------|-------------|------|
| carry | 88 + 88A2 | 888 | 需要真实价差，888 复权消除换月跳空 |
| momentum | 889 | 888 | 889 计算更平滑，888 回测避免跳空失真 |
| reversal | 889 | 888 | 同上 |
| volatility | 889 | 889 | 统计类因子对价格连续性要求高 |
| skewness | 889 | 889 | 889 版本回撤更小，夏普更高 |
| statistical | 889 | 889 | 峰度/偏度在 889 上表现最优 |
| technical | 88/889 | 888 | 技术指标对价格水平敏感，88 更真实 |

### 7.3 下一步行动

1. **修复负价格过滤**: 在数据加载层增加 `close > 0` 校验
2. **重新跑简化回测**: 排除异常因子后获取准确结果
3. **因子组合**: 对 Top 5 不相关因子做等权组合，测试分散化效果
4. **参数优化**: 对 skew_180、carry_ret 等核心因子做持仓周期和交易比例网格搜索
""")
    
    # 写入文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"[OK] 汇总报告已生成: {output_path}")
    return output_path


def main():
    args = parse_args()
    
    simple_df = load_simple_backtest(args.simple_csv)
    full_df = load_full_backtest(args.full_csv)
    
    report_path = generate_report(simple_df, full_df, args.output)
    print(f"\n报告路径: {report_path}")
    print(f"  - 简化回测数据: {'已加载' if not simple_df.empty else '未找到'}")
    print(f"  - 完整回测数据: {'已加载' if not full_df.empty else '未找到'}")


if __name__ == "__main__":
    main()
