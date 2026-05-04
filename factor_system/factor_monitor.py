#!/usr/bin/env python3
"""
Factor Monitor — 因子监控分析模块

职责：
1. IC 分析：计算因子与未来收益的秩相关系数
2. IC 衰减：不同持有期下的 IC 表现
3. 分组收益：按因子分位数的收益分析
4. 因子相关性矩阵：检测多重共线性
5. 定期报告生成：HTML/Markdown 格式
6. 因子衰减预警：IC 均值、IR 的滚动监控

使用示例：
    from factor_monitor import FactorMonitor
    monitor = FactorMonitor(engine)
    
    # IC 分析
    ic_results = monitor.analyze_ic(factor_df, forward_returns)
    
    # 批量分析所有因子
    report = monitor.batch_analyze(factor_dict, forward_periods=[1, 5, 10, 20])
    
    # 生成报告
    monitor.generate_report(report, output_path="/path/to/report.md")
"""

import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import traceback

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# IC 计算核心
# =============================================================================

def calc_ic_series(factor_df: pd.DataFrame, forward_return_df: pd.DataFrame) -> pd.Series:
    """
    计算日度 IC（Spearman 秩相关系数）
    
    参数:
        factor_df: index=datetime, columns=symbols
        forward_return_df: 同结构，未来收益
    
    返回:
        pd.Series: 日度 IC 序列
    """
    ic_values = []
    index_list = []
    
    common_index = factor_df.index.intersection(forward_return_df.index)
    
    for dt in common_index:
        f = factor_df.loc[dt]
        r = forward_return_df.loc[dt]
        
        valid = f.notna() & r.notna()
        if valid.sum() < 5:
            continue
        
        corr, _ = stats.spearmanr(f[valid], r[valid])
        if not np.isnan(corr):
            ic_values.append(corr)
            index_list.append(dt)
    
    return pd.Series(ic_values, index=index_list)


def calc_ic_stats(ic_series: pd.Series) -> dict:
    """计算 IC 统计量"""
    ic_clean = ic_series.dropna()
    if len(ic_clean) == 0:
        return {}
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    ir = mean_ic / std_ic if std_ic > 0 else 0
    ic_pos_ratio = (ic_clean > 0).sum() / len(ic_clean)
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_clean))) if std_ic > 0 else 0
    ir_annual = mean_ic * np.sqrt(252) / std_ic if std_ic > 0 else 0
    
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "ir_annual": ir_annual,
        "ic_pos_ratio": ic_pos_ratio,
        "t_stat": t_stat,
        "ic_count": len(ic_clean),
        "max_ic": ic_clean.max(),
        "min_ic": ic_clean.min(),
        "ic_skew": ic_clean.skew(),
        "ic_kurt": ic_clean.kurt(),
    }


def calc_quantile_returns(
    factor_df: pd.DataFrame,
    forward_return_df: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    分组收益分析
    
    返回:
        DataFrame: index=datetime, columns=[Q1, Q2, ..., Qn]
    """
    daily_results = []
    common_index = factor_df.index.intersection(forward_return_df.index)
    
    for dt in common_index:
        f = factor_df.loc[dt]
        r = forward_return_df.loc[dt]
        
        valid = f.notna() & r.notna()
        if valid.sum() < n_quantiles * 3:
            continue
        
        f_valid = f[valid]
        r_valid = r[valid]
        
        try:
            labels = [f"Q{i+1}" for i in range(n_quantiles)]
            q_labels = pd.qcut(f_valid, n_quantiles, labels=labels, duplicates='drop')
            if q_labels.isna().sum() > 0:
                continue
            group_ret = r_valid.groupby(q_labels).mean()
            daily_results.append(group_ret)
        except Exception:
            continue
    
    if not daily_results:
        return pd.DataFrame()
    
    quantile_df = pd.DataFrame(daily_results)
    return quantile_df


# =============================================================================
# FactorMonitor 类
# =============================================================================

class FactorMonitor:
    """因子监控分析器"""
    
    def __init__(
        self,
        close_df: Optional[pd.DataFrame] = None,
        forward_periods: List[int] = None,
        output_dir: str = "/root/cs_developer/factor_system/reports"
    ):
        self.close_df = close_df
        self.forward_periods = forward_periods or [1, 5, 10, 20]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 预计算 forward returns
        self._forward_returns: Dict[int, pd.DataFrame] = {}
        if close_df is not None:
            self._precompute_forward_returns()
    
    def _precompute_forward_returns(self):
        """预计算多期 forward returns"""
        if self.close_df is None or self.close_df.empty:
            return
        
        returns = self.close_df.pct_change()
        for p in self.forward_periods:
            # 未来收益 = (close_{t+p} - close_t) / close_t
            fwd = self.close_df.shift(-p) / self.close_df - 1.0
            self._forward_returns[p] = fwd
    
    def set_close_df(self, close_df: pd.DataFrame):
        """设置收盘价数据（用于计算 forward returns）"""
        self.close_df = close_df
        self._forward_returns.clear()
        self._precompute_forward_returns()
    
    # ------------------------------------------------------------------
    # 单因子分析
    # ------------------------------------------------------------------
    
    def analyze_single(
        self,
        factor_df: pd.DataFrame,
        factor_name: str = "unknown"
    ) -> Dict[int, dict]:
        """
        对单个因子做完整分析
        
        返回:
            {period: {
                "ic_series": pd.Series,
                "ic_stats": dict,
                "quantile_returns": pd.DataFrame,
                "quantile_mean": pd.Series
            }}
        """
        results = {}
        
        for period in self.forward_periods:
            if period not in self._forward_returns:
                continue
            
            fwd = self._forward_returns[period]
            
            # IC
            ic_series = calc_ic_series(factor_df, fwd)
            ic_stats = calc_ic_stats(ic_series)
            
            # 分组收益
            q_df = calc_quantile_returns(factor_df, fwd, n_quantiles=5)
            q_mean = q_df.mean() if not q_df.empty else pd.Series()
            
            results[period] = {
                "ic_series": ic_series,
                "ic_stats": ic_stats,
                "quantile_returns": q_df,
                "quantile_mean": q_mean,
            }
        
        return results
    
    # ------------------------------------------------------------------
    # 批量分析
    # ------------------------------------------------------------------
    
    def batch_analyze(
        self,
        factor_dict: Dict[str, pd.DataFrame],
        factor_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, dict]]:
        """
        批量分析多个因子
        
        返回:
            {factor_name: {period: result_dict}}
        """
        names = factor_names or list(factor_dict.keys())
        all_results = {}
        
        print("\n" + "=" * 70)
        print("Factor Monitor — 批量 IC 分析")
        print("=" * 70)
        
        for idx, name in enumerate(names, 1):
            df = factor_dict.get(name)
            if df is None or df.empty:
                print(f"[{idx}/{len(names)}] {name} — 数据为空，跳过")
                continue
            
            print(f"[{idx}/{len(names)}] {name} ...", end=" ")
            try:
                result = self.analyze_single(df, name)
                all_results[name] = result
                
                # 打印核心指标（5D持有期）
                if 5 in result and result[5]["ic_stats"]:
                    stats = result[5]["ic_stats"]
                    print(f"IC={stats['mean_ic']:+.3f}, IR={stats['ir']:+.3f}, t={stats['t_stat']:+.2f}")
                else:
                    print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
        
        return all_results
    
    # ------------------------------------------------------------------
    # 相关性分析
    # ------------------------------------------------------------------
    
    def correlation_matrix(
        self,
        factor_dict: Dict[str, pd.DataFrame],
        method: str = "spearman"
    ) -> pd.DataFrame:
        """计算因子截面相关性矩阵"""
        # 对齐所有因子到共同日期
        common_index = None
        aligned = {}
        
        for name, df in factor_dict.items():
            if df is None or df.empty:
                continue
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()
        
        # 每天截面取因子均值，构建时间序列
        ts_data = {}
        for name, df in factor_dict.items():
            df_aligned = df.loc[common_index]
            ts_data[name] = df_aligned.mean(axis=1)
        
        ts_df = pd.DataFrame(ts_data)
        return ts_df.corr(method=method)
    
    # ------------------------------------------------------------------
    # 图表生成
    # ------------------------------------------------------------------
    
    def plot_ic_timeseries(
        self,
        results: Dict[str, Dict[int, dict]],
        period: int = 5,
        output_path: Optional[str] = None
    ) -> str:
        """绘制 IC 时间序列图"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        ax1 = axes[0]
        for name, res in results.items():
            if period not in res:
                continue
            ic_series = res[period]["ic_series"]
            if len(ic_series) > 0:
                ax1.plot(ic_series.index, ic_series.values, label=name, alpha=0.7, linewidth=0.8)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Daily IC')
        ax1.set_title(f'Daily IC Time Series ({period}D Forward)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        for name, res in results.items():
            if period not in res:
                continue
            ic_series = res[period]["ic_series"]
            if len(ic_series) > 0:
                cum_ic = ic_series.fillna(0).cumsum()
                ax2.plot(cum_ic.index, cum_ic.values, label=name, linewidth=1.0)
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_ylabel('Cumulative IC')
        ax2.set_xlabel('Date')
        ax2.set_title('Cumulative IC')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = f"{self.output_dir}/ic_timeseries_{period}d.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    
    def plot_ic_heatmap(
        self,
        results: Dict[str, Dict[int, dict]],
        output_path: Optional[str] = None
    ) -> str:
        """绘制 IC 统计量热力图"""
        # 构建汇总表
        rows = []
        for name, res in results.items():
            for period in sorted(res.keys()):
                stats = res[period].get("ic_stats", {})
                if stats:
                    rows.append({
                        "factor": name,
                        "period": f"{period}D",
                        "mean_ic": stats.get("mean_ic", np.nan),
                        "ir": stats.get("ir", np.nan),
                        "t_stat": stats.get("t_stat", np.nan),
                        "ic_pos": stats.get("ic_pos_ratio", np.nan),
                    })
        
        if not rows:
            return ""
        
        df = pd.DataFrame(rows)
        pivot_ic = df.pivot(index="factor", columns="period", values="mean_ic")
        pivot_ir = df.pivot(index="factor", columns="period", values="ir")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(pivot_ic) * 0.4)))
        
        # Mean IC
        im1 = axes[0].imshow(pivot_ic.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        axes[0].set_xticks(range(len(pivot_ic.columns)))
        axes[0].set_xticklabels(pivot_ic.columns)
        axes[0].set_yticks(range(len(pivot_ic.index)))
        axes[0].set_yticklabels(pivot_ic.index, fontsize=8)
        axes[0].set_title('Mean IC')
        for i in range(len(pivot_ic.index)):
            for j in range(len(pivot_ic.columns)):
                val = pivot_ic.iloc[i, j]
                if not np.isnan(val):
                    axes[0].text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=7)
        plt.colorbar(im1, ax=axes[0])
        
        # IR
        im2 = axes[1].imshow(pivot_ir.values, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
        axes[1].set_xticks(range(len(pivot_ir.columns)))
        axes[1].set_xticklabels(pivot_ir.columns)
        axes[1].set_yticks(range(len(pivot_ir.index)))
        axes[1].set_yticklabels(pivot_ir.index, fontsize=8)
        axes[1].set_title('IR')
        for i in range(len(pivot_ir.index)):
            for j in range(len(pivot_ir.columns)):
                val = pivot_ir.iloc[i, j]
                if not np.isnan(val):
                    axes[1].text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=7)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = f"{self.output_dir}/ic_heatmap.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    
    def plot_correlation_matrix(
        self,
        corr_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """绘制因子相关性矩阵图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_yticklabels(corr_df.index, fontsize=8)
        ax.set_title('Factor Cross-Sectional Correlation Matrix (Spearman)')
        
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                val = corr_df.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6,
                        color='white' if abs(val) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if output_path is None:
            output_path = f"{self.output_dir}/factor_correlation.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    
    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------
    
    def generate_summary_table(
        self,
        results: Dict[str, Dict[int, dict]],
        period: int = 5
    ) -> pd.DataFrame:
        """生成因子汇总表"""
        rows = []
        for name, res in results.items():
            if period not in res:
                continue
            stats = res[period].get("ic_stats", {})
            q_mean = res[period].get("quantile_mean", pd.Series())
            
            if not stats:
                continue
            
            row = {
                "factor": name,
                "mean_ic": stats.get("mean_ic", np.nan),
                "std_ic": stats.get("std_ic", np.nan),
                "ir": stats.get("ir", np.nan),
                "t_stat": stats.get("t_stat", np.nan),
                "ic_pos": stats.get("ic_pos_ratio", np.nan),
                "ic_count": stats.get("ic_count", 0),
                "q1_ret": q_mean.get("Q1", np.nan) if not q_mean.empty else np.nan,
                "q5_ret": q_mean.get("Q5", np.nan) if not q_mean.empty else np.nan,
                "q5_q1": (q_mean.get("Q5", np.nan) - q_mean.get("Q1", np.nan)) if not q_mean.empty else np.nan,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ir", key=abs, ascending=False)
        return df
    
    def generate_markdown_report(
        self,
        results: Dict[str, Dict[int, dict]],
        corr_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
    ) -> str:
        """生成 Markdown 格式的因子分析报告"""
        lines = []
        lines.append("# Factor Analysis Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Factors Analyzed: {len(results)}")
        lines.append(f"Forward Periods: {self.forward_periods}")
        lines.append("")
        
        # IC 汇总表（5D）
        lines.append("## IC Summary (5D Forward Return)")
        lines.append("")
        summary = self.generate_summary_table(results, period=5)
        if not summary.empty:
            lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        
        # 各持有期详细表
        for period in self.forward_periods:
            lines.append(f"## Detailed Stats ({period}D Forward)")
            lines.append("")
            sub = self.generate_summary_table(results, period=period)
            if not sub.empty:
                lines.append(sub.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
        
        # 相关性
        if corr_df is not None and not corr_df.empty:
            lines.append("## Factor Correlation Matrix")
            lines.append("")
            lines.append(corr_df.to_markdown(floatfmt=".3f"))
            lines.append("")
        
        # Top/Bottom 因子
        if not summary.empty:
            lines.append("## Top 5 Factors by IR (Absolute)")
            lines.append("")
            top5 = summary.head(5)[["factor", "mean_ic", "ir", "t_stat", "q5_q1"]]
            lines.append(top5.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
            
            lines.append("## Bottom 5 Factors by IR (Absolute)")
            lines.append("")
            bottom5 = summary.tail(5)[["factor", "mean_ic", "ir", "t_stat", "q5_q1"]]
            lines.append(bottom5.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_path is None:
            output_path = f"{self.output_dir}/factor_report.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return output_path
