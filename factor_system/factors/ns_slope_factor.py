#!/usr/bin/env python3
"""
Nelson-Siegel Slope Factor — 商品期货期限结构斜率因子

基于论文 "Exploiting the dynamics of commodity futures curves" (Bianchi et al., 2023)

核心逻辑：
1. 用 Nelson-Siegel 模型拟合每个品种的期限结构（需要近月、次近月、远月等多个合约）
2. 提取斜率因子 beta_2（slope）
3. 信号: Δbeta_2 > 0 → 做多斜率价差（买近月 + 卖远月）
    Δbeta_2 < 0 → 做空斜率价差（卖近月 + 买远月）

在中国期货市场复现的简化方案：
- 使用 88（主力连续）和 88A2（次主力连续）作为期限结构的两个端点
- 用这两个合约的价格关系近似斜率变化
- 因子值 = 斜率变化的方向和强度

因子定义：
- slope = (F1 - F2) / F1  的时序变化率
- 其中 F1 = 主力连续(88), F2 = 次主力连续(88A2)
- 因子值 = slope_t - slope_{t-1}  的滚动平均

交易方向（ic_direction=+1）:
- 因子值越高 → 斜率变陡（更backwardation）→ 做多斜率价差
- 因子值越低 → 斜率变平（更contango）→ 做空斜率价差

注意：
- 严格使用 88/88A2 计算因子值（AGENTS.md 规则1）
- 交易方向由 factor_registry 中的 ic_direction 确定（AGENTS.md 规则2）
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factor_engine import FactorEngine


def safe_div(a, b, fill=0.0):
    """安全除法"""
    result = np.where(np.abs(b) > 1e-12, a / b, fill)
    if hasattr(b, 'index') and hasattr(b, 'columns'):
        return pd.DataFrame(result, index=b.index, columns=b.columns)
    if hasattr(a, 'index') and hasattr(a, 'columns'):
        return pd.DataFrame(result, index=a.index, columns=a.columns)
    return result


def calc_ns_slope(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """
    Nelson-Siegel 斜率因子
    
    参数:
        engine: FactorEngine 实例
        cycle: 斜率变化的平滑周期（默认5天）
    
    返回:
        DataFrame: index=datetime, columns=dominant_symbols
        因子值 = slope变化率的滚动平均
    """
    f1 = engine.close  # 主力连续 88
    f2 = engine.f2_close  # 次主力连续 88A2
    
    if f2 is None or f2.empty:
        # 如果没有次主力数据，返回空DataFrame
        return pd.DataFrame(index=f1.index, columns=f1.columns)
    
    # 对齐列：只保留共同品种
    common_cols = f1.columns.intersection(f2.columns)
    f1 = f1[common_cols]
    f2 = f2[common_cols]
    
    # 计算斜率 = (F1 - F2) / F1
    # > 0: backwardation (主力 > 次主力，远期贴水)
    # < 0: contango (主力 < 次主力，远期升水)
    slope = safe_div(f1 - f2, f1, fill=0.0)
    
    # 计算斜率的变化率（一阶差分）
    slope_change = slope.diff()
    
    # 滚动平滑（减少噪声）
    factor = slope_change.rolling(window=cycle, min_periods=cycle).mean()
    
    # 扩展回原始列
    result = pd.DataFrame(index=engine.close.index, columns=engine.close.columns)
    result[common_cols] = factor
    
    return result


def calc_ns_slope_zscore(engine: "FactorEngine", cycle: int = 20, smooth: int = 5) -> pd.DataFrame:
    """
    Nelson-Siegel 斜率因子的Z-Score版本（更稳健）
    
    参数:
        engine: FactorEngine 实例
        cycle: Z-Score计算的回看周期（默认20天）
        smooth: 斜率变化的平滑周期（默认5天）
    
    返回:
        DataFrame: 斜率变化的Z-Score
    """
    f1 = engine.close
    f2 = engine.f2_close
    
    if f2 is None or f2.empty:
        return pd.DataFrame(index=f1.index, columns=f1.columns)
    
    common_cols = f1.columns.intersection(f2.columns)
    f1 = f1[common_cols]
    f2 = f2[common_cols]
    
    # 计算斜率
    slope = safe_div(f1 - f2, f1, fill=0.0)
    
    # 斜率变化
    slope_change = slope.diff()
    
    # 滚动Z-Score标准化
    mean = slope_change.rolling(window=cycle, min_periods=cycle).mean()
    std = slope_change.rolling(window=cycle, min_periods=cycle).std()
    
    # 避免除零
    zscore = safe_div(slope_change - mean, std, fill=0.0)
    
    # 再平滑一层
    factor = zscore.rolling(window=smooth, min_periods=smooth).mean()
    
    # 扩展回原始列
    result = pd.DataFrame(index=engine.close.index, columns=engine.close.columns)
    result[common_cols] = factor
    
    return result


def calc_ns_level(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """
    Nelson-Siegel 水平因子（Level）
    
    近似实现：用主力连续价格的变化作为水平因子
    论文中 L 策略不盈利，这里作为对比
    
    参数:
        engine: FactorEngine 实例
        cycle: 平滑周期
    
    返回:
        DataFrame: 水平变化率
    """
    f1 = engine.close
    
    # 水平 = 主力连续价格的时序变化
    level_change = f1.pct_change()
    
    # 滚动平滑
    factor = level_change.rolling(window=cycle, min_periods=cycle).mean()
    
    return factor


def calc_ns_curvature(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """
    Nelson-Siegel 曲率因子（Curvature）
    
    简化实现：用斜率变化的二阶差分近似曲率变化
    需要至少3个合约点才能计算真正的曲率
    这里用 88/88A2 只能近似
    
    参数:
        engine: FactorEngine 实例
        cycle: 平滑周期
    
    返回:
        DataFrame: 曲率变化率
    """
    f1 = engine.close
    f2 = engine.f2_close
    
    if f2 is None or f2.empty:
        return pd.DataFrame(index=f1.index, columns=f1.columns)
    
    common_cols = f1.columns.intersection(f2.columns)
    f1 = f1[common_cols]
    f2 = f2[common_cols]
    
    # 斜率
    slope = safe_div(f1 - f2, f1, fill=0.0)
    
    # 曲率近似 = 斜率变化的加速度（二阶差分）
    curvature = slope.diff().diff()
    
    # 滚动平滑
    factor = curvature.rolling(window=cycle, min_periods=cycle).mean()
    
    # 扩展回原始列
    result = pd.DataFrame(index=engine.close.index, columns=engine.close.columns)
    result[common_cols] = factor
    
    return result
