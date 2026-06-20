import numpy as np
import pandas as pd
from typing import Dict

def compute(engine, cycle=5, lookback_short=5, lookback_long=20, **kwargs) -> pd.DataFrame:
    """
    Nelson-Siegel Slope Factor - Momentum Version
    基于期限结构斜率的变化率（动量）而非绝对水平
    
    核心洞察：
    - 中国市场888 vs 889价差长期稳定但非平稳（ADF p=0.618）
    - 价差绝对水平不适合作为信号（有长期趋势）
    - 价差的短期变化（动量）更适合预测未来收益
    
    因子构建：
    1. 计算888 vs 889的价差（期限结构斜率代理）
    2. 计算价差的短期变化率（斜率动量）
    3. 结合斜率动量和价格动量构建综合因子
    
    论文参考：Bianchi et al. (2023) - Slope Factor Strategy
    改进：使用动量而非绝对水平，适应中国市场非平稳特性
    """
    # 兼容两种调用方式：engine对象或DataCache对象
    if hasattr(engine, 'cache'):
        cache = engine.cache
    else:
        cache = engine
    
    # 获取价格数据
    # 兼容两种调用方式：engine对象或DataCache对象
    if hasattr(engine, '_f2_close'):
        # 直接传入engine对象
        f1_close = engine._close  # 888合约（主力）
        f2_close = engine._f2_close  # 889合约（次主力）
    else:
        # 传入cache对象
        f1_close = engine._ohlcv_wide.get('close_price') if engine._ohlcv_wide else None
        # 次主力数据需要从其他方式获取
        f2_close = getattr(engine, '_f2_close', None)
    
    # 计算期限结构斜率（888 vs 889价差百分比）
    spread_pct = (f1_close - f2_close) / f2_close * 100
    
    # 计算斜率的短期和长期变化（动量）
    spread_mom_short = spread_pct.diff(lookback_short)
    spread_mom_long = spread_pct.diff(lookback_long)
    
    # 计算斜率的加速度（二阶导数）
    # 当斜率在加速变陡或变平时，信号更强
    spread_accel = spread_mom_short - spread_mom_short.shift(lookback_short)
    
    # 计算价格的短期动量（作为辅助信号）
    price_mom = f1_close.pct_change(lookback_short)
    
    # 综合因子构建
    # 逻辑：
    # - 斜率在快速变陡（spread_mom_short很大正）→ 近月相对走强 → 后续可能回调 → 做空高因子值
    # - 斜率在快速变平（spread_mom_short很大负）→ 近月相对走弱 → 后续可能反弹 → 做多低因子值
    
    # 对各个组件进行截面排名（0-1）
    spread_mom_short_rank = spread_mom_short.rank(axis=1, pct=True)
    spread_mom_long_rank = spread_mom_long.rank(axis=1, pct=True)
    spread_accel_rank = spread_accel.rank(axis=1, pct=True)
    price_mom_rank = price_mom.rank(axis=1, pct=True)
    
    # 综合因子（加权组合）
    # 短期斜率动量权重最高（最敏感）
    # 加速度作为确认信号
    # 价格动量作为辅助（避免逆势）
    factor = (
        0.50 * spread_mom_short_rank +
        0.20 * spread_mom_long_rank +
        0.20 * spread_accel_rank +
        0.10 * price_mom_rank
    )
    
    # 处理缺失值
    factor = factor.fillna(method='ffill', limit=5)
    
    # 再次排名确保分布均匀
    factor = factor.rank(axis=1, pct=True)
    
    # 最终填充
    factor = factor.fillna(0.5)
    
    # 确保输出格式与输入一致
    factor = factor.loc[f1_close.index]
    
    print(f"[NS-Momentum] 计算完成: cycle={cycle}, lookback_short={lookback_short}, lookback_long={lookback_long}")
    print(f"[NS-Momentum] 因子组件: 斜率短期动量(50%) + 斜率长期动量(20%) + 斜率加速度(20%) + 价格动量(10%)")
    
    return factor

# 别名，兼容引擎查找
calc_ns_slope_momentum = compute
