"""
Skew Factor（收益率偏度因子）

核心逻辑：
- 对每个商品品种，计算过去 lookback 天日收益率的偏度（skewness）
- Skew > 0：右偏，存在较多正向极端收益（右侧肥尾）
- Skew < 0：左偏，存在较多负向极端收益（左侧肥尾）
- 标准做法：做多负偏度（低风险补偿后被低估），做空正偏度（高乐观情绪被高估）

作者：cross_sectional
"""

from datetime import datetime
from typing import List, Dict

from pandas import DataFrame, Series

from .factor_template import FactorTemplate


class SkewFactor(FactorTemplate):
    """收益率偏度因子"""

    start: datetime = None                  # 开始时间
    end: datetime = None                    # 结束时间
    dominant_symbols: List[str] = None      # 主力合约表键（如 RB88.SHFE）
    lookback: int = 180                     # 历史回看周期（默认180个交易日≈9个月）

    def __init__(self, vt_symbols: List[str], setting: dict) -> None:
        """"""
        super().__init__(vt_symbols, setting)

    def calculate_factor(self, df: DataFrame) -> dict:
        """逐日回放计算 Skew 因子（用于生产环境策略回测）"""
        factor_data: Dict[str, float] = {}

        for dominant_symbol in self.dominant_symbols:
            try:
                close_series: Series = df[dominant_symbol]["close_price"]
            except (KeyError, TypeError):
                factor_data[dominant_symbol] = float("nan")
                continue

            if len(close_series) < self.lookback + 1:
                factor_data[dominant_symbol] = float("nan")
                continue

            recent_close: Series = close_series.iloc[-(self.lookback + 1):]
            returns: Series = recent_close.pct_change().dropna()

            if len(returns) >= 3:
                skew_value: float = returns.skew()
                factor_data[dominant_symbol] = skew_value
            else:
                factor_data[dominant_symbol] = float("nan")

        return factor_data

    @classmethod
    def compute_vectorized(cls, price_df: DataFrame, lookback: int = 180) -> DataFrame:
        """
        向量化批量计算 Skew 因子（用于参数搜索）
        
        参数:
            price_df: DataFrame, index=datetime, columns=dominant_symbols, values=close_price
            lookback: 收益率回看天数
        
        返回:
            factor_df: DataFrame, index=datetime, columns=dominant_symbols, values=skew_value
        """
        returns = price_df.pct_change()
        factor_df = returns.rolling(window=lookback, min_periods=lookback).skew()
        return factor_df
