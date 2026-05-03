"""
本地扩展的因子模板
在 vnpy_alpharesearch.factor.FactorTemplate 基础上新增 compute_vectorized 接口，
用于高效批量参数搜索，不破坏现有 calculate_factor 逐日回放机制。
"""
from typing import List

from pandas import DataFrame

from vnpy_alpharesearch.factor import FactorTemplate as BaseFactorTemplate


class FactorTemplate(BaseFactorTemplate):
    """扩展因子模板：支持向量化批量计算"""

    @classmethod
    def compute_vectorized(cls, **kwargs) -> DataFrame:
        """
        向量化批量计算因子值（用于参数搜索）
        
        参数:
            由各子类定义，通常为 wide DataFrame + 参数
        
        返回:
            DataFrame: 因子值，index=datetime, columns=dominant_symbols
        """
        raise NotImplementedError(
            f"{cls.__name__} 未实现 compute_vectorized，"
            f"请使用 calculate_factor 逐日回放方式"
        )
