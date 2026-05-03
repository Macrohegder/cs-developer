from datetime import datetime
from typing import List, Dict

from pandas import DataFrame, Series, isna as pd_isna

from vnpy_alpharesearch.factor import FactorTemplate
from vnpy_alpharesearch import DataCenter


class SpreadZScoreFactor(FactorTemplate):
    """
    期限结构价差 Z-Score 因子

    核心逻辑（严格按文章定义）：
    - F1 = 主力连续合约 (88)
    - F2 = 次主力连续合约 (88A2)
    - 每日价差 spread = F1_close - F2_close
    - 因子 = 当前价差偏离历史均值的程度（Z-Score）
    - Z-Score = (current_spread - mean(spread, lookback)) / std(spread, lookback)
    - 基于均值回归假设：Z-Score 越高，下周越倾向反转（F1 相对 F2 下跌）
    """

    start: datetime = None                  # 开始时间
    end: datetime = None                    # 结束时间
    dominant_symbols: List[str] = None      # 主力合约表键（如 RB88.SHFE）
    lookback: int = 20                      # 价差历史回看周期（默认20个交易日≈1个月）

    def __init__(self, vt_symbols: List[str], setting: dict) -> None:
        """"""
        super().__init__(vt_symbols, setting)

        self.dc: DataCenter = DataCenter()

        # 初始化主力合约映射关系的查询主键
        keys: List[str] = []
        for s in self.dominant_symbols:
            keys.append(f"{s}@1")  # 主力
            keys.append(f"{s}@2")  # 次主力

        # 加载合约映射数据
        self.dc.load_reference_df(
            "vnpy_dominant_contract",
            keys,
            self.start,
            self.end,
            fillna=True
        )

        # 加载合约信息数据
        self.contract_df: DataFrame = self.dc.load_contract_df()

    def calculate_factor(self, df: DataFrame) -> dict:
        """计算因子数值"""
        dt: datetime = df.index[-1]
        self.dc.update_datetime(dt)

        # 获取当前时点的主力合约信息
        dominant_df: DataFrame = self.dc.get_data_history("vnpy_dominant_contract", dt)
        if len(dominant_df) == 0:
            return {s: float('nan') for s in self.dominant_symbols}
        dominant_series: Series = dominant_df.iloc[-1, :]

        # 计算价差 Z-Score
        factor_data: Dict[str, float] = {}

        for dominant_symbol in self.dominant_symbols:
            # 提取主力和次主力合约的收盘价序列
            try:
                # 主力连续合约 (88) 收盘价序列
                f1_close_series: Series = df[dominant_symbol]["close_price"]
                # 次主力连续合约 (88A2) 收盘价序列
                f2_symbol: str = dominant_symbol.replace("88.", "88A2.")
                f2_close_series: Series = df[f2_symbol]["close_price"]
            except (KeyError, IndexError):
                factor_data[dominant_symbol] = float('nan')
                continue

            # 检查数据长度是否足够计算 Z-Score
            if len(f1_close_series) < self.lookback or len(f2_close_series) < self.lookback:
                factor_data[dominant_symbol] = float('nan')
                continue

            # 计算价差序列（最近 lookback 天）
            f1_recent: Series = f1_close_series.iloc[-self.lookback:]
            f2_recent: Series = f2_close_series.iloc[-self.lookback:]
            spread_series: Series = f1_recent - f2_recent

            # 当前价差
            current_spread: float = spread_series.iloc[-1]

            # 历史均值和标准差
            spread_mean: float = spread_series.mean()
            spread_std: float = spread_series.std()

            # 避免除零
            if spread_std == 0 or pd_isna(spread_std):
                factor_data[dominant_symbol] = float('nan')
                continue

            # 计算 Z-Score = (当前 - 均值) / 标准差
            z_score: float = (current_spread - spread_mean) / spread_std
            factor_data[dominant_symbol] = z_score

        # 返回因子数值
        return factor_data
