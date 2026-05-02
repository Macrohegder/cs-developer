from datetime import datetime, timedelta
from typing import List, Dict

from pandas import DataFrame, Series

from vnpy_alpharesearch.factor import FactorTemplate
from vnpy_alpharesearch import DataCenter


class SpreadReturnFactor(FactorTemplate):
    """
    周度 F1-F2 价差收益率因子
    
    核心逻辑：
    - F1 = 主力连续合约 (88)
    - F2 = 次主力连续合约 (889)
    - 每周计算 spread_return = F1周收益率 - F2周收益率
    - 基于负自相关假设：本周 spread 越高，下周越倾向反转
    """

    start: datetime = None                  # 开始时间
    end: datetime = None                    # 结束时间
    dominant_symbols: List[str] = None      # 主力合约表键（如 RB88.SHFE）
    lookback: int = 5                       # 回看周期（默认5个交易日≈1周）

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

        # 计算周度 spread return
        factor_data: Dict[str, float] = {}

        for dominant_symbol in self.dominant_symbols:
            # 提取主力和次主力合约的收盘价
            try:
                # 主力连续合约 (88)
                f1_close_today: float = df[dominant_symbol]["close_price"].iloc[-1]
                f1_close_ago: float = df[dominant_symbol]["close_price"].iloc[-self.lookback]
                
                # 次主力连续合约 (889) - 将 88. 替换为 889.
                f2_symbol: str = dominant_symbol.replace("88.", "889.")
                f2_close_today: float = df[f2_symbol]["close_price"].iloc[-1]
                f2_close_ago: float = df[f2_symbol]["close_price"].iloc[-self.lookback]
            except (KeyError, IndexError):
                # 数据缺失（合约未上市或行情缺失），跳过
                factor_data[dominant_symbol] = float('nan')
                continue

            # 计算周度收益率
            f1_return: float = (f1_close_today - f1_close_ago) / f1_close_ago
            f2_return: float = (f2_close_today - f2_close_ago) / f2_close_ago

            # 计算 spread return = F1收益率 - F2收益率
            spread_return: float = f1_return - f2_return
            factor_data[dominant_symbol] = spread_return

        # 返回因子数值
        return factor_data
