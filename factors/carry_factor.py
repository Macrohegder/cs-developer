"""
Carry Factor（基差/展期收益因子）

核心逻辑：
- 对每个商品品种，计算主力合约（F1）与次主力合约（F2）的年化价差
- Carry = (F2 - F1) / F2 / ΔT * 365
- 其中 ΔT 为两个合约的到期时间差（天）
- Carry 越高，说明远期溢价越大，展期收益越负（多头持有成本越高）
- 标准做法：做多低 Carry（正向市场/Contango 浅的品种），做空高 Carry

性能优化：
- 所有参考数据（主力映射、合约信息）在 __init__ 中预加载到内存
- calculate_factor 中只做纯内存计算，避免任何 IO/数据库查询

作者：cross_sectional
"""

from datetime import datetime
from typing import List, Dict

from pandas import DataFrame, Series

from vnpy_alpharesearch.factor import FactorTemplate
from vnpy_alpharesearch import DataCenter


class CarryFactor(FactorTemplate):
    """Carry（展期收益）因子"""

    start: datetime = None                  # 开始时间
    end: datetime = None                    # 结束时间
    dominant_symbols: List[str] = None      # 主力合约表键（如 RB88.SHFE）

    def __init__(self, vt_symbols: List[str], setting: dict) -> None:
        """预加载所有参考数据到内存，避免循环内查询"""
        super().__init__(vt_symbols, setting)

        dc: DataCenter = DataCenter()

        # 1. 初始化主力合约映射关系的查询主键
        keys: List[str] = []
        for s in self.dominant_symbols:
            keys.append(f"{s}@1")  # 主力
            keys.append(f"{s}@2")  # 次主力

        # 2. 加载完整的合约映射 DataFrame 到内存
        self.dominant_df: DataFrame = dc.load_reference_df(
            "vnpy_dominant_contract",
            keys,
            self.start,
            self.end,
            fillna=True
        )

        # 3. 加载合约信息并构建到期时间字典 {symbol: expiry_datetime}
        contract_df: DataFrame = dc.load_contract_df()
        self.expiry_map: Dict[str, datetime] = {}
        for symbol in contract_df.index:
            try:
                expiry_str = contract_df.loc[symbol, "enddate"]
                if isinstance(expiry_str, str):
                    self.expiry_map[symbol] = datetime.strptime(expiry_str, "%Y-%m-%d")
            except (ValueError, KeyError):
                continue

    def calculate_factor(self, df: DataFrame) -> dict:
        """计算 Carry 因子数值（纯内存计算）"""
        dt: datetime = df.index[-1]

        # 从预加载的 dominant_df 中查找当前日期的映射
        try:
            dominant_series: Series = self.dominant_df.loc[dt]
        except KeyError:
            return {s: float("nan") for s in self.dominant_symbols}

        price_series: Series = df.iloc[-1, :]
        factor_data: Dict[str, float] = {}

        for dominant_symbol in self.dominant_symbols:
            # 提取主力和次主力合约代码
            current_symbol: str = self._get_contract(dominant_series, dominant_symbol, 1)
            next_symbol: str = self._get_contract(dominant_series, dominant_symbol, 2)

            if not isinstance(current_symbol, str) or not isinstance(next_symbol, str):
                factor_data[dominant_symbol] = float("nan")
                continue

            # 提取收盘价
            try:
                current_close: float = price_series[dominant_symbol]["close_price"]
                next_close: float = price_series[dominant_symbol.replace("88.", "88A2.")]["close_price"]
            except (KeyError, TypeError):
                factor_data[dominant_symbol] = float("nan")
                continue

            # 提取到期时间差
            current_expiry: datetime = self.expiry_map.get(current_symbol.split(".")[0])
            next_expiry: datetime = self.expiry_map.get(next_symbol.split(".")[0])

            if current_expiry is None or next_expiry is None:
                factor_data[dominant_symbol] = float("nan")
                continue

            delta_days: int = (next_expiry - current_expiry).days
            if delta_days <= 0 or next_close == 0:
                factor_data[dominant_symbol] = float("nan")
                continue

            # 计算年化 Carry
            # 正值 = 远期升水（Contango），负值 = 远期贴水（Backwardation）
            carry_value: float = (next_close - current_close) / next_close / delta_days * 365
            factor_data[dominant_symbol] = carry_value

        return factor_data

    @staticmethod
    def _get_contract(series: Series, dominant_symbol: str, level: int) -> str:
        """从 Series 中查询主力/次主力合约代码"""
        key: str = f"{dominant_symbol}@{level}"
        return series[key]
