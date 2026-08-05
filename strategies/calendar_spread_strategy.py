"""
Calendar Spread Strategy（日历价差套利策略）

用于 IC/IM/IF 等股指期货的远近月套利：
- 反向套利：做多近月（@1），做空远月（@2）
- 波动过滤：品种 88 指数 20 日年化波动 >= vol_threshold 时才开仓/持仓
- 数据清洗：过滤 @1=@2、过滤近远月到期月份跨度 >3 个月的异常映射
- 支持单品种或多品种同时运行

在 vnpy_alpharesearch 的 StrategyBacktester 框架下运行。
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from math import floor

from pandas import DataFrame, Series, Timestamp
from vnpy.trader.constant import Interval

from vnpy_alpharesearch.strategy import StrategyTemplate


class CalendarSpreadStrategy(StrategyTemplate):
    """股指期货日历价差套利策略"""

    # --- 核心参数 ---
    # 支持 str（单品种）或 List[str]（多品种）
    products: Union[str, List[str]] = "IM"
    vol_threshold: float = 0.15         # 20 日年化波动阈值；<=0 表示不过滤
    leverage: float = 2.0               # 名义杠杆
    capital_ratio: float = 1.0          # 每日资金使用比例
    commission: float = 0.0001          # 单边手续费率（仅用于手数/市值估算，实际手续费由回测器扣）

    # --- 外部注入数据 ---
    # mapping/vol/contract_df/history_df 可以是单个（单品种）或按品种的字典（多品种）
    mapping: Union[DataFrame, Dict[str, DataFrame]] = None
    idx_vol: Union[Series, Dict[str, Series]] = None
    contract_df: DataFrame = None
    history_df: DataFrame = None

    def __init__(
        self,
        vt_symbols: List[str],
        interval: Interval,
        start: datetime,
        end: datetime,
        capital: int,
        setting: dict,
        output: callable
    ) -> None:
        super().__init__(vt_symbols, interval, start, end, capital, setting, output)

        # 统一 products 为列表
        if isinstance(self.products, str):
            self._products = [self.products]
        else:
            self._products = list(self.products)

        # 统一外部数据为按品种字典
        if isinstance(self.mapping, DataFrame):
            self._mapping = {self._products[0]: self.mapping}
        else:
            self._mapping = dict(self.mapping)

        if isinstance(self.idx_vol, Series):
            self._idx_vol = {self._products[0]: self.idx_vol}
        else:
            self._idx_vol = dict(self.idx_vol)

        # 每个品种独立状态
        self._current_position: Dict[str, Dict[str, int]] = {p: {} for p in self._products}
        self._current_pair: Dict[str, Optional[Tuple[str, str]]] = {p: None for p in self._products}
        self._current_lots: Dict[str, int] = {p: 0 for p in self._products}

        # 资金按品种均分
        self._daily_target_nominal = self.capital * self.leverage * self.capital_ratio / len(self._products)

    def calculate_target(self, df: DataFrame) -> dict:
        """
        计算当前目标仓位
        """
        dt = self.current_dt
        target: Dict[str, int] = {}

        for product in self._products:
            product_target = self._calculate_product_target(product, dt)
            for vt_symbol, lots in product_target.items():
                target[vt_symbol] = target.get(vt_symbol, 0) + lots

        return target

    def _calculate_product_target(self, product: str, dt: Timestamp) -> Dict[str, int]:
        """单个品种的目标仓位"""
        mapping = self._mapping.get(product)
        idx_vol = self._idx_vol.get(product)

        if mapping is None:
            return self._clear_position(product)

        k1 = f"{product}88.CFFEX@1"
        k2 = f"{product}88.CFFEX@2"

        if k1 not in mapping.columns or k2 not in mapping.columns:
            return self._clear_position(product)

        near = mapping.loc[dt, k1] if pd.notna(mapping.loc[dt, k1]) else None
        far = mapping.loc[dt, k2] if pd.notna(mapping.loc[dt, k2]) else None

        # 数据清洗：缺失或 @1=@2
        if near is None or far is None or near == far:
            return self._clear_position(product)

        # 数据清洗：月份跨度 > 3 个月
        near_month = self._get_expiry_month(near)
        far_month = self._get_expiry_month(far)
        span = self._month_span(near_month, far_month)
        if span <= 0 or span > 3:
            return self._clear_position(product)

        # 波动过滤（vol_threshold <= 0 表示不过滤）
        if self.vol_threshold > 0:
            vol = idx_vol.get(dt, float('nan')) if idx_vol is not None else float('nan')
            signal_active = (vol >= self.vol_threshold) if pd.notna(vol) else False
            if not signal_active:
                return self._clear_position(product)

        # 信号活跃：决定目标仓位
        target_pair = (near, far)
        current_pair = self._current_pair[product]

        if current_pair is None:
            lots = self._calculate_lots(product, near, dt)
            if lots < 1:
                return self._clear_position(product)
            self._current_pair[product] = target_pair
            self._current_lots[product] = lots
            self._current_position[product] = {near: lots, far: -lots}
            return self._to_target_dict(product)

        if current_pair != target_pair:
            lots = self._calculate_lots(product, near, dt)
            if lots < 1:
                return self._clear_position(product)
            self._current_pair[product] = target_pair
            self._current_lots[product] = lots
            self._current_position[product] = {near: lots, far: -lots}
            return self._to_target_dict(product)

        # 持仓不变
        return self._to_target_dict(product)

    def _calculate_lots(self, product: str, near_vt_symbol: str, dt: Timestamp) -> int:
        """基于近月价格计算目标手数"""
        try:
            close_price = self.history_df.loc[dt, (near_vt_symbol, "close_price")]
        except Exception:
            return 0

        if pd.isna(close_price) or close_price <= 0:
            return 0

        symbol = near_vt_symbol.split(".")[0]
        multiplier = self.contract_df.loc[symbol, "size"]
        if pd.isna(multiplier) or multiplier <= 0:
            multiplier = 200 if product in ["IC", "IM"] else 300

        lots = int(floor(self._daily_target_nominal / 2 / close_price / multiplier))
        return lots if lots >= 1 else 0

    def _clear_position(self, product: str) -> Dict[str, int]:
        """清空单个品种持仓"""
        self._current_position[product] = {}
        self._current_pair[product] = None
        self._current_lots[product] = 0
        return {}

    def _to_target_dict(self, product: str) -> Dict[str, int]:
        """把单个品种持仓转换为目标仓位字典"""
        return dict(self._current_position[product])

    @staticmethod
    def _get_expiry_month(vt_symbol: str) -> Tuple[int, int]:
        """解析合约到期年月，如 IM2609 -> (2026, 9)"""
        code = vt_symbol.split(".")[0]
        if len(code) < 6:
            return (0, 0)
        year_short = int(code[-4:-2])
        month = int(code[-2:])
        return (2000 + year_short, month)

    @staticmethod
    def _month_span(m1: Tuple[int, int], m2: Tuple[int, int]) -> int:
        """计算两个年月之间的月份跨度"""
        return (m2[0] - m1[0]) * 12 + (m2[1] - m1[1])


import pandas as pd  # noqa: E402
