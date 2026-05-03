from datetime import datetime
from typing import List, Dict
from math import floor

from pandas import DataFrame, Series, concat, isna as pd_isna
from vnpy.trader.constant import Interval

from vnpy_alpharesearch.strategy import (
    StrategyTemplate,
    DominantManager,
    FactorManager
)
from vnpy_alpharesearch import DataCenter


class NonferrousMetalTermStructureStrategy(StrategyTemplate):
    """
    有色金属板块期限结构价差反转策略

    核心逻辑：
    - 限定品种：CU, AL, ZN, NI, SN, AU, AG, PB（8个有色金属品种）
    - 每天根据 spread_return 因子排序
    - 只选 1 对：做多因子最低的 1 个品种，做空因子最高的 1 个品种
    - 每个品种同时交易 F1（主力）和 F2（次主力），方向相反
    - 等权重分仓，持仓周期 holding_period 天
    """

    holding_period: int = 10
    factor_name: str = "spread_return"
    factor_parameter: str = "lookback10"
    factor_author: str = "futures_term_structure"

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

        self.daily_capital = self.capital / self.holding_period
        self.daily_target_history: Dict[datetime, Series] = {}
        self.dm = DominantManager(self)

        self.dc = DataCenter()
        keys_2: List[str] = [f"{s}@2" for s in self.vt_symbols]
        self.dominant_2_df: DataFrame = self.dc.load_reference_df(
            "vnpy_dominant_contract",
            keys_2,
            self.start,
            self.end,
            fillna=True
        )

        self.fm = FactorManager(self)
        self.fm.load_factor_df(self.factor_name, self.factor_parameter, self.factor_author)

    def calculate_target(self, df: DataFrame) -> dict:
        if len(self.daily_target_history) >= self.holding_period:
            first_key = list(self.daily_target_history.keys())[0]
            self.daily_target_history.pop(first_key)

        daily_target_series = self.calculate_daily_target_series(df)
        self.daily_target_history[self.current_dt] = daily_target_series

        target_df = concat(self.daily_target_history, axis=1)
        total_target_series = target_df.sum(axis=1)
        return total_target_series.to_dict()

    def _has_valid_f2(self, dominant_symbol: str) -> bool:
        try:
            key_2: str = f"{dominant_symbol}@2"
            if key_2 not in self.dominant_2_df.columns:
                return False
            f2_vt_symbol: str = self.dominant_2_df.loc[self.current_dt, key_2]
            return bool(f2_vt_symbol) and not pd_isna(f2_vt_symbol)
        except (KeyError, IndexError):
            return False

    def calculate_daily_target_series(self, df: DataFrame) -> Series:
        factor_series: Series = self.fm.get_factor_series(self.factor_name, self.current_dt)
        factor_series = factor_series.dropna()

        if len(factor_series) == 0:
            return Series(dtype=float)

        factor_series.sort_values(inplace=True)

        long_candidate = None
        for symbol in factor_series.index:
            if self._has_valid_f2(symbol):
                long_candidate = symbol
                break

        short_candidate = None
        for symbol in reversed(factor_series.index):
            if self._has_valid_f2(symbol):
                short_candidate = symbol
                break

        if long_candidate is None or short_candidate is None or long_candidate == short_candidate:
            return Series(dtype=float)

        signal_series = Series(0, index=factor_series.index, dtype=float)
        signal_series[long_candidate] = 1
        signal_series[short_candidate] = -1

        target_capital = self.daily_capital / (2 * 1 * 2)

        target_data = {}
        for dominant_symbol, signal in signal_series.items():
            if signal == 0:
                continue

            try:
                f1_vt_symbol, f1_size = self.dm.calculate_trading_volume(
                    self.current_dt,
                    dominant_symbol,
                    target_capital
                )
                if f1_size > 0:
                    target_data[f1_vt_symbol] = signal * f1_size
                else:
                    continue
            except Exception:
                continue

            key_2: str = f"{dominant_symbol}@2"
            f2_vt_symbol: str = self.dominant_2_df.loc[self.current_dt, key_2]
            target_data[f2_vt_symbol] = -signal * f1_size

        return Series(target_data)
