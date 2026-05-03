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


class TimeSeriesTermStructureStrategy(StrategyTemplate):
    """
    期限结构价差时间序列反转策略

    核心逻辑：
    - 每个品种独立判断，不做横截面排序
    - spread_return > 0（过去10天价差上涨）→ 预期反转 → short 价差
    - spread_return < 0（过去10天价差下跌）→ 预期反转 → long 价差
    - 所有有信号的品种同时交易，等权重分仓
    - 每个品种同时交易 F1（主力）和 F2（次主力），方向相反
    - 持仓周期 holding_period 天
    """

    holding_period: int = 10
    max_positions: int = 10       # 每天最多交易 N 个品种（按 |factor| 排序选最强的 N 个）
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

        # === 时间序列逻辑：每个品种根据因子正负独立判断 ===
        # factor > 0 → short 价差 (short F1, long F2)
        # factor < 0 → long 价差 (long F1, short F2)
        
        # 筛选有有效 F2 映射且因子不为0的品种
        valid_signals = {}
        for symbol, factor_value in factor_series.items():
            if factor_value == 0:
                continue
            if not self._has_valid_f2(symbol):
                continue
            # 信号：factor > 0 → -1 (short), factor < 0 → +1 (long)
            signal = -1 if factor_value > 0 else 1
            valid_signals[symbol] = signal

        n_signals = len(valid_signals)
        if n_signals == 0:
            return Series(dtype=float)

        # 按 |factor| 绝对值排序，只保留信号最强的 max_positions 个
        # 这样避免资金过度分散导致手数为0
        if n_signals > self.max_positions:
            # 获取原始 factor 值用于排序
            sorted_symbols = factor_series.reindex(
                [s for s in valid_signals.keys()]
            ).abs().sort_values(ascending=False).head(self.max_positions).index.tolist()
            valid_signals = {s: valid_signals[s] for s in sorted_symbols}
            n_signals = self.max_positions

        # 资金分配：
        # - 总资金分成 holding_period 份，每天用一份
        # - 每份资金平均分配到 n_signals 个品种
        # - 每个品种内部再平分到 F1 和 F2 两个合约
        target_capital = self.daily_capital / (n_signals * 2)

        target_data = {}
        for dominant_symbol, signal in valid_signals.items():
            # F1（主力合约）
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

            # F2（次主力合约）
            key_2: str = f"{dominant_symbol}@2"
            f2_vt_symbol: str = self.dominant_2_df.loc[self.current_dt, key_2]
            target_data[f2_vt_symbol] = -signal * f1_size

        return Series(target_data)
