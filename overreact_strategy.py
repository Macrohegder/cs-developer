from datetime import datetime
from typing import List, Dict
from math import floor

from pandas import DataFrame, Series, concat, Timestamp
from vnpy.trader.constant import Interval

from vnpy_alpharesearch.strategy import (
    StrategyTemplate,
    DominantManager,
    FactorManager
)


class OverreactStrategy(StrategyTemplate):
    """
    期货期限结构价差反转策略（Futures Overreact）
    
    核心逻辑：
    - 每周五（或每 holding_period 天）根据 spread_return 因子排序
    - 做多 spread return 最低的 N 个品种（预期反弹）
    - 做空 spread return 最高的 N 个品种（预期反转）
    - 等权重，持有一周
    """

    holding_period: int = 5              # 持仓天数（默认5个交易日≈1周）
    trading_signal: float = 0.1          # 交易信号比例（每端10%，即各选约5个品种）
    factor_name: str = "spread_return"   # 因子名称

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
        """"""
        super().__init__(vt_symbols, interval, start, end, capital, setting, output)

        # 计算每天的交易资金
        self.daily_capital = self.capital / self.holding_period

        # 计算每份仓位的交易资金
        # 多空各一半资金，每端再分配到 trading_signal 比例的品种上
        self.target_capital = self.daily_capital / (2 * len(vt_symbols) * self.trading_signal)

        # 缓存每日目标历史的字典
        self.daily_target_history: Dict[Timestamp, Series] = {}

        # 初始化主力合约管理器
        self.dm = DominantManager(self)

        # 初始化因子管理器
        self.fm = FactorManager(self)
        self.fm.load_factor_df(self.factor_name, "", "")

    def calculate_target(self, df: DataFrame) -> dict:
        """计算目标仓位"""
        # 检查是否到达调仓日（每周五或每 holding_period 天）
        # 简化：只在周五调仓，其他日子保持持仓
        if self.current_dt.weekday() != 4:  # 4 = Friday
            # 非调仓日，返回空字典（保持现有持仓）
            return {}

        # 检查是否有老的目标要移除（超过 holding_period 的持仓）
        if len(self.daily_target_history) >= self.holding_period:
            first_key = list(self.daily_target_history.keys())[0]
            self.daily_target_history.pop(first_key)

        # 计算添加今天的目标仓位
        daily_target_series = self.calculate_daily_target_series(df)
        self.daily_target_history[self.current_dt] = daily_target_series

        # 汇总整体目标仓位
        target_df = concat(self.daily_target_history, axis=1)
        total_target_series = target_df.sum(axis=1)

        # 返回字典格式的目标仓位
        return total_target_series.to_dict()

    def calculate_daily_target_series(self, df: DataFrame) -> Series:
        """计算每天的目标仓位Series"""
        # 读取当前切片的因子
        factor_series: Series = self.fm.get_factor_series(self.factor_name, self.current_dt)

        # 丢弃为NA的因子数值
        factor_series = factor_series.dropna()

        if len(factor_series) == 0:
            return Series()

        # 执行因子排序（升序：负值在前，正值在后）
        factor_series.sort_values(inplace=True)

        # 筛选交易成分
        choose: int = int(floor(self.trading_signal * len(factor_series)))
        if choose < 1:
            choose = 1

        # 生成目标信号
        signal_series = Series(0, index=factor_series.index)    # 目标仓位序列
        signal_series.iloc[:choose] = 1                         # 多头部分（spread return 最低）
        signal_series.iloc[-choose:] = -1                       # 空头部分（spread return 最高）

        # 生成目标仓位（具体交易合约）
        target_data = {}
        for dominant_symbol, signal in signal_series.iteritems():
            if signal == 0:
                continue
            vt_symbol, size = self.dm.calculate_trading_volume(
                self.current_dt,
                dominant_symbol,
                self.target_capital
            )
            target_data[vt_symbol] = signal * size

        return Series(target_data)
