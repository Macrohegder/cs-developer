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


class TermStructureStrategy(StrategyTemplate):
    """
    期货期限结构价差反转策略（Futures Term Structure Overreact）

    核心逻辑：
    - 每天根据 spread_zscore 因子排序
    - 做多 Z-Score 最低的 N 个品种（价差远低于均值，预期 F1 相对 F2 反弹）
    - 做空 Z-Score 最高的 N 个品种（价差远高于均值，预期 F1 相对 F2 回落）
    - 每个品种同时交易 F1（主力）和 F2（次主力），方向相反
    - 等权重分仓，持仓周期 holding_period 天（每日新开一份，滚动持仓）

    标准化说明：
    - 严格继承 StrategyTemplate
    - 使用 DominantManager 处理主力合约映射与手数计算
    - 使用 FactorManager 加载因子（含未来函数防护）
    - 额外通过 DataCenter 加载次主力 @2 映射（唯一超出模板的扩展，因策略需交易双合约）
    """

    holding_period: int = 5              # 持仓天数（默认5个交易日≈1周）
    trading_signal: float = 0.1          # 交易信号比例（每端10%，即各选约5个品种）
    factor_name: str = "spread_zscore"   # 因子名称
    factor_parameter: str = "lookback20" # 因子参数标识
    factor_author: str = "futures_term_structure"  # 因子作者标识

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

        # 计算每天的交易资金（资金分成 holding_period 份，每日用一份）
        self.daily_capital = self.capital / self.holding_period

        # 缓存每日目标历史的字典
        self.daily_target_history: Dict[datetime, Series] = {}

        # 初始化主力合约管理器（标准组件，处理 @1 映射与手数计算）
        self.dm = DominantManager(self)

        # 额外加载次主力合约映射数据（@2）
        # 说明：DominantManager 标准只加载 @1，本策略需交易 F2，故自行加载 @2
        self.dc = DataCenter()
        keys_2: List[str] = [f"{s}@2" for s in self.vt_symbols]
        self.dominant_2_df: DataFrame = self.dc.load_reference_df(
            "vnpy_dominant_contract",
            keys_2,
            self.start,
            self.end,
            fillna=True
        )

        # 初始化因子管理器（标准组件）
        self.fm = FactorManager(self)
        self.fm.load_factor_df(self.factor_name, self.factor_parameter, self.factor_author)

    def calculate_target(self, df: DataFrame) -> dict:
        """
        计算目标仓位 — 每日新开一份仓位，持仓 holding_period 天后自动退出
        与标准 CarryStrategy 相同的滚动持仓机制
        """
        # 检查是否有老的目标要移除（超出持仓周期）
        if len(self.daily_target_history) >= self.holding_period:
            first_key = list(self.daily_target_history.keys())[0]
            self.daily_target_history.pop(first_key)

        # 计算添加今天的目标仓位
        daily_target_series = self.calculate_daily_target_series(df)
        self.daily_target_history[self.current_dt] = daily_target_series

        # 汇总整体目标仓位（最近 holding_period 天仓位的总和）
        target_df = concat(self.daily_target_history, axis=1)
        total_target_series = target_df.sum(axis=1)

        # 返回字典格式的目标仓位
        return total_target_series.to_dict()

    def _has_valid_f2(self, dominant_symbol: str) -> bool:
        """检查当前日期该品种是否有有效的次主力合约映射"""
        try:
            key_2: str = f"{dominant_symbol}@2"
            if key_2 not in self.dominant_2_df.columns:
                return False
            f2_vt_symbol: str = self.dominant_2_df.loc[self.current_dt, key_2]
            return bool(f2_vt_symbol) and not pd_isna(f2_vt_symbol)
        except (KeyError, IndexError):
            return False

    def calculate_daily_target_series(self, df: DataFrame) -> Series:
        """计算每天的目标仓位 Series
        
        修正点：
        1. 仅选择有完整 F1+F2 映射的品种，确保多空合约数量严格对称
        2. 若目标品种 F2 缺失，从候选池顺延寻找替代品种
        """
        # 读取当前切片的因子
        factor_series: Series = self.fm.get_factor_series(self.factor_name, self.current_dt)

        # 丢弃为 NA 的因子数值
        factor_series = factor_series.dropna()

        if len(factor_series) == 0:
            return Series(dtype=float)

        # 执行因子排序（因子值从低到高）
        factor_series.sort_values(inplace=True)

        # 筛选交易成分：每端选 trading_signal 比例的品种
        choose: int = int(floor(self.trading_signal * len(factor_series)))
        if choose < 1:
            choose = 1

        # === 确保多空对称：仅选择有有效 F2 映射的品种 ===
        # 从低到高选取有效的多头品种
        long_candidates = []
        for symbol in factor_series.index:
            if len(long_candidates) >= choose:
                break
            if self._has_valid_f2(symbol):
                long_candidates.append(symbol)

        # 从高到低选取有效的空头品种
        short_candidates = []
        for symbol in reversed(factor_series.index):
            if len(short_candidates) >= choose:
                break
            if self._has_valid_f2(symbol):
                short_candidates.append(symbol)

        # 如果有效品种不足，取实际可交易的数量（确保两端数量一致）
        actual_choose = min(len(long_candidates), len(short_candidates))
        if actual_choose < 1:
            return Series(dtype=float)

        long_candidates = long_candidates[:actual_choose]
        short_candidates = short_candidates[:actual_choose]

        # 构建信号序列
        signal_series = Series(0, index=factor_series.index, dtype=float)
        for s in long_candidates:
            signal_series[s] = 1
        for s in short_candidates:
            signal_series[s] = -1

        # 计算每个合约的资金分配
        # 总资金分成 holding_period 份，每天用一份
        # 每份资金再分两半：一半做多、一半做空
        # 每端资金再平均分配到 actual_choose 个品种上
        # 每个品种内部再平分到 F1 和 F2 两个合约
        target_capital = self.daily_capital / (2 * actual_choose * 2)

        # 生成目标仓位（具体交易合约）— 双合约交易
        target_data = {}
        for dominant_symbol, signal in signal_series.items():
            if signal == 0:
                continue

            # F1（主力合约）仓位 — 使用标准 DominantManager
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

            # F2（次主力合约）仓位 — 方向相反，手数与 F1 相同
            # 前面已验证 F2 映射存在，这里直接读取（理论上不会出错）
            key_2: str = f"{dominant_symbol}@2"
            f2_vt_symbol: str = self.dominant_2_df.loc[self.current_dt, key_2]
            target_data[f2_vt_symbol] = -signal * f1_size

        return Series(target_data)
