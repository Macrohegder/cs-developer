"""
Cross-Sectional Multi-Factor Strategy Template（横截面多因子策略模板）

通用框架特性：
- 支持任意横截面因子（Carry、Momentum、Skew 等）
- 每天根据因子值对所有品种排序
- 做多/做空两端指定比例的品种
- 通过 DominantManager 直接交易主力合约（非价差）
- 支持滚动持仓（分仓）或信号平均两种仓位汇总模式
- 参数化：因子名称、持仓周期、多空比例、因子方向

使用方式：
1. 继承本类或直接使用（通过 setting 传入因子参数）
2. 实现/加载对应的 Factor（继承 FactorTemplate）
3. 在回测脚本中配置 CONFIG

作者：cross_sectional
"""

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


class CrossSectionalStrategy(StrategyTemplate):
    """横截面多因子多空策略模板"""

    # --- 核心参数（可通过 setting 覆盖）---
    holding_period: int = 5             # 持仓天数/分仓份数
    trading_signal: float = 0.2         # 每端交易比例（0.2 = 多空各20%）
    factor_name: str = ""               # 因子名称
    factor_parameter: str = ""          # 因子参数标识
    factor_author: str = ""             # 因子作者标识
    long_low: bool = True               # True=做多低因子值，False=做多高因子值
    aggregation: str = "sum"            # "sum"=分仓滚动叠加, "mean"=信号平均
    leverage: float = 1.0               # 名义价值杠杆倍数（1.0=无杠杆，2.0=2倍名义杠杆）

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

        # 计算每天的目标名义价值（资金分成 holding_period 份，每日用一份，再乘以杠杆）
        if self.aggregation == "mean":
            # mean 模式下，每天使用全部杠杆后的资金
            self.daily_capital = self.capital * self.leverage
        else:
            # sum 模式下，每天使用 capital * leverage / holding_period
            self.daily_capital = self.capital * self.leverage / self.holding_period

        # 缓存每日目标历史的字典
        self.daily_target_history: Dict[Timestamp, Series] = {}

        # 初始化主力合约管理器（处理 @1 映射与手数计算）
        self.dm = DominantManager(self)

        # 初始化因子管理器
        self.fm = FactorManager(self)
        self.fm.load_factor_df(self.factor_name, self.factor_parameter, self.factor_author)

    def calculate_target(self, df: DataFrame) -> dict:
        """
        计算目标仓位 — 每日新开一份仓位，汇总历史持仓
        """
        # 检查是否有老的目标要移除（超出持仓周期）
        if len(self.daily_target_history) >= self.holding_period:
            first_key = list(self.daily_target_history.keys())[0]
            self.daily_target_history.pop(first_key)

        # 计算添加今天的目标仓位
        daily_target_series = self.calculate_daily_target_series(df)
        self.daily_target_history[self.current_dt] = daily_target_series

        # 汇总整体目标仓位
        target_df = concat(self.daily_target_history, axis=1)

        if self.aggregation == "mean":
            # 信号平均：持仓量会随时间平滑变化
            total_target_series = target_df.mean(axis=1)
        else:
            # 分仓滚动：每日独立仓位求和
            total_target_series = target_df.sum(axis=1)

        # 返回字典格式的目标仓位
        return total_target_series.to_dict()

    def calculate_daily_target_series(self, df: DataFrame) -> Series:
        """
        计算每天的目标仓位 Series

        流程：
        1. 获取当日因子值
        2. 排序并筛选两端
        3. 动态计算每品种资金
        4. 通过 DominantManager 计算具体合约手数
        """
        # 读取当前切片的因子
        factor_series: Series = self.fm.get_factor_series(self.factor_name, self.current_dt)

        # 丢弃为 NA 的因子数值
        factor_series = factor_series.dropna()

        if len(factor_series) == 0:
            return Series(dtype=float)

        # 执行因子排序（从低到高）
        factor_series.sort_values(inplace=True)

        # 筛选交易成分：每端选 trading_signal 比例的品种
        choose: int = int(floor(self.trading_signal * len(factor_series)))
        if choose < 1:
            choose = 1

        # 动态计算每品种目标资金
        # 每端总资金 = daily_capital / 2，平均分配到 choose 个品种
        target_capital: float = self.daily_capital / (2 * choose)

        # 生成目标信号序列
        signal_series = Series(0, index=factor_series.index, dtype=float)

        if self.long_low:
            # 标准模式：做多低因子值，做空高因子值
            signal_series.iloc[:choose] = 1      # 多头
            signal_series.iloc[-choose:] = -1    # 空头
        else:
            # 反向模式：做空低因子值，做多高因子值
            signal_series.iloc[:choose] = -1     # 空头
            signal_series.iloc[-choose:] = 1     # 多头

        # 生成目标仓位（具体交易合约）
        target_data = {}
        for dominant_symbol, signal in signal_series.items():
            if signal == 0:
                continue

            try:
                vt_symbol, size = self.dm.calculate_trading_volume(
                    self.current_dt,
                    dominant_symbol,
                    target_capital
                )
                if size > 0:
                    target_data[vt_symbol] = signal * size
            except Exception:
                # 映射缺失或手数计算失败时跳过该品种
                continue

        return Series(target_data)
