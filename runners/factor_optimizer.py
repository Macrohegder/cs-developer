#!/usr/bin/env python3
"""
FactorParameterOptimizer —— 因子参数优化器基类

核心流程（一次性数据加载 + 批量向量化计算 + 批量绩效评估）:
  1. preload_data(): 加载价格/合约映射/乘数（只一次）
  2. optimize(): 遍历参数网格，调用 factor.compute_vectorized() + backtest_vectorized()
  3. 汇总结果，按 Sharpe 排序输出

设计原则:
  - 不破坏现有 calculate_factor / StrategyBacktester 生产路径
  - 纯 pandas 向量化，避免逐日循环
  - 因子参数搜索阶段不写 DataCenter（内存计算，确认最优后再持久化）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from math import floor, sqrt
from itertools import product
from typing import Dict, List, Type

import numpy as np
import pandas as pd

from vnpy.trader.constant import Interval
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df

from factors.factor_template import FactorTemplate


class FactorParameterOptimizer:
    """因子参数优化器"""

    def __init__(
        self,
        dominant_symbols: List[str],
        start: datetime,
        end: datetime,
        capital: int = 10_000_000,
        commission: float = 0.0001,
        risk_free: float = 0,
    ):
        self.dominant_symbols = dominant_symbols
        self.start = start
        self.end = end
        self.capital = capital
        self.commission = commission
        self.risk_free = risk_free

        # 预加载的数据
        self.close_df: pd.DataFrame = None
        self.open_df: pd.DataFrame = None
        self.multiplier_series: pd.Series = None
        self._data_loaded = False

    # ==================== 数据加载 ====================

    def preload_data(self) -> None:
        """一次性加载所有需要的数据"""
        if self._data_loaded:
            return

        print("[Optimizer] 预加载价格数据...")
        close_list, open_list = [], []
        for sym in self.dominant_symbols:
            df = load_bar_df(sym, Interval.DAILY, self.start, self.end)
            close_list.append(df["close_price"].rename(sym))
            open_list.append(df["open_price"].rename(sym))
        self.close_df = pd.concat(close_list, axis=1)
        self.open_df = pd.concat(open_list, axis=1)
        print(f"  价格数据: {self.close_df.shape}")

        print("[Optimizer] 预加载合约乘数...")
        self.multiplier_series = self._load_multipliers()
        print(f"  乘数加载完成")

        self._data_loaded = True

    def _load_multipliers(self) -> pd.Series:
        """从合约信息表获取每个品种的乘数"""
        dc = DataCenter()
        contract_df = dc.load_contract_df()

        mapping_keys = [f"{s}@1" for s in self.dominant_symbols]
        mapping_df = dc.load_reference_df(
            table="vnpy_dominant_contract",
            keys=mapping_keys,
            start=self.start,
            end=self.end,
        )

        multipliers = {}
        for ds in self.dominant_symbols:
            key = f"{ds}@1"
            first_contract = mapping_df[key].dropna().iloc[0]
            symbol = first_contract.split(".")[0]
            multipliers[ds] = contract_df.loc[symbol, "multiplier"]

        return pd.Series(multipliers).reindex(self.dominant_symbols)

    # ==================== 因子计算 ====================

    def compute_factor_batch(
        self,
        factor_class: Type[FactorTemplate],
        param_grid: Dict[str, List],
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算多组参数的因子值
        
        返回: {param_key -> factor_df}
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        factor_dict = {}

        for combo in product(*param_values):
            setting = dict(zip(param_names, combo))
            param_key = "_".join(f"{k}{v}" for k, v in setting.items())

            print(f"  [Factor] 计算 {param_key} ...")
            factor_df = factor_class.compute_vectorized(
                price_df=self.close_df,
                **setting
            )
            factor_dict[param_key] = factor_df

        return factor_dict

    # ==================== 绩效评估 ====================

    def evaluate_factor(
        self,
        factor_df: pd.DataFrame,
        holding_period: int = 5,
        trading_signal: float = 0.2,
        leverage: float = 1.0,
    ) -> Dict:
        """
        向量化回测单个因子，返回绩效指标
        """
        # 对齐索引
        common_index = factor_df.index.intersection(self.close_df.index)
        factor_df = factor_df.loc[common_index]
        close_df = self.close_df.loc[common_index]
        open_df = self.open_df.loc[common_index]

        n = len(self.dominant_symbols)
        choose = max(1, int(floor(trading_signal * n)))

        # 1. 每日原始信号：排名做多低因子值，做空高因子值
        rank_df = factor_df.rank(axis=1, method="first", ascending=True)
        signal_raw = pd.DataFrame(0, index=factor_df.index, columns=factor_df.columns)
        signal_raw[rank_df <= choose] = 1
        signal_raw[rank_df > (n - choose)] = -1

        # 2. 滚动聚合
        if holding_period > 1:
            signal_rolling = signal_raw.rolling(window=holding_period, min_periods=1).sum()
        else:
            signal_rolling = signal_raw.copy()

        # 3. 计算每品种目标手数
        daily_capital = self.capital * leverage / holding_period
        target_capital_per_symbol = daily_capital / (2 * choose)

        position = signal_rolling.multiply(
            target_capital_per_symbol / (close_df.multiply(self.multiplier_series, axis=1))
        ).fillna(0)

        # 4. 收益计算（复现 calculate_symbol_pnl 逻辑）
        end_pos = position.shift(1).fillna(0)
        start_pos = end_pos.shift(1).fillna(0)
        pos_change = end_pos - start_pos

        trading_pnl = pos_change.multiply(
            (close_df - open_df).multiply(self.multiplier_series, axis=1)
        )
        holding_pnl = start_pos.multiply(
            (close_df - close_df.shift(1)).multiply(self.multiplier_series, axis=1)
        )
        total_pnl = trading_pnl + holding_pnl

        turnover = pos_change.abs().multiply(
            open_df.multiply(self.multiplier_series, axis=1)
        )
        trading_cost = turnover * self.commission
        net_pnl = total_pnl - trading_cost

        portfolio_pnl = net_pnl.sum(axis=1)

        # 去掉 warmup 期（因子计算需要 lookback  warmup + 2 天 position 延迟）
        # 这里用 60 天作为保守估计，不同 lookback 会有不同 warmup
        valid_pnl = portfolio_pnl.iloc[60:]

        return self._calculate_stats(valid_pnl)

    def _calculate_stats(self, pnl_series: pd.Series) -> Dict:
        """计算绩效指标"""
        balance = pnl_series.cumsum() + self.capital
        if len(balance) == 0 or balance.isna().all():
            return {}

        highlevel = balance.cummax()
        drawdown = balance - highlevel
        ret = balance.pct_change().fillna(0)
        ddpercent = drawdown / highlevel

        start_balance = balance.iloc[0]
        end_balance = balance.iloc[-1]
        total_days = len(balance)

        total_return = end_balance / start_balance - 1
        annual_return = total_return / total_days * 240
        daily_return = ret.mean()
        return_std = ret.std()
        sharpe_ratio = (daily_return - self.risk_free / 240) / return_std * sqrt(240) if return_std > 0 else 0
        max_drawdown = drawdown.min()
        max_ddpercent = ddpercent.min()
        calmar_ratio = -pnl_series.sum() / max_drawdown if max_drawdown != 0 else 0
        win_rate = (pnl_series > 0).sum() / (pnl_series != 0).sum() if (pnl_series != 0).any() else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
        }

    # ==================== 主入口 ====================

    def optimize(
        self,
        factor_class: Type[FactorTemplate],
        param_grid: Dict[str, List],
        strategy_params: Dict = None,
    ) -> tuple:
        """
        主入口：批量因子计算 + 批量绩效评估
        
        参数:
            factor_class: 因子类（需实现 compute_vectorized）
            param_grid: 参数网格，如 {"lookback": [60, 120, 180, 240]}
            strategy_params: 固定策略参数，如 {"holding_period": 5, "trading_signal": 0.2, "leverage": 1.0}
        
        返回:
            (results_df, factor_dict): 结果表 + 因子数据字典
        """
        if strategy_params is None:
            strategy_params = {"holding_period": 5, "trading_signal": 0.2, "leverage": 1.0}

        self.preload_data()

        # Step 1: 批量计算因子
        print("\n[Optimizer] 批量计算因子...")
        factor_dict = self.compute_factor_batch(factor_class, param_grid)

        # Step 2: 批量评估绩效
        print("\n[Optimizer] 批量评估绩效...")
        results = []
        for param_key, factor_df in factor_dict.items():
            stats = self.evaluate_factor(factor_df, **strategy_params)
            result = {
                "param_key": param_key,
                **{k: v for k, v in zip(param_grid.keys(), param_key.split("_"))},
                **stats,
            }
            # 解析 param_key 为各个参数值
            param_names = list(param_grid.keys())
            for name in param_names:
                # 从 param_key 中提取值，如 "lookback60" -> 60
                prefix = name
                # 找到 param_key 中以 prefix 开头的部分
                for part in param_key.split("_"):
                    if part.startswith(prefix):
                        try:
                            result[name] = int(part[len(prefix):])
                        except ValueError:
                            result[name] = part[len(prefix):]
                        break

            results.append(result)
            print(
                f"  {param_key}: "
                f"total={stats.get('total_return', 0):.2%}, "
                f"ann={stats.get('annual_return', 0):.2%}, "
                f"sharpe={stats.get('sharpe_ratio', 0):.2f}, "
                f"maxdd={stats.get('max_ddpercent', 0):.2%}"
            )

        results_df = pd.DataFrame(results)
        results_df.sort_values("sharpe_ratio", ascending=False, inplace=True)
        return results_df, factor_dict
