#!/usr/bin/env python3
"""
BatchBacktestEngine — 向量化批量回测引擎

职责：
1. 一次性加载所有价格数据（主力连续合约）
2. 从本地 parquet 批量加载预计算因子
3. 对每个因子执行向量化回测（截面排序 → 信号生成 → 滚动持仓 → 组合收益）
4. 输出标准化绩效指标（年化收益、夏普、最大回撤、Calmar）

核心假设（简化模型）：
- 使用主力连续合约 close-to-close 收益率近似实际收益
- 忽略合约切换映射和整数手数限制（等权资金分配）
- 手续费按名义调仓比例估算
- 目标：秒级完成单个因子回测，分钟级完成 39 个因子批量回测

与 StrategyBacktester 的区别：
- 不生成 target_df（不映射到具体合约）
- 不逐日迭代策略对象
- 完全向量化计算

作者：cross_sectional
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from vnpy.trader.constant import Interval

# 项目路径
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from factor_registry import FactorRegistry, FactorMeta, get_registry
from factor_engine import FactorEngine


# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------
DATA_DIR = Path("/root/cs_developer/factor_system/data")
TRADING_DAYS_PER_YEAR = 240


@dataclass
class BacktestResult:
    """单个因子的回测结果"""
    factor_name: str
    factor_meta: Optional[FactorMeta]
    
    # 参数
    holding_period: int
    trading_signal: float
    long_low: bool
    aggregation: str
    leverage: float
    commission: float
    
    # 绩效指标
    annual_return: float
    annual_vol: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_daily_return: float
    avg_turnover: float
    
    # 时间序列
    nav_series: pd.Series
    net_return_series: pd.Series
    gross_return_series: pd.Series
    turnover_series: pd.Series
    cost_series: pd.Series
    
    # 统计
    total_days: int
    profit_days: int
    loss_days: int
    
    def to_dict(self) -> dict:
        """转换为字典（不含时间序列）"""
        return {
            "factor_name": self.factor_name,
            "category": self.factor_meta.category if self.factor_meta else "",
            "ic_direction": self.factor_meta.ic_direction if self.factor_meta else 0,
            "holding_period": self.holding_period,
            "trading_signal": self.trading_signal,
            "long_low": self.long_low,
            "aggregation": self.aggregation,
            "leverage": self.leverage,
            "commission": self.commission,
            "annual_return": self.annual_return,
            "annual_vol": self.annual_vol,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "avg_daily_return": self.avg_daily_return,
            "avg_turnover": self.avg_turnover,
            "total_days": self.total_days,
            "profit_days": self.profit_days,
            "loss_days": self.loss_days,
        }


class BatchBacktestEngine:
    """向量化批量回测引擎"""
    
    def __init__(
        self,
        dominant_symbols: List[str],
        start: str,
        end: str,
        data_dir: Path = DATA_DIR,
        primary_suffix: str = "99",
        secondary_suffix: str = "889",
        verbose: bool = True
    ):
        self.dominant_symbols = dominant_symbols
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.data_dir = data_dir
        self.primary_suffix = primary_suffix
        self.secondary_suffix = secondary_suffix
        self.verbose = verbose
        
        # 预加载价格数据
        self._close_df: Optional[pd.DataFrame] = None
        self._returns_df: Optional[pd.DataFrame] = None
        self._load_price_data()
        
        # 注册表
        self.registry = get_registry()
    
    # =================================================================
    # 数据加载
    # =================================================================
    
    def _load_price_data(self):
        """加载主力连续合约收盘价（wide format）"""
        if self.verbose:
            print("=" * 70)
            print(f"BatchBacktestEngine — 加载价格数据 ({self.primary_suffix} 指数)")
            print("=" * 70)
        
        # 使用 FactorEngine 加载数据（复用其缓存机制）
        engine = FactorEngine(
            dominant_symbols=self.dominant_symbols,
            start=self.start,
            end=self.end,
            primary_suffix=self.primary_suffix,
            secondary_suffix=self.secondary_suffix,
            verbose=self.verbose
        )
        engine.load_all_data()
        
        self._close_df = engine._close.copy()
        self._returns_df = self._close_df.pct_change()
        
        if self.verbose:
            print(f"\n[OK] 价格数据加载完成")
            print(f"     品种数: {len(self.dominant_symbols)}")
            print(f"     交易日: {len(self._close_df)}")
    
    def load_factor_df(self, factor_name: str) -> Optional[pd.DataFrame]:
        """从本地 parquet 加载因子数据"""
        # 尝试多种文件名格式
        candidates = [
            f"{factor_name}_{self.start.strftime('%Y-%m-%d')}_{self.end.strftime('%Y-%m-%d')}.parquet",
        ]
        
        for fname in candidates:
            fpath = self.data_dir / fname
            if fpath.exists():
                df = pd.read_parquet(fpath)
                # 确保列名与 dominant_symbols 对齐
                df = df.reindex(columns=self.dominant_symbols)
                return df
        
        # 尝试通配匹配
        for fpath in self.data_dir.glob(f"{factor_name}_*.parquet"):
            df = pd.read_parquet(fpath)
            df = df.reindex(columns=self.dominant_symbols)
            return df
        
        if self.verbose:
            print(f"[WARN] 因子 '{factor_name}' 的 parquet 文件未找到")
        return None
    
    # =================================================================
    # 核心回测逻辑
    # =================================================================
    
    def _ensure_price_data(self, primary_suffix: str, secondary_suffix: str):
        """确保价格数据已加载，如后缀不同则重新加载"""
        if self._close_df is not None and self.primary_suffix == primary_suffix and self.secondary_suffix == secondary_suffix:
            return
        self.primary_suffix = primary_suffix
        self.secondary_suffix = secondary_suffix
        self._load_price_data()
    
    def run_factor(
        self,
        factor_name: str,
        factor_df: Optional[pd.DataFrame] = None,
        holding_period: int = 5,
        trading_signal: float = 0.2,
        long_low: Optional[bool] = None,
        aggregation: str = "sum",
        leverage: float = 2.0,
        commission: float = 0.0001,
        auto_direction: bool = False,
    ) -> Optional[BacktestResult]:
        """
        向量化回测单个因子
        
        参数:
            factor_name: 因子名称（用于加载数据和元数据）
            factor_df: 可选，直接传入因子 DataFrame（避免重复加载）
            holding_period: 持仓天数
            trading_signal: 每端交易比例
            long_low: True=做多低因子值，False=做多高因子值。None=自动根据 ic_direction 推断
            aggregation: "sum"=分仓滚动, "mean"=信号平均
            leverage: 名义杠杆倍数
            commission: 单边手续费率
        """
        # 0. 根据因子注册表确定回测价格基础
        meta = self.registry.get(factor_name)
        if meta and meta.backtest_price_suffix:
            target_suffix = meta.backtest_price_suffix
            # Carry 类因子需要次主力，其他不需要
            target_sec = "88A2" if factor_name in ["carry_ret", "carry_momentum", "spread_zscore", "spread_return"] else "889"
            self._ensure_price_data(target_suffix, target_sec)
        
        # 1. 加载因子数据
        if factor_df is None:
            factor_df = self.load_factor_df(factor_name)
        
        if factor_df is None or factor_df.empty:
            return None
        
        # 2. 从注册表推断 long_low（强制与标准化模板一致）
        if long_low is None:
            meta = self.registry.get(factor_name)
            if meta and meta.ic_direction == -1:
                long_low = True   # IC 负相关 → 做多低值
            else:
                long_low = False  # IC 正相关 → 做多高值
        
        # 2b. 自动方向选择（仅当显式启用时，且会提示警告）
        if auto_direction and long_low is not None:
            res_true = self._run_single_direction(
                factor_name=factor_name,
                factor_df=factor_df,
                holding_period=holding_period,
                trading_signal=trading_signal,
                long_low=True,
                aggregation=aggregation,
                leverage=leverage,
                commission=commission,
            )
            res_false = self._run_single_direction(
                factor_name=factor_name,
                factor_df=factor_df,
                holding_period=holding_period,
                trading_signal=trading_signal,
                long_low=False,
                aggregation=aggregation,
                leverage=leverage,
                commission=commission,
            )
            
            candidates = []
            if res_true is not None:
                candidates.append((res_true.sharpe_ratio, True, res_true))
            if res_false is not None:
                candidates.append((res_false.sharpe_ratio, False, res_false))
            
            if not candidates:
                return None
            
            # 选择夏普更高的方向
            best_sharpe, best_direction, best_result = max(candidates, key=lambda x: x[0])
            
            if self.verbose:
                print(f"       [WARN] 自动方向选择已启用，结果可能偏离标准化模板")
                print(f"       方向选择: long_low={best_direction} (True夏普={res_true.sharpe_ratio:+.2f}, "
                      f"False夏普={res_false.sharpe_ratio:+.2f})")
            
            return best_result
        
        return self._run_single_direction(
            factor_name=factor_name,
            factor_df=factor_df,
            holding_period=holding_period,
            trading_signal=trading_signal,
            long_low=long_low,
            aggregation=aggregation,
            leverage=leverage,
            commission=commission,
        )
    
    def _run_single_direction(
        self,
        factor_name: str,
        factor_df: Optional[pd.DataFrame],
        holding_period: int,
        trading_signal: float,
        long_low: bool,
        aggregation: str,
        leverage: float,
        commission: float,
    ) -> Optional[BacktestResult]:
        """回测单个方向（内部方法）"""
        
        # 1. 加载因子数据
        if factor_df is None:
            factor_df = self.load_factor_df(factor_name)
        
        if factor_df is None or factor_df.empty:
            return None
        
        # 3. 对齐日期
        factor_df, close_df = factor_df.align(self._close_df, join='inner', axis=0)
        returns_df = close_df.pct_change()
        
        if len(factor_df) < holding_period + 10:
            if self.verbose:
                print(f"[WARN] 因子 '{factor_name}' 有效数据不足 ({len(factor_df)} 天)")
            return None
        
        # 4. 生成每日原始信号（截面排名）
        raw_signal = self._generate_raw_signals(
            factor_df, trading_signal, long_low
        )
        
        # 5. 滚动持仓
        if aggregation == "mean":
            position_df = raw_signal.rolling(
                window=holding_period, min_periods=1
            ).mean()
        else:
            position_df = raw_signal.rolling(
                window=holding_period, min_periods=1
            ).sum()
        
        # 6. 计算组合日收益率（gross，忽略手续费）
        # 信号是 t 日收盘后生成的，对应 t+1 日收益
        position_shifted = position_df.shift(1)
        
        # 归一化权重（等权假设）
        weight_abs_sum = position_shifted.abs().sum(axis=1).replace(0, np.nan)
        weight_df = position_shifted.div(weight_abs_sum, axis=0)
        
        # 组合日收益率 = sum(weight * return)
        gross_return = (weight_df * returns_df).sum(axis=1)
        gross_return = gross_return.fillna(0)
        
        # 7. 估算手续费
        # 调仓比例 = 仓位变化绝对值之和 / 仓位绝对值之和
        pos_change = position_df.diff().abs().sum(axis=1)
        total_pos = position_df.abs().sum(axis=1).replace(0, np.nan)
        turnover_ratio = (pos_change / total_pos).fillna(0)
        
        # 双边交易成本（买卖各一次）
        # turnover_ratio 是名义市值比例，乘以 leverage 转换为按 capital 计算
        cost = turnover_ratio * commission * 2 * leverage
        cost = cost.fillna(0)
        
        # 8. 净值曲线
        net_return = gross_return * leverage - cost
        net_return = net_return.fillna(0)
        nav = (1 + net_return).cumprod()
        
        # 9. 计算绩效指标
        result = self._calculate_metrics(
            factor_name=factor_name,
            net_return=net_return,
            gross_return=gross_return * leverage,
            turnover_ratio=turnover_ratio,
            cost=cost,
            nav=nav,
            holding_period=holding_period,
            trading_signal=trading_signal,
            long_low=long_low,
            aggregation=aggregation,
            leverage=leverage,
            commission=commission,
        )
        
        return result
    
    def _generate_raw_signals(
        self,
        factor_df: pd.DataFrame,
        trading_signal: float,
        long_low: bool
    ) -> pd.DataFrame:
        """
        每天对因子值截面排序，生成原始信号
        
        返回 DataFrame: index=date, columns=symbols, values=+1/-1/0
        """
        signal_df = pd.DataFrame(
            0.0,
            index=factor_df.index,
            columns=factor_df.columns
        )
        
        # 向量化优化：对每行排序并赋值
        for dt in factor_df.index:
            row = factor_df.loc[dt]
            row = row.dropna()
            
            if len(row) < 4:
                continue
            
            row_sorted = row.sort_values()
            choose = max(1, int(np.floor(trading_signal * len(row_sorted))))
            
            if long_low:
                long_symbols = row_sorted.index[:choose]
                short_symbols = row_sorted.index[-choose:]
            else:
                long_symbols = row_sorted.index[-choose:]
                short_symbols = row_sorted.index[:choose]
            
            signal_df.loc[dt, long_symbols] = 1.0
            signal_df.loc[dt, short_symbols] = -1.0
        
        return signal_df
    
    def _calculate_metrics(
        self,
        factor_name: str,
        net_return: pd.Series,
        gross_return: pd.Series,
        turnover_ratio: pd.Series,
        cost: pd.Series,
        nav: pd.Series,
        **params
    ) -> BacktestResult:
        """计算绩效指标"""
        
        # 过滤掉前段的 warmup 期（避免持仓未满时的失真）
        valid = net_return.notna()
        net_return = net_return[valid]
        gross_return = gross_return[valid]
        turnover_ratio = turnover_ratio[valid]
        cost = cost[valid]
        nav = nav[valid]
        
        total_days = len(net_return)
        profit_days = int((net_return > 0).sum())
        loss_days = int((net_return < 0).sum())
        
        # 年化指标
        avg_daily_return = net_return.mean()
        annual_return = avg_daily_return * TRADING_DAYS_PER_YEAR
        annual_vol = net_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        sharpe = annual_return / annual_vol if annual_vol > 1e-12 else 0.0
        
        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_dd = drawdown.min()
        
        calmar = -annual_return / max_dd if max_dd < -1e-12 else np.inf
        
        win_rate = profit_days / total_days if total_days > 0 else 0.0
        avg_turnover = turnover_ratio.mean()
        
        meta = self.registry.get(factor_name)
        
        return BacktestResult(
            factor_name=factor_name,
            factor_meta=meta,
            **params,
            annual_return=annual_return,
            annual_vol=annual_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            avg_daily_return=avg_daily_return,
            avg_turnover=avg_turnover,
            nav_series=nav,
            net_return_series=net_return,
            gross_return_series=gross_return,
            turnover_series=turnover_ratio,
            cost_series=cost,
            total_days=total_days,
            profit_days=profit_days,
            loss_days=loss_days,
        )
    
    # =================================================================
    # 批量接口
    # =================================================================
    
    def run_all(
        self,
        factor_names: Optional[List[str]] = None,
        holding_period: int = 5,
        trading_signal: float = 0.2,
        aggregation: str = "sum",
        leverage: float = 2.0,
        commission: float = 0.0001,
    ) -> Dict[str, BacktestResult]:
        """
        批量回测多个因子
        
        返回: {factor_name: BacktestResult}
        """
        if factor_names is None:
            factor_names = sorted([f.stem.split('_')[0] for f in self.data_dir.glob("*.parquet")
                                   if not f.stem.startswith("factor_results")])
            # 更精确的提取：去掉日期后缀
            factor_names = self._extract_factor_names()
        
        results = {}
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"批量回测开始 — 共 {len(factor_names)} 个因子")
            print("=" * 70)
        
        for i, name in enumerate(factor_names, 1):
            if self.verbose:
                print(f"\n[{i}/{len(factor_names)}] 回测因子: {name}")
            
            try:
                result = self.run_factor(
                    factor_name=name,
                    holding_period=holding_period,
                    trading_signal=trading_signal,
                    aggregation=aggregation,
                    leverage=leverage,
                    commission=commission,
                )
                
                if result:
                    results[name] = result
                    if self.verbose:
                        print(f"       年化收益: {result.annual_return:+.2%}  "
                              f"夏普: {result.sharpe_ratio:+.2f}  "
                              f"最大回撤: {result.max_drawdown:.2%}  "
                              f"Calmar: {result.calmar_ratio:+.2f}")
                else:
                    if self.verbose:
                        print(f"       [SKIP] 回测失败或数据不足")
            except Exception as e:
                if self.verbose:
                    print(f"       [ERROR] {e}")
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"批量回测完成 — 成功 {len(results)}/{len(factor_names)} 个因子")
            print("=" * 70)
        
        return results
    
    def _extract_factor_names(self) -> List[str]:
        """从 parquet 文件名中提取因子名称"""
        names = set()
        for fpath in self.data_dir.glob("*_*.parquet"):
            if fpath.name.startswith("factor_results"):
                continue
            stem = fpath.stem
            # 去掉日期后缀 _YYYY-MM-DD_YYYY-MM-DD
            parts = stem.split('_')
            if len(parts) >= 3 and parts[-2].count('-') == 2 and parts[-1].count('-') == 2:
                name = '_'.join(parts[:-2])
            else:
                name = stem
            names.add(name)
        return sorted(names)
    
    def compare_results(
        self,
        results: Dict[str, BacktestResult],
        sort_by: str = "sharpe_ratio",
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        将多个回测结果汇总为对比表
        
        参数:
            results: run_all() 的返回值
            sort_by: 排序列名
            ascending: 是否升序
        """
        rows = []
        for name, res in results.items():
            d = res.to_dict()
            rows.append(d)
        
        df = pd.DataFrame(rows)
        
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        return df.reset_index(drop=True)


# =============================================================================
# 快捷函数
# =============================================================================

def quick_backtest(
    factor_name: str,
    dominant_symbols: Optional[List[str]] = None,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    **kwargs
) -> Optional[BacktestResult]:
    """快速回测单个因子的便捷函数"""
    if dominant_symbols is None:
        # 默认品种池
        dominant_symbols = [
            'A88.DCE', 'AG88.SHFE', 'AL88.SHFE', 'AP88.CZCE', 'AU88.SHFE',
            'B88.DCE', 'BU88.SHFE', 'C88.DCE', 'CF88.CZCE', 'CJ88.CZCE',
            'CS88.DCE', 'CU88.SHFE', 'CY88.CZCE', 'EB88.DCE', 'EG88.DCE',
            'FG88.CZCE', 'FU88.SHFE', 'HC88.SHFE', 'I88.DCE', 'J88.DCE',
            'JD88.DCE', 'JM88.DCE', 'L88.DCE', 'LH88.DCE', 'LU88.INE',
            'M88.DCE', 'MA88.CZCE', 'NI88.SHFE', 'OI88.CZCE', 'P88.DCE',
            'PB88.SHFE', 'PF88.CZCE', 'PG88.DCE', 'PK88.CZCE', 'PP88.DCE',
            'RB88.SHFE', 'RM88.CZCE', 'RU88.SHFE', 'SA88.CZCE', 'SC88.INE',
            'SF88.CZCE', 'SI88.GFEX', 'SM88.CZCE', 'SN88.SHFE', 'SP88.SHFE',
            'SR88.CZCE', 'SS88.SHFE', 'UR88.CZCE', 'V88.DCE', 'Y88.DCE',
            'ZN88.SHFE'
        ]
    
    engine = BatchBacktestEngine(
        dominant_symbols=dominant_symbols,
        start=start,
        end=end,
        verbose=True
    )
    
    return engine.run_factor(factor_name, **kwargs)


if __name__ == "__main__":
    # 快速测试 carry_ret 因子
    result = quick_backtest("carry_ret", holding_period=5, trading_signal=0.2)
    if result:
        print("\n【carry_ret 回测结果】")
        print(f"  年化收益: {result.annual_return:+.2%}")
        print(f"  年化波动: {result.annual_vol:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:+.2f}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  Calmar:   {result.calmar_ratio:+.2f}")
        print(f"  胜率:     {result.win_rate:.1%}")
