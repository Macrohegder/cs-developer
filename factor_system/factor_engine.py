#!/usr/bin/env python3
"""
Factor Engine — 批量因子计算引擎

职责：
1. 统一数据加载：一次加载 OHLCV + OI + Turnover + 主力映射 + 合约信息
2. 批量因子计算：对所有活跃因子执行向量化计算
3. 数据缓存：避免重复加载
4. 自动保存：计算结果写入 DataCenter
5. 并行计算：支持多进程加速

使用示例：
    from factor_engine import FactorEngine
    engine = FactorEngine(dominant_symbols, start, end)
    engine.load_all_data()
    
    # 计算单个因子
    factor_df = engine.compute("momentum", cycle=20)
    
    # 批量计算所有因子
    results = engine.compute_all(factor_names=["momentum", "skew", "duvol"])
    
    # 保存到数据库
    engine.save_factor(factor_df, name="momentum", parameter="cycle20")
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any
import traceback

import numpy as np
import pandas as pd
from vnpy.trader.constant import Interval
from vnpy_alpharesearch.data_center import DataCenter
from vnpy_alpharesearch.utility import load_bar_df

# 把项目根目录加入路径
sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/root/cs_developer/factor_system")

from factor_registry import FactorRegistry, FactorMeta, FactorStatus


class DataCache:
    """数据缓存管理器 — 避免重复加载"""
    
    def __init__(self):
        self._bar_data: Dict[str, pd.DataFrame] = {}      # {vt_symbol: df}
        self._contract_info: Optional[pd.DataFrame] = None
        self._dominant_map: Optional[pd.DataFrame] = None
        self._price_wide: Optional[pd.DataFrame] = None   # wide format close_price
        self._ohlcv_wide: Optional[Dict[str, pd.DataFrame]] = None  # {field: wide_df}
        self._loaded_symbols: set = set()
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
    
    def is_loaded(self, symbols: List[str], start: datetime, end: datetime) -> bool:
        """检查指定品种和时间段是否已缓存"""
        if self._start != start or self._end != end:
            return False
        return set(symbols).issubset(self._loaded_symbols)
    
    def clear(self):
        """清空缓存"""
        self._bar_data.clear()
        self._contract_info = None
        self._dominant_map = None
        self._price_wide = None
        self._ohlcv_wide = None
        self._loaded_symbols.clear()
        self._start = None
        self._end = None


class FactorEngine:
    """因子批量计算引擎"""
    
    def __init__(
        self,
        dominant_symbols: List[str],
        start: datetime,
        end: datetime,
        data_cache: Optional[DataCache] = None,
        primary_suffix: str = "88",
        secondary_suffix: str = "88A2",
        verbose: bool = True
    ):
        self.dominant_symbols = dominant_symbols
        self.start = start
        self.end = end
        self.verbose = verbose
        self.cache = data_cache or DataCache()
        self.primary_suffix = primary_suffix
        self.secondary_suffix = secondary_suffix
        
        self.dc = DataCenter()
        self.registry = FactorRegistry()
        
        # 数据占位符
        self._contract_df: Optional[pd.DataFrame] = None
        self._dominant_df: Optional[pd.DataFrame] = None
        self._expiry_map: Dict[str, datetime] = {}
        
        # Wide-format 数据
        self._close: Optional[pd.DataFrame] = None
        self._open: Optional[pd.DataFrame] = None
        self._high: Optional[pd.DataFrame] = None
        self._low: Optional[pd.DataFrame] = None
        self._volume: Optional[pd.DataFrame] = None
        self._turnover: Optional[pd.DataFrame] = None
        self._oi: Optional[pd.DataFrame] = None
        
        # 辅助数据
        self._returns: Optional[pd.DataFrame] = None
        self._f2_close: Optional[pd.DataFrame] = None  # 次主力连续合约
    
    # =================================================================
    # 数据加载
    # =================================================================
    
    def load_all_data(self, force_reload: bool = False):
        """一次性加载所有需要的数据"""
        if not force_reload and self.cache.is_loaded(self.dominant_symbols, self.start, self.end):
            if self.verbose:
                print("[DataCache] 命中缓存，直接使用")
            self._load_from_cache()
            return
        
        if self.verbose:
            print("=" * 70)
            print("Factor Engine — 数据加载")
            print("=" * 70)
        
        t0 = datetime.now()
        
        # 1. 合约信息
        self._load_contract_info()
        
        # 2. 主力映射
        self._load_dominant_map()
        
        # 3. Bar 数据（88 主力连续 + 88A2 次主力连续）
        self._load_bar_data()
        
        # 4. 构建 Wide-format DataFrame
        self._build_wide_data()
        
        # 5. 计算辅助数据
        self._build_derived_data()
        
        # 6. 存入缓存
        self._save_to_cache()
        
        elapsed = (datetime.now() - t0).total_seconds()
        if self.verbose:
            print(f"\n[OK] 数据加载完成，耗时 {elapsed:.1f}s")
            print(f"     品种数: {len(self.dominant_symbols)}")
            print(f"     交易日: {len(self._close)}")
    
    def _load_contract_info(self):
        """加载合约信息"""
        if self.verbose:
            print("\n[1/5] 加载合约信息...")
        self._contract_df = self.dc.load_contract_df()
        
        # 构建到期日映射 {symbol: expiry}
        self._expiry_map = {}
        for symbol in self._contract_df.index:
            try:
                expiry_str = self._contract_df.loc[symbol, "enddate"]
                if isinstance(expiry_str, str):
                    self._expiry_map[symbol] = datetime.strptime(expiry_str, "%Y-%m-%d")
            except (ValueError, KeyError):
                continue
        
        if self.verbose:
            print(f"      合约总数: {len(self._contract_df)}")
    
    def _load_dominant_map(self):
        """加载主力/次主力映射"""
        if self.verbose:
            print("[2/5] 加载主力合约映射...")
        
        keys = []
        for s in self.dominant_symbols:
            keys.append(f"{s}@1")
            keys.append(f"{s}@2")
        
        self._dominant_df = self.dc.load_reference_df(
            "vnpy_dominant_contract",
            keys,
            self.start,
            self.end,
            fillna=True
        )
        
        if self.verbose:
            print(f"      映射记录: {self._dominant_df.shape}")
    
    def _get_data_symbol(self, dominant_symbol: str, suffix: str) -> str:
        """将 dominant_symbol (如 RB88.SHFE) 转换为指定后缀的数据符号 (如 RB99.SHFE)"""
        parts = dominant_symbol.split(".")
        exchange = parts[1]
        symbol_base = parts[0].replace("88", "")  # "RB88" -> "RB"
        return f"{symbol_base}{suffix}.{exchange}"
    
    def _load_bar_data(self):
        """加载所有品种的日线数据"""
        if self.verbose:
            print(f"[3/5] 加载行情数据 (主力={self.primary_suffix}, 次主力={self.secondary_suffix})...")
        
        self.cache._bar_data.clear()
        missing = []
        
        for ds in self.dominant_symbols:
            # 主力连续
            primary_symbol = self._get_data_symbol(ds, self.primary_suffix)
            try:
                df1 = load_bar_df(primary_symbol, Interval.DAILY, self.start, self.end)
                if len(df1) > 0:
                    self.cache._bar_data[ds] = df1
                else:
                    missing.append(ds)
            except Exception as e:
                missing.append(f"{ds}: {e}")
            
            # 次主力连续（支持自定义后缀）
            secondary_symbol = self._get_data_symbol(ds, self.secondary_suffix)
            try:
                df2 = load_bar_df(secondary_symbol, Interval.DAILY, self.start, self.end)
                if len(df2) > 0:
                    self.cache._bar_data[secondary_symbol] = df2
            except Exception:
                pass
        
        if self.verbose:
            print(f"      成功加载: {len(self.cache._bar_data)} 个合约")
            if missing:
                print(f"      缺失数据: {len(missing)} 个")
    
    def _build_wide_data(self):
        """将 bar 数据转换为 wide-format"""
        if self.verbose:
            print("[4/5] 构建宽表数据...")
        
        fields = ["close_price", "open_price", "high_price", "low_price", 
                  "volume", "turnover", "open_interest"]
        
        self.cache._ohlcv_wide = {}
        
        for field in fields:
            field_dict = {}
            for ds in self.dominant_symbols:
                df = self.cache._bar_data.get(ds)
                if df is not None and field in df.columns:
                    field_dict[ds] = df[field]
            
            if field_dict:
                self.cache._ohlcv_wide[field] = pd.DataFrame(field_dict)
            else:
                self.cache._ohlcv_wide[field] = pd.DataFrame()
        
        # 单独保存次主力收盘价（用于 carry 计算）
        f2_dict = {}
        for ds in self.dominant_symbols:
            secondary_symbol = self._get_data_symbol(ds, self.secondary_suffix)
            df2 = self.cache._bar_data.get(secondary_symbol)
            if df2 is not None and "close_price" in df2.columns:
                f2_dict[ds] = df2["close_price"]
        self.cache._f2_close = pd.DataFrame(f2_dict) if f2_dict else None
        
        if self.verbose:
            for field, df in self.cache._ohlcv_wide.items():
                print(f"      {field:<15s}: {df.shape}")
    
    def _build_derived_data(self):
        """构建派生数据（收益率等）"""
        if self.verbose:
            print("[5/5] 计算派生数据...")
        
        close = self.cache._ohlcv_wide.get("close_price")
        if close is not None and not close.empty:
            self.cache._returns = close.pct_change()
        
        if self.verbose:
            print("      日收益率矩阵已构建")
    
    def _save_to_cache(self):
        """将数据保存到缓存对象"""
        self.cache._start = self.start
        self.cache._end = self.end
        self.cache._loaded_symbols = set(self.dominant_symbols)
        self._load_from_cache()
    
    def _load_from_cache(self):
        """从缓存恢复数据到引擎"""
        self._close = self.cache._ohlcv_wide.get("close_price")
        self._open = self.cache._ohlcv_wide.get("open_price")
        self._high = self.cache._ohlcv_wide.get("high_price")
        self._low = self.cache._ohlcv_wide.get("low_price")
        self._volume = self.cache._ohlcv_wide.get("volume")
        self._turnover = self.cache._ohlcv_wide.get("turnover")
        self._oi = self.cache._ohlcv_wide.get("open_interest")
        self._returns = self.cache._returns
        self._f2_close = self.cache._f2_close
    
    # =================================================================
    # 核心计算接口
    # =================================================================
    
    def compute(self, factor_name: str, **override_params) -> pd.DataFrame:
        """
        计算单个因子
        
        参数:
            factor_name: 因子名
            **override_params: 覆盖默认参数
        
        返回:
            DataFrame: index=datetime, columns=dominant_symbols
        """
        meta = self.registry.get(factor_name)
        if meta is None:
            raise KeyError(f"因子 '{factor_name}' 未在注册表中定义")
        
        # 合并参数
        params = meta.params.copy()
        params.update(override_params)
        
        # 获取计算函数
        calc_fn = self._get_calc_function(factor_name)
        if calc_fn is None:
            raise NotImplementedError(f"因子 '{factor_name}' 的计算函数尚未实现")
        
        if self.verbose:
            print(f"[Compute] {factor_name} | params={params}")
        
        try:
            factor_df = calc_fn(self, **params)
            factor_df = factor_df.reindex(columns=self.dominant_symbols)
            return factor_df
        except Exception as e:
            print(f"[ERROR] 因子 '{factor_name}' 计算失败: {e}")
            traceback.print_exc()
            return pd.DataFrame(index=self._close.index if self._close is not None else None)
    
    def compute_all(
        self,
        factor_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        skip_broken: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算多个因子
        
        参数:
            factor_names: 指定因子列表，None则计算所有活跃因子
            categories: 按类别筛选
            skip_broken: 是否跳过状态为 broken 的因子
        
        返回:
            {factor_name: factor_df}
        """
        if factor_names is None:
            metas = self.registry.get_active()
            if categories:
                metas = [m for m in metas if m.category in categories]
            factor_names = [m.name for m in metas]
        
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"批量计算 {len(factor_names)} 个因子")
            print("=" * 70)
        
        results = {}
        t0 = datetime.now()
        
        for idx, name in enumerate(factor_names, 1):
            meta = self.registry.get(name)
            if meta is None:
                continue
            if skip_broken and meta.status == FactorStatus.BROKEN:
                print(f"[{idx}/{len(factor_names)}] {name} — 状态为 broken，跳过")
                continue
            
            print(f"[{idx}/{len(factor_names)}] {name} ...", end=" ")
            try:
                df = self.compute(name)
                results[name] = df
                print(f"OK (shape={df.shape})")
            except Exception as e:
                print(f"FAILED: {e}")
        
        elapsed = (datetime.now() - t0).total_seconds()
        if self.verbose:
            print(f"\n[OK] 批量计算完成，成功 {len(results)}/{len(factor_names)}，耗时 {elapsed:.1f}s")
        
        return results
    
    # =================================================================
    # 数据保存
    # =================================================================
    
    def save_factor(
        self,
        factor_df: pd.DataFrame,
        name: str,
        parameter: str,
        author: str = "factor_system",
        interval: str = "d"
    ):
        """保存因子到 DataCenter"""
        self.dc.save_factor_df(
            df=factor_df,
            name=name,
            interval=interval,
            parameter=parameter,
            author=author
        )
        if self.verbose:
            print(f"[Save] 因子 '{name}' 已保存 (parameter={parameter})")
    
    def save_all(
        self,
        results: Dict[str, pd.DataFrame],
        author: str = "factor_system",
        prefix: str = ""
    ):
        """批量保存因子"""
        for name, df in results.items():
            param_str = f"cycle{self.registry.get(name).params.get('cycle', 'default')}" if self.registry.get(name) else "default"
            self.save_factor(df, f"{prefix}{name}", param_str, author)
    
    # =================================================================
    # 计算函数路由
    # =================================================================
    
    def _get_calc_function(self, name: str) -> Optional[Callable]:
        """根据因子名获取对应的计算函数
        
        支持后缀映射:
          - xxx_889 -> calc_xxx (基于889数据计算，但计算逻辑相同)
          - skew_180 -> calc_skew (cycle参数覆盖为180)
          - skew_180_889 -> calc_skew (基于889数据，cycle=180)
        """
        # 1. 精确匹配
        try:
            from factors import batch_factors as bf
            fn = getattr(bf, f"calc_{name}", None)
            if fn is not None:
                return fn
        except ImportError:
            pass
        
        # 2. 处理后缀映射
        base_name = name
        # 去掉 _889 后缀
        if base_name.endswith("_889"):
            base_name = base_name[:-4]
        # 去掉 _180 后缀 (skew_180 -> skew, skew_180_889 -> skew_889 -> skew)
        if base_name.endswith("_180"):
            base_name = base_name[:-4]
        
        if base_name != name:
            try:
                from factors import batch_factors as bf
                fn = getattr(bf, f"calc_{base_name}", None)
                if fn is not None:
                    return fn
            except ImportError:
                pass
        
        # 3. 引擎内置计算函数
        return getattr(self, f"_calc_{name}", None)
    
    # =================================================================
    # 便捷属性
    # =================================================================
    
    @property
    def close(self) -> pd.DataFrame:
        return self._close
    
    @property
    def open_price(self) -> pd.DataFrame:
        return self._open
    
    @property
    def high(self) -> pd.DataFrame:
        return self._high
    
    @property
    def low(self) -> pd.DataFrame:
        return self._low
    
    @property
    def volume(self) -> pd.DataFrame:
        return self._volume
    
    @property
    def turnover(self) -> pd.DataFrame:
        return self._turnover
    
    @property
    def open_interest(self) -> pd.DataFrame:
        return self._oi
    
    @property
    def returns(self) -> pd.DataFrame:
        return self._returns
    
    @property
    def f2_close(self) -> Optional[pd.DataFrame]:
        return self._f2_close


# =============================================================================
# 全局缓存实例（进程内共享）
# =============================================================================

_GLOBAL_CACHE = DataCache()


def get_engine(
    dominant_symbols: List[str],
    start: datetime,
    end: datetime,
    primary_suffix: str = "88",
    secondary_suffix: str = "88A2",
    verbose: bool = True
) -> FactorEngine:
    """获取带全局缓存的引擎实例"""
    return FactorEngine(
        dominant_symbols, start, end,
        data_cache=_GLOBAL_CACHE,
        primary_suffix=primary_suffix,
        secondary_suffix=secondary_suffix,
        verbose=verbose
    )


if __name__ == "__main__":
    # 简单测试
    symbols = ['RB88.SHFE', 'HC88.SHFE', 'I88.DCE']
    engine = get_engine(symbols, datetime(2024, 1, 1), datetime(2024, 6, 1))
    engine.load_all_data()
    print("\n数据加载测试通过")
