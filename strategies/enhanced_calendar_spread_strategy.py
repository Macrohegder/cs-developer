"""
Enhanced Calendar Spread Strategy（增强版日历价差套利策略）

在 CalendarSpreadStrategy 基础上增加：
- 可配置波动窗口
- 止盈 / 止损
- 只做季月合约（3/6/9/12）
- 提前 N 天移仓（基于近月到期日）
- 支持单品种或多品种

在 vnpy_alpharesearch 的 StrategyBacktester 框架下运行。
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from math import floor

import numpy as np
from pandas import DataFrame, Series, Timestamp
from vnpy.trader.constant import Interval

from vnpy_alpharesearch.strategy import StrategyTemplate


class EnhancedCalendarSpreadStrategy(StrategyTemplate):
    """增强版股指期货日历价差套利策略"""

    # --- 核心参数 ---
    products: Union[str, List[str]] = "IM"
    vol_threshold: Union[float, Dict[str, float]] = 0.15  # 年化波动阈值；可统一或按品种设置
    vol_window: int = 20                # 波动计算窗口
    leverage: float = 2.0               # 名义杠杆
    capital_ratio: float = 1.0          # 每日资金使用比例

    # 套利方向：+1 = 多近空远（默认），-1 = 空近多远
    # 可统一或按品种设置，例如 {"IC": 1, "IM": -1}
    direction: Union[int, Dict[str, int]] = 1

    # 动态方向切换：基于近远月价差 spread_pct 的滚动 z-score 自动切换套利方向
    dynamic_direction: bool = False               # 是否开启动态方向切换
    spread_lookback: Union[int, Dict[str, int]] = 60      # 价差滚动统计窗口（可统一或按品种）
    spread_z_threshold: Union[float, Dict[str, float]] = 1.0  # z-score 阈值（可统一或按品种）
    buy_the_dip_mode: bool = False               # 是否开启 long-only 抄底模式
    dip_entry_z: Union[float, Dict[str, float]] = -1.5      # z-score 低于该值才允许开仓
    dip_exit_z: Union[float, Dict[str, float]] = -0.25      # z-score 回到该值上方平仓
    regime_lookback: Union[int, Dict[str, int]] = 90        # 慢速 regime 均值窗口
    require_positive_regime: bool = True         # 仅在正 carry regime 中允许开仓

    # 止盈止损（以该 pair 名义市值为基准的比例）
    stop_loss_pct: float = 0.0          # 止损比例，0 表示不止损
    take_profit_pct: float = 0.0        # 止盈比例，0 表示不止盈

    # 季月过滤
    only_quarterly: bool = False        # 是否只交易季月合约

    # 季度合约对模式：为 True 时忽略 dominant @1/@2，自动选择“最近季月 vs 下一季月”
    use_quarterly_pair: bool = False

    # 提前移仓
    roll_before_days: int = 0           # 近月到期前 N 天平仓，0 表示不提前移仓

    # 季节性月份过滤：屏蔽月份平仓观望（空列表表示不过滤）
    blocked_months: List[int] = []

    # 近远月合约月份跨度上限（超过则视为映射异常不开仓）
    # 股指 @1/@2 相邻季月跨度 <= 3；商品主力多为 1/5/9 月轮换，跨度常为 4
    max_month_span: int = 3

    # --- 外部注入数据 ---
    mapping: Union[DataFrame, Dict[str, DataFrame]] = None
    idx_close: Union[Series, Dict[str, Series]] = None   # 88 指数收盘价（计算波动）
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

        # 统一 vol_threshold 为按品种字典
        if isinstance(self.vol_threshold, (int, float)):
            self._vol_threshold = {p: float(self.vol_threshold) for p in self._products}
        else:
            self._vol_threshold = {p: float(self.vol_threshold.get(p, 0.15)) for p in self._products}

        # 统一 direction 为按品种字典
        if isinstance(self.direction, (int, float)):
            self._direction = {p: int(self.direction) for p in self._products}
        else:
            self._direction = {p: int(self.direction.get(p, 1)) for p in self._products}

        # 统一动态方向参数为按品种字典
        if isinstance(self.spread_lookback, (int, float)):
            self._spread_lookback = {p: int(self.spread_lookback) for p in self._products}
        else:
            self._spread_lookback = {p: int(self.spread_lookback.get(p, 60)) for p in self._products}

        if isinstance(self.spread_z_threshold, (int, float)):
            self._spread_z_threshold = {p: float(self.spread_z_threshold) for p in self._products}
        else:
            self._spread_z_threshold = {p: float(self.spread_z_threshold.get(p, 1.0)) for p in self._products}

        if isinstance(self.dip_entry_z, (int, float)):
            self._dip_entry_z = {p: float(self.dip_entry_z) for p in self._products}
        else:
            self._dip_entry_z = {p: float(self.dip_entry_z.get(p, -1.5)) for p in self._products}

        if isinstance(self.dip_exit_z, (int, float)):
            self._dip_exit_z = {p: float(self.dip_exit_z) for p in self._products}
        else:
            self._dip_exit_z = {p: float(self.dip_exit_z.get(p, -0.25)) for p in self._products}

        if isinstance(self.regime_lookback, (int, float)):
            self._regime_lookback = {p: int(self.regime_lookback) for p in self._products}
        else:
            self._regime_lookback = {p: int(self.regime_lookback.get(p, 90)) for p in self._products}

        # 统一外部数据为按品种字典
        if isinstance(self.mapping, DataFrame):
            self._mapping = {self._products[0]: self.mapping}
        else:
            self._mapping = dict(self.mapping)

        if isinstance(self.idx_close, Series):
            self._idx_close = {self._products[0]: self.idx_close}
        else:
            self._idx_close = dict(self.idx_close)

        # 计算波动率序列
        self._idx_vol: Dict[str, Series] = {}
        for p in self._products:
            close = self._idx_close.get(p)
            if close is not None:
                ret = close.pct_change()
                self._idx_vol[p] = ret.rolling(window=self.vol_window, min_periods=self.vol_window//2).std() * np.sqrt(250)
            else:
                self._idx_vol[p] = None

        # 构建每个品种的近月/远月合约对（支持 dominant 映射或季度合约对）
        self._pair_df: Dict[str, DataFrame] = {}
        for p in self._products:
            self._pair_df[p] = self._build_pair_df(p)

        # 预计算近远月价差 spread_pct 的滚动统计（用于动态切换和抄底模式）
        self._spread_z: Dict[str, Series] = {}
        self._spread_pct: Dict[str, Series] = {}
        self._spread_regime_mean: Dict[str, Series] = {}
        if (self.dynamic_direction or self.buy_the_dip_mode) and self.history_df is not None:
            close_df = self.history_df.xs("close_price", axis=1, level=1)
            close_stack = close_df.stack()
            for p in self._products:
                pair_df = self._pair_df.get(p)
                if pair_df is None or pair_df.empty:
                    continue
                # 把近月/远月合约映射到具体收盘价
                near_idx = pd.MultiIndex.from_arrays([pair_df.index, pair_df["near"]])
                far_idx = pd.MultiIndex.from_arrays([pair_df.index, pair_df["far"]])
                near_price = close_stack.reindex(near_idx).values
                far_price = close_stack.reindex(far_idx).values
                spread_pct = pd.Series(
                    (near_price - far_price) / near_price,
                    index=pair_df.index,
                    name=f"{p}_spread_pct",
                )
                self._spread_pct[p] = spread_pct
                lookback = self._spread_lookback[p]
                mean = spread_pct.rolling(
                    window=lookback,
                    min_periods=lookback // 2,
                ).mean()
                std = spread_pct.rolling(
                    window=lookback,
                    min_periods=lookback // 2,
                ).std()
                z = (spread_pct - mean) / std
                self._spread_z[p] = z
                regime_lookback = self._regime_lookback[p]
                self._spread_regime_mean[p] = spread_pct.rolling(
                    window=regime_lookback,
                    min_periods=max(5, regime_lookback // 2),
                ).mean()

        # 每个品种独立状态
        self._current_position: Dict[str, Dict[str, int]] = {p: {} for p in self._products}
        self._current_pair: Dict[str, Optional[Tuple[str, str]]] = {p: None for p in self._products}
        self._current_lots: Dict[str, int] = {p: 0 for p in self._products}
        # 当前持仓方向（用于动态方向切换时识别是否需要平仓翻转）
        self._current_direction: Dict[str, int] = {p: self._direction[p] for p in self._products}
        # 开仓时的价格（用于止盈止损）
        self._entry_prices: Dict[str, Dict[str, float]] = {p: {} for p in self._products}
        # 开仓时的名义市值
        self._entry_nominal: Dict[str, float] = {p: 0.0 for p in self._products}

        # 资金按品种均分
        self._daily_target_nominal = self.capital * self.leverage * self.capital_ratio / len(self._products)

    def calculate_target(self, df: DataFrame) -> dict:
        """计算当前目标仓位"""
        dt = self.current_dt
        target: Dict[str, int] = {}

        for product in self._products:
            product_target = self._calculate_product_target(product, dt)
            for vt_symbol, lots in product_target.items():
                target[vt_symbol] = target.get(vt_symbol, 0) + lots

        return target

    def _calculate_product_target(self, product: str, dt: Timestamp) -> Dict[str, int]:
        """单个品种的目标仓位"""
        pair_df = self._pair_df.get(product)
        if pair_df is None or pair_df.empty:
            return self._clear_position(product)

        if dt not in pair_df.index:
            return self._clear_position(product)

        pair_row = pair_df.loc[dt]
        near = pair_row["near"] if pd.notna(pair_row["near"]) else None
        far = pair_row["far"] if pd.notna(pair_row["far"]) else None

        # 数据清洗：缺失或近远相同
        if near is None or far is None or near == far:
            return self._clear_position(product)

        # 数据清洗：近月/远月当日无有效收盘价（合约未入库、未上市或已摘牌）
        # 避免持有无行情数据合约的"幽灵仓位"导致绩效引擎加载失败
        if not self._has_valid_price(near, dt) or not self._has_valid_price(far, dt):
            return self._clear_position(product)

        # 数据清洗：月份跨度超过上限
        near_month = self._get_expiry_month(near)
        far_month = self._get_expiry_month(far)
        span = self._month_span(near_month, far_month)
        if span <= 0 or span > self.max_month_span:
            return self._clear_position(product)

        # 季月过滤
        if self.only_quarterly:
            if near_month[1] not in (3, 6, 9, 12) or far_month[1] not in (3, 6, 9, 12):
                return self._clear_position(product)

        # 波动过滤（按品种）
        product_vol_threshold = self._vol_threshold.get(product, 0.15)
        if product_vol_threshold > 0:
            vol = self._idx_vol.get(product)
            v = vol.get(dt, float('nan')) if vol is not None else float('nan')
            signal_active = (v >= product_vol_threshold) if pd.notna(v) else False
            if not signal_active:
                return self._clear_position(product)

        # 季节性月份过滤：屏蔽月份平仓观望
        if self.blocked_months and dt.month in self.blocked_months:
            return self._clear_position(product)

        if self.buy_the_dip_mode:
            self._direction[product] = 1
            return self._calculate_buy_dip_target(product, dt, near, far)

        # 动态方向切换：基于近远月价差 z-score 决定当日套利方向
        signal_direction = self._get_signal_direction(product, dt)
        if self.dynamic_direction:
            self._direction[product] = signal_direction

        current_pair = self._current_pair[product]

        # 若当前持仓方向与信号方向相反，先平仓并在本日按新方向重新开仓
        if current_pair is not None and self.dynamic_direction:
            if self._current_direction.get(product, 1) != signal_direction:
                self._clear_position(product)
                current_pair = None

        # 如果有持仓，检查止盈止损 / 提前移仓
        if current_pair is not None:
            # 提前移仓检查
            if self.roll_before_days > 0:
                near_expiry = self._get_expiry_date(current_pair[0])
                if near_expiry and (near_expiry - dt).days <= self.roll_before_days:
                    return self._clear_position(product)

            # 止盈止损检查（仅当当前 pair 未变化时）
            if current_pair == (near, far):
                if self._check_stop_loss_take_profit(product, dt):
                    return self._clear_position(product)

        # 信号活跃：决定目标仓位
        target_pair = (near, far)

        if current_pair is None:
            lots = self._calculate_lots(product, near, dt)
            if lots < 1:
                return self._clear_position(product)
            self._open_position(product, target_pair, lots, dt)
            return self._to_target_dict(product)

        if current_pair != target_pair:
            # 换月：平旧对，开新对
            lots = self._calculate_lots(product, near, dt)
            if lots < 1:
                return self._clear_position(product)
            self._open_position(product, target_pair, lots, dt)
            return self._to_target_dict(product)

        # 持仓不变
        return self._to_target_dict(product)

    def _calculate_buy_dip_target(
        self,
        product: str,
        dt: Timestamp,
        near: str,
        far: str,
    ) -> Dict[str, int]:
        """long-only 抄底模式：仅在价差回落到低位时做多近月、做空远月。"""
        current_pair = self._current_pair[product]
        target_pair = (near, far)

        if current_pair is not None:
            if self.roll_before_days > 0:
                near_expiry = self._get_expiry_date(current_pair[0])
                if near_expiry and (near_expiry - dt).days <= self.roll_before_days:
                    return self._clear_position(product)

            if current_pair == target_pair and self._check_stop_loss_take_profit(product, dt):
                return self._clear_position(product)

            if self._should_exit_buy_dip(product, dt):
                return self._clear_position(product)

        if current_pair is None:
            if not self._should_enter_buy_dip(product, dt):
                return self._clear_position(product)
            lots = self._calculate_lots(product, near, dt)
            if lots < 1:
                return self._clear_position(product)
            self._open_position(product, target_pair, lots, dt)
            return self._to_target_dict(product)

        if current_pair != target_pair:
            if not self._should_enter_buy_dip(product, dt):
                return self._clear_position(product)
            lots = self._calculate_lots(product, near, dt)
            if lots < 1:
                return self._clear_position(product)
            self._open_position(product, target_pair, lots, dt)
            return self._to_target_dict(product)

        return self._to_target_dict(product)

    def _check_stop_loss_take_profit(self, product: str, dt: Timestamp) -> bool:
        """检查是否触发止盈止损"""
        if self.stop_loss_pct <= 0 and self.take_profit_pct <= 0:
            return False

        position = self._current_position.get(product, {})
        entry_prices = self._entry_prices.get(product, {})
        if not position or not entry_prices:
            return False

        total_pnl = 0.0
        for vt_symbol, lots in position.items():
            entry_price = entry_prices.get(vt_symbol)
            if entry_price is None:
                continue
            try:
                curr_price = self.history_df.loc[dt, (vt_symbol, "close_price")]
            except Exception:
                continue
            if pd.isna(curr_price):
                continue
            multiplier = self._get_multiplier(product, vt_symbol)
            total_pnl += lots * (curr_price - entry_price) * multiplier

        nominal = self._entry_nominal.get(product, 0)
        if nominal <= 0:
            return False

        pnl_pct = total_pnl / nominal
        if self.stop_loss_pct > 0 and pnl_pct <= -self.stop_loss_pct:
            return True
        if self.take_profit_pct > 0 and pnl_pct >= self.take_profit_pct:
            return True
        return False

    def _calculate_lots(self, product: str, near_vt_symbol: str, dt: Timestamp) -> int:
        """基于近月价格计算目标手数"""
        try:
            close_price = self.history_df.loc[dt, (near_vt_symbol, "close_price")]
        except Exception:
            return 0

        if pd.isna(close_price) or close_price <= 0:
            return 0

        multiplier = self._get_multiplier(product, near_vt_symbol)
        lots = int(floor(self._daily_target_nominal / 2 / close_price / multiplier))
        return lots if lots >= 1 else 0

    def _open_position(self, product: str, pair: Tuple[str, str], lots: int, dt: Timestamp):
        """记录新开仓状态"""
        near, far = pair
        direction = self._direction.get(product, 1)
        self._current_pair[product] = pair
        self._current_lots[product] = lots
        self._current_direction[product] = direction
        self._current_position[product] = {
            near: direction * lots,
            far: -direction * lots,
        }

        entry_prices = {}
        nominal = 0.0
        for vt_symbol in pair:
            try:
                price = self.history_df.loc[dt, (vt_symbol, "close_price")]
            except Exception:
                price = float('nan')
            entry_prices[vt_symbol] = price
            multiplier = self._get_multiplier(product, vt_symbol)
            nominal += lots * price * multiplier

        self._entry_prices[product] = entry_prices
        self._entry_nominal[product] = nominal

    def _clear_position(self, product: str) -> Dict[str, int]:
        """清空单个品种持仓"""
        self._current_position[product] = {}
        self._current_pair[product] = None
        self._current_lots[product] = 0
        self._current_direction[product] = self._direction.get(product, 1)
        self._entry_prices[product] = {}
        self._entry_nominal[product] = 0.0
        return {}

    def _get_signal_direction(self, product: str, dt: Timestamp) -> int:
        """
        获取当日信号方向。
        未开启动态切换时返回注册表/参数指定的方向；
        开启动态切换时，基于近远月价差 spread_pct 的滚动 z-score 判断：
            z > threshold  -> 远月过度贴水，切换为 -1（空近多远）
            z <= threshold -> 默认 +1（多近空远）
        """
        if not self.dynamic_direction:
            return self._direction.get(product, 1)

        z_series = self._spread_z.get(product)
        if z_series is None:
            return self._direction.get(product, 1)

        z_val = z_series.get(dt, float('nan'))
        if pd.isna(z_val):
            return self._direction.get(product, 1)

        threshold = self._spread_z_threshold.get(product, 1.0)
        return -1 if z_val > threshold else 1

    def _get_spread_z_value(self, product: str, dt: Timestamp) -> float:
        """获取当日价差 z-score；无数据时返回 NaN。"""
        z_series = self._spread_z.get(product)
        if z_series is None:
            return float("nan")
        return z_series.get(dt, float("nan"))

    def _is_positive_regime(self, product: str, dt: Timestamp) -> bool:
        """慢速正 carry 过滤，避免在结构性重定价阶段抄底。"""
        if not self.require_positive_regime:
            return True

        regime_series = self._spread_regime_mean.get(product)
        if regime_series is None:
            return False

        regime_val = regime_series.get(dt, float("nan"))
        if pd.isna(regime_val):
            return False
        return regime_val > 0

    def _should_enter_buy_dip(self, product: str, dt: Timestamp) -> bool:
        """抄底开仓条件：正 carry regime 下，价差 z-score 足够低。"""
        if not self._is_positive_regime(product, dt):
            return False

        z_val = self._get_spread_z_value(product, dt)
        if pd.isna(z_val):
            return False

        entry_z = self._dip_entry_z.get(product, -1.5)
        return z_val <= entry_z

    def _should_exit_buy_dip(self, product: str, dt: Timestamp) -> bool:
        """抄底平仓条件：regime 失效或价差已从低位均值回归。"""
        if not self._is_positive_regime(product, dt):
            return True

        z_val = self._get_spread_z_value(product, dt)
        if pd.isna(z_val):
            return False

        exit_z = self._dip_exit_z.get(product, -0.25)
        return z_val >= exit_z

    def _to_target_dict(self, product: str) -> Dict[str, int]:
        """把单个品种持仓转换为目标仓位字典"""
        return dict(self._current_position[product])

    def _build_pair_df(self, product: str) -> DataFrame:
        """
        构建每个交易日的近月/远月合约对 DataFrame。
        - use_quarterly_pair=False：使用 dominant @1/@2 映射
        - use_quarterly_pair=True：自动选择“最近季月 vs 下一季月”
        """
        if not self.use_quarterly_pair:
            mapping = self._mapping.get(product)
            if mapping is None:
                return pd.DataFrame(columns=["near", "far"])
            # 兼容任意交易所的主力映射键：按 @1/@2 后缀匹配（如 IM88.CFFEX@1、I88.DCE@1）
            cols = [str(c) for c in mapping.columns]
            k1 = next((c for c in cols if c.endswith("@1")), None)
            k2 = next((c for c in cols if c.endswith("@2")), None)
            if k1 is None or k2 is None:
                return pd.DataFrame(columns=["near", "far"])
            return mapping[[k1, k2]].rename(columns={k1: "near", k2: "far"})

        # 季度合约对模式：从 history_df 中挑选最近季月和下一季月
        if self.history_df is None:
            return pd.DataFrame(columns=["near", "far"])

        close_df = self.history_df.xs("close_price", axis=1, level=1)
        symbols = [s for s in close_df.columns
                   if s.startswith(product) and "." in s and len(s.split(".")[0]) == 6]

        records = []
        for dt in close_df.index:
            candidates = []
            for s in symbols:
                price = close_df.loc[dt, s]
                if pd.isna(price):
                    continue
                expiry = self._get_expiry_date(s)
                listed = self._get_listed_date(s)
                if expiry is None or expiry <= dt or listed is None or dt < listed:
                    continue
                month = self._get_expiry_month(s)[1]
                if month not in (3, 6, 9, 12):
                    continue
                candidates.append((expiry, s))

            if len(candidates) < 2:
                records.append((dt, None, None))
                continue

            candidates.sort()
            near = candidates[0][1]
            far = candidates[1][1]

            near_month = self._get_expiry_month(near)
            far_month = self._get_expiry_month(far)
            span = self._month_span(near_month, far_month)
            if span <= 0 or span > self.max_month_span:
                records.append((dt, None, None))
            else:
                records.append((dt, near, far))

        return pd.DataFrame(records, columns=["datetime", "near", "far"]).set_index("datetime")

    # 常见商品合约乘数兜底（contract_df 缺失时使用，避免静默套用股指默认值）
    COMMODITY_MULTIPLIERS = {
        "I": 100, "JM": 60, "UR": 20, "P": 10, "RB": 10,
    }

    def _has_valid_price(self, vt_symbol: str, dt: Timestamp) -> bool:
        """检查合约当日是否有有效收盘价"""
        try:
            price = self.history_df.loc[dt, (vt_symbol, "close_price")]
        except Exception:
            return False
        return pd.notna(price) and price > 0

    def _get_multiplier(self, product: str, vt_symbol: str) -> int:
        """获取合约乘数"""
        symbol = vt_symbol.split(".")[0]
        try:
            multiplier = self.contract_df.loc[symbol, "size"]
            if pd.notna(multiplier) and multiplier > 0:
                return int(multiplier)
        except Exception:
            pass
        if product in self.COMMODITY_MULTIPLIERS:
            return self.COMMODITY_MULTIPLIERS[product]
        return 200 if product in ["IC", "IM"] else 300

    def _get_expiry_date(self, vt_symbol: str) -> Optional[Timestamp]:
        """从 contract_df 获取合约到期日"""
        symbol = vt_symbol.split(".")[0]
        try:
            expiry = self.contract_df.loc[symbol, "expiry"]
            if pd.notna(expiry):
                dt = pd.to_datetime(expiry, errors='coerce')
                if pd.notna(dt):
                    return dt
        except Exception:
            pass
        return None

    def _get_listed_date(self, vt_symbol: str) -> Optional[Timestamp]:
        """从 contract_df 获取合约上市日"""
        symbol = vt_symbol.split(".")[0]
        try:
            listed = self.contract_df.loc[symbol, "listed"]
            if pd.notna(listed):
                dt = pd.to_datetime(listed, errors='coerce')
                if pd.notna(dt):
                    return dt
        except Exception:
            pass
        return None

    @staticmethod
    def _get_expiry_month(vt_symbol: str) -> Tuple[int, int]:
        """解析合约到期年月，如 IM2609 -> (2026, 9)、I2609 -> (2026, 9)。

        取合约代码末尾 4 位数字（YYMM），兼容单字母品种（I/P 等代码总长 5 位）。
        """
        code = vt_symbol.split(".")[0]
        digits = code[-4:]
        if len(digits) < 4 or not digits.isdigit():
            return (0, 0)
        year_short = int(digits[:2])
        month = int(digits[2:])
        return (2000 + year_short, month)

    @staticmethod
    def _month_span(m1: Tuple[int, int], m2: Tuple[int, int]) -> int:
        """计算两个年月之间的月份跨度"""
        return (m2[0] - m1[0]) * 12 + (m2[1] - m1[1])


import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
