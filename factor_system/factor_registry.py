#!/usr/bin/env python3
"""
Factor Registry — 因子注册中心

职责：
1. 统一管理所有因子的元数据（名称、类别、参数、数据依赖、作者、状态）
2. 提供因子注册、查询、分类、批量获取接口
3. 内置 CSstrategy_summary 中可复现的因子元数据
4. 支持因子版本管理与状态跟踪（active / deprecated / testing）

使用示例：
    from factor_registry import FactorRegistry
    registry = FactorRegistry()
    
    # 获取所有波动率类因子
    vol_factors = registry.get_by_category("volatility")
    
    # 获取因子元数据
    meta = registry.get("momentum_20d")
    
    # 注册新因子
    registry.register(FactorMeta(name="my_factor", category="custom", ...))
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import json
import os


class FactorStatus(Enum):
    ACTIVE = "active"           # 生产中
    TESTING = "testing"         # 测试中
    DEPRECATED = "deprecated"   # 已废弃
    BROKEN = "broken"           # 数据/逻辑异常


@dataclass
class FactorMeta:
    """因子元数据"""
    name: str                           # 因子唯一标识名
    category: str                       # 类别: momentum, volatility, liquidity, carry, etc.
    sub_category: str = ""              # 子类别
    description: str = ""               # 因子逻辑描述
    params: Dict[str, Any] = field(default_factory=dict)   # 默认参数
    data_requirements: List[str] = field(default_factory=list)  # 所需数据字段
    author: str = "system"              # 作者
    status: FactorStatus = FactorStatus.ACTIVE
    source: str = ""                    # 来源: CSstrategy_summary / custom / literature
    references: List[str] = field(default_factory=list)  # 参考文献
    lookback_days: int = 20             # 典型回看周期
    ic_direction: int = 1               # 预期IC方向: 1=正相关, -1=负相关
    created_at: str = ""                # 创建日期
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "sub_category": self.sub_category,
            "description": self.description,
            "params": self.params,
            "data_requirements": self.data_requirements,
            "author": self.author,
            "status": self.status.value,
            "source": self.source,
            "references": self.references,
            "lookback_days": self.lookback_days,
            "ic_direction": self.ic_direction,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "FactorMeta":
        d = d.copy()
        d["status"] = FactorStatus(d.get("status", "active"))
        return cls(**d)


# =============================================================================
# 内置因子库 — 从 CSstrategy_summary 筛选出的可复现因子
# =============================================================================

BUILT_IN_FACTORS = [
    # ==================== 动量类 (Momentum) ====================
    FactorMeta(
        name="momentum",
        category="momentum",
        sub_category="price_momentum",
        description="N日收盘价收益率动量。因子值 = close_t / close_{t-N} - 1",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="stable_momentum",
        category="momentum",
        sub_category="robust_momentum",
        description="稳健动量：每日收益率截面排名均值的时序平均，降低极端值影响",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="trend_coeff",
        category="momentum",
        sub_category="trend_efficiency",
        description="趋势效率系数 = 总收益 / 收益绝对值之和。越接近1趋势越强，接近0则震荡",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="intraday_momentum",
        category="momentum",
        sub_category="intraday",
        description="日内动量 = mean(close/open - 1)，衡量日内趋势强度",
        params={"cycle": 20},
        data_requirements=["open_price", "close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="overnight_momentum",
        category="momentum",
        sub_category="overnight",
        description="隔夜动量 = mean(open/close.shift(1) - 1)，衡量隔夜跳空方向",
        params={"cycle": 20},
        data_requirements=["open_price", "close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="rsi_momentum",
        category="momentum",
        sub_category="rsi",
        description="RSI动量 = 上涨日收益和 / 总收益绝对值和。0~1之间，越大越强",
        params={"cycle": 14},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=1,
    ),
    FactorMeta(
        name="bias_indicator",
        category="momentum",
        sub_category="mean_reversion",
        description="乖离率 = close / SMA(close, N) - 1。衡量价格偏离均线的程度",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,
    ),
    
    # ==================== 反转类 (Reversal) ====================
    FactorMeta(
        name="reversal",
        category="reversal",
        sub_category="price_reversal",
        description="价格反转因子：N日收益率的负值，做多 losers 做空 winners",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,
    ),
    
    # ==================== 期限结构类 (Carry/Term Structure) ====================
    FactorMeta(
        name="carry_ret",
        category="carry",
        sub_category="term_structure",
        description="年化展期收益 = (F1-F2)/F1 / 到期日差 × 365。正值=Backwardation（远期贴水），负值=Contango（远期升水）",
        params={"cycle": 5},
        data_requirements=["close_price", "contract_expiry"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=5,
        ic_direction=1,
    ),
    FactorMeta(
        name="carry_momentum",
        category="carry",
        sub_category="spread_momentum",
        description="展期动量 = F1区间收益 - F2区间收益，捕捉近远月价差动量",
        params={"cycle": 20},
        data_requirements=["close_price", "contract_expiry"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    
    # ==================== 波动率类 (Volatility) ====================
    FactorMeta(
        name="basic_volatility",
        category="volatility",
        sub_category="realized_vol",
        description="实现波动率 = std(日收益率) × √252",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,

    ),
    FactorMeta(
        name="parkinson_volatility",
        category="volatility",
        sub_category="high_low_vol",
        description="Parkinson波动率 = √(Σln(high/low)² / 4Nln2) × √252，利用高低价信息",
        params={"cycle": 20},
        data_requirements=["high_price", "low_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,

    ),
    FactorMeta(
        name="gk_volatility",
        category="volatility",
        sub_category="ohl_vol",
        description="Garman-Klass波动率，利用开高低收四价信息，效率更高",
        params={"cycle": 20},
        data_requirements=["open_price", "high_price", "low_price", "close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,

    ),
    FactorMeta(
        name="duvol",
        category="volatility",
        sub_category="down_up_asymmetry",
        description="DUVOL = 上行标准差 / 下行标准差。>1表示下行波动更大",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,

    ),
    FactorMeta(
        name="timevol",
        category="volatility",
        sub_category="term_structure_vol",
        description="时序波动率 = 近期std / 远期std，衡量波动率收敛/发散",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=40,
        ic_direction=1,

    ),
    FactorMeta(
        name="coef_of_variation",
        category="volatility",
        sub_category="relative_vol",
        description="变异系数 = std(return) / |mean(return)|，标准化波动率度量",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,

    ),
    
    # ==================== 偏度类 (Skewness) ====================
    FactorMeta(
        name="skew",
        category="skewness",
        sub_category="return_skew",
        description="收益率偏度：衡量收益分布不对称性。负偏度品种有崩盘风险溢价",
        params={"cycle": 60},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=60,
        ic_direction=-1,

    ),
    
    # ==================== 流动性类 (Liquidity) ====================
    FactorMeta(
        name="amivest",
        category="liquidity",
        sub_category="price_impact",
        description="Amivest流动性 = Σ(return / volume)。值越大流动性越差",
        params={"cycle": 20},
        data_requirements=["close_price", "volume"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,
    ),
    FactorMeta(
        name="abs_amivest",
        category="liquidity",
        sub_category="price_impact",
        description="绝对Amivest = Σ(|return| / volume)。更稳健的流动性度量",
        params={"cycle": 20},
        data_requirements=["close_price", "volume"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,
    ),
    
    # ==================== 资金流向类 (Cash Flow) ====================
    FactorMeta(
        name="cashflow",
        category="cashflow",
        sub_category="money_flow",
        description="资金流向 = -return + volume_ret - turnover_ret。综合量价变化",
        params={"cycle": 20},
        data_requirements=["close_price", "volume", "turnover"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="cf_rsi",
        category="cashflow",
        sub_category="money_flow_momentum",
        description="资金流RSI = 基于cashflow计算的RSI动量指标",
        params={"cycle": 14},
        data_requirements=["close_price", "volume", "turnover"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=1,
    ),
    
    # ==================== 持仓/套保压力类 (Open Interest) ====================
    FactorMeta(
        name="oi_change",
        category="open_interest",
        sub_category="oi_growth",
        description="持仓变化 = ln(OI_t / OI_{t-N})，衡量资金关注度变化",
        params={"cycle": 5},
        data_requirements=["open_interest"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=5,
        ic_direction=1,
    ),
    FactorMeta(
        name="hedging_pressure",
        category="open_interest",
        sub_category="hp_ratio",
        description="套保压力 = (OI_t / OI_{t-N}) / volume_{t-N}。持仓增速相对成交量的比率",
        params={"cycle": 5},
        data_requirements=["open_interest", "volume"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=5,
        ic_direction=1,
    ),
    
    # ==================== 技术指标类 (Technical) ====================
    FactorMeta(
        name="aroon_osc",
        category="technical",
        sub_category="trend",
        description="Aroon Oscillator = AroonUp - AroonDown，趋势强度与方向",
        params={"cycle": 14},
        data_requirements=["high_price", "low_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=1,
    ),
    FactorMeta(
        name="cci",
        category="technical",
        sub_category="mean_reversion",
        description="CCI = (TP - MA) / (0.015 × MD)。商品通道指数，超买超卖指标",
        params={"cycle": 20},
        data_requirements=["high_price", "low_price", "close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,
    ),
    FactorMeta(
        name="williams_r",
        category="technical",
        sub_category="momentum",
        description="Williams %R = (HH - Close) / (HH - LL) × -100。动量超买超卖",
        params={"cycle": 14},
        data_requirements=["high_price", "low_price", "close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=-1,
    ),
    FactorMeta(
        name="force_index",
        category="technical",
        sub_category="volume_price",
        description="Force Index = (close_diff × volume) 的 EMA。量价推动力",
        params={"cycle": 13},
        data_requirements=["close_price", "volume"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=13,
        ic_direction=1,
    ),
    FactorMeta(
        name="adx",
        category="technical",
        sub_category="trend_strength",
        description="ADX = 平均趋向指数。衡量趋势强度，不论方向",
        params={"cycle": 14},
        data_requirements=["high_price", "low_price", "close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=1,
    ),
    FactorMeta(
        name="mfi",
        category="technical",
        sub_category="volume_price",
        description="MFI = Money Flow Index。量价结合的超买超卖指标",
        params={"cycle": 14},
        data_requirements=["high_price", "low_price", "close_price", "volume"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=-1,
    ),
    FactorMeta(
        name="chaikin_osc",
        category="technical",
        sub_category="volume_price",
        description="Chaikin Oscillator = ADL的短期EMA - 长期EMA。资金流趋势指标",
        params={"cycle": 10},
        data_requirements=["high_price", "low_price", "close_price", "volume"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=10,
        ic_direction=1,
    ),
    FactorMeta(
        name="ulcer_index",
        category="technical",
        sub_category="risk",
        description="Ulcer Index = √(mean(drawdown²))。回撤深度指标",
        params={"cycle": 14},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=14,
        ic_direction=-1,
    ),
    FactorMeta(
        name="trix",
        category="technical",
        sub_category="trend",
        description="TRIX = EMA(EMA(EMA(close)))的1日变化率。三重平滑动量",
        params={"cycle": 15},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=45,
        ic_direction=1,
    ),
    
    # ==================== 统计类 (Statistical) ====================
    FactorMeta(
        name="zscore_price",
        category="statistical",
        sub_category="standardization",
        description="价格Z-Score = (close - rolling_mean) / rolling_std。价格偏离度",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=-1,

    ),
    FactorMeta(
        name="kurtosis",
        category="statistical",
        sub_category="tail_risk",
        description="收益率峰度 = 衡量尾部厚度。高峰度表示极端事件风险",
        params={"cycle": 60},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=60,
        ic_direction=-1,

    ),
    FactorMeta(
        name="win_rate",
        category="statistical",
        sub_category="behavioral",
        description="胜率 = N日正收益天数 / 总天数。衡量盈利一致性",
        params={"cycle": 20},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=20,
        ic_direction=1,

    ),
    FactorMeta(
        name="sharpe_ratio",
        category="statistical",
        sub_category="risk_adjusted_return",
        description="滚动夏普比率 = mean(return) / std(return) × √252",
        params={"cycle": 60},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=60,
        ic_direction=1,

    ),
    FactorMeta(
        name="max_drawdown",
        category="statistical",
        sub_category="risk",
        description="滚动最大回撤 = max((peak - current) / peak)",
        params={"cycle": 60},
        data_requirements=["close_price"],
        author="CSstrategy_summary",
        source="CSstrategy_summary",
        lookback_days=60,
        ic_direction=-1,

    ),
    
    # ==================== 现有 cs_developer 因子 ====================
    FactorMeta(
        name="spread_zscore",
        category="carry",
        sub_category="mean_reversion",
        description="期限结构价差Z-Score = (spread - mean) / std。均值回归策略",
        params={"lookback": 20},
        data_requirements=["close_price", "contract_expiry"],
        author="futures_term_structure",
        source="cs_developer",
        lookback_days=20,
        ic_direction=-1,
    ),
    FactorMeta(
        name="spread_return",
        category="carry",
        sub_category="momentum",
        description="价差动量 = (spread_t - spread_{t-N}) / |spread_{t-N}|",
        params={"lookback": 10},
        data_requirements=["close_price", "contract_expiry"],
        author="futures_term_structure",
        source="cs_developer",
        lookback_days=10,
        ic_direction=-1,
    ),
    
    # ==================== 新增快速验证因子 ====================
    FactorMeta(
        name="speculation_ratio",
        category="sentiment",
        sub_category="speculation",
        description="投机度 = 成交量 / 持仓量。高投机度品种通常后续收益更低（IC-）",
        params={"cycle": 20},
        data_requirements=["volume", "open_interest"],
        author="cs_developer",
        source="cs_developer",
        lookback_days=20,
        ic_direction=-1,
    ),
    FactorMeta(
        name="term_structure_slope",
        category="carry",
        sub_category="term_structure_momentum",
        description="期限结构斜率动量 = carry_ret 的时序动量。捕捉期限结构变化的持续性",
        params={"cycle": 20},
        data_requirements=["close_price", "contract_expiry"],
        author="cs_developer",
        source="cs_developer",
        lookback_days=20,
        ic_direction=1,
    ),
    FactorMeta(
        name="opening_gap_reversal",
        category="technical",
        sub_category="mean_reversion",
        description="开盘跳空反转 = 隔夜跳空幅度的绝对值均值。大幅跳空后预期反向修复（IC-）",
        params={"cycle": 20},
        data_requirements=["open_price", "close_price"],
        author="cs_developer",
        source="cs_developer",
        lookback_days=20,
        ic_direction=-1,
    ),
    
    # ==================== 889 版本（基于 889 数据计算因子值）====================
    # 动态生成：为每个内置因子（除已带_889后缀的）创建 889 版本
]

# 动态生成 _889 版本
_BUILT_IN_889_FACTORS = []
for _meta in BUILT_IN_FACTORS:
    # 跳过已带 _889 后缀的
    if _meta.name.endswith("_889"):
        continue
    
    _889_params = _meta.params.copy()
    # 如果是 carry 类，次主力也需要对应调整（889 的次主力用 889A2 不存在，用 88A2）
    _889_meta = FactorMeta(
        name=f"{_meta.name}_889",
        category=_meta.category,
        sub_category=_meta.sub_category,
        description=f"[{_meta.name}] 基于 889 数据计算的版本。" + _meta.description,
        params=_889_params,
        data_requirements=_meta.data_requirements,
        author=_meta.author,
        source="cs_developer_889",
        status=_meta.status,
        lookback_days=_meta.lookback_days,
        ic_direction=_meta.ic_direction,
        # backtest_price_suffix removed - only full backtest is used
        created_at=_meta.created_at,
    )
    _BUILT_IN_889_FACTORS.append(_889_meta)

# 添加 skew 180 天版本（88 和 889）
_skew_meta = next((m for m in BUILT_IN_FACTORS if m.name == "skew"), None)
if _skew_meta:
    _BUILT_IN_889_FACTORS.append(FactorMeta(
        name="skew_180",
        category="skewness",
        sub_category="return_skew_longterm",
        description="收益率偏度（180天长期版本）：衡量收益分布不对称性，捕捉更长期的崩盘风险溢价",
        params={"cycle": 180},
        data_requirements=["close_price"],
        author="cs_developer",
        source="cs_developer",
        status=FactorStatus.ACTIVE,
        lookback_days=180,
        ic_direction=-1,

    ))
    _BUILT_IN_889_FACTORS.append(FactorMeta(
        name="skew_180_889",
        category="skewness",
        sub_category="return_skew_longterm",
        description="收益率偏度（180天长期版本，基于889数据计算）：衡量收益分布不对称性",
        params={"cycle": 180},
        data_requirements=["close_price"],
        author="cs_developer",
        source="cs_developer_889",
        status=FactorStatus.ACTIVE,
        lookback_days=180,
        ic_direction=-1,

    ))

# 合并到内置因子库
BUILT_IN_FACTORS.extend(_BUILT_IN_889_FACTORS)


class FactorRegistry:
    """因子注册中心 — 单例模式"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        
        self._registry: Dict[str, FactorMeta] = {}
        self._category_index: Dict[str, List[str]] = {}
        self._status_index: Dict[str, List[str]] = {}
        self._source_index: Dict[str, List[str]] = {}
        
        # 加载内置因子
        self._load_builtin_factors()
        
        # 加载外部配置
        self.config_path = config_path or "/root/cs_developer/factor_system/data/factor_registry.json"
        if os.path.exists(self.config_path):
            self._load_from_disk()
    
    def _load_builtin_factors(self):
        """加载内置因子库"""
        for meta in BUILT_IN_FACTORS:
            self._add_to_indices(meta)
            self._registry[meta.name] = meta
    
    def _add_to_indices(self, meta: FactorMeta):
        """更新索引"""
        # 类别索引
        if meta.category not in self._category_index:
            self._category_index[meta.category] = []
        if meta.name not in self._category_index[meta.category]:
            self._category_index[meta.category].append(meta.name)
        
        # 状态索引
        status_key = meta.status.value
        if status_key not in self._status_index:
            self._status_index[status_key] = []
        if meta.name not in self._status_index[status_key]:
            self._status_index[status_key].append(meta.name)
        
        # 来源索引
        if meta.source not in self._source_index:
            self._source_index[meta.source] = []
        if meta.name not in self._source_index[meta.source]:
            self._source_index[meta.source].append(meta.name)
    
    def _remove_from_indices(self, meta: FactorMeta):
        """从索引中移除"""
        for idx_dict in [self._category_index, self._status_index, self._source_index]:
            for key, names in list(idx_dict.items()):
                if meta.name in names:
                    names.remove(meta.name)
    
    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    
    def register(self, meta: FactorMeta) -> bool:
        """注册新因子"""
        if meta.name in self._registry:
            print(f"[WARN] 因子 '{meta.name}' 已存在，将被覆盖")
            self._remove_from_indices(self._registry[meta.name])
        
        self._registry[meta.name] = meta
        self._add_to_indices(meta)
        print(f"[OK] 因子 '{meta.name}' 注册成功")
        return True
    
    def get(self, name: str) -> Optional[FactorMeta]:
        """获取因子元数据"""
        return self._registry.get(name)
    
    def exists(self, name: str) -> bool:
        """检查因子是否存在"""
        return name in self._registry
    
    def unregister(self, name: str) -> bool:
        """注销因子"""
        if name not in self._registry:
            return False
        meta = self._registry.pop(name)
        self._remove_from_indices(meta)
        return True
    
    def update_status(self, name: str, status: FactorStatus):
        """更新因子状态"""
        meta = self._registry.get(name)
        if meta is None:
            raise KeyError(f"因子 '{name}' 不存在")
        self._remove_from_indices(meta)
        meta.status = status
        self._add_to_indices(meta)
    
    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    
    def list_all(self) -> List[str]:
        """列出所有因子名"""
        return list(self._registry.keys())
    
    def get_by_category(self, category: str) -> List[FactorMeta]:
        """按类别获取因子"""
        names = self._category_index.get(category, [])
        return [self._registry[n] for n in names]
    
    def get_by_status(self, status: FactorStatus) -> List[FactorMeta]:
        """按状态获取因子"""
        names = self._status_index.get(status.value, [])
        return [self._registry[n] for n in names]
    
    def get_by_source(self, source: str) -> List[FactorMeta]:
        """按来源获取因子"""
        names = self._source_index.get(source, [])
        return [self._registry[n] for n in names]
    
    def get_active(self) -> List[FactorMeta]:
        """获取所有活跃因子"""
        return self.get_by_status(FactorStatus.ACTIVE)
    
    def filter_factors(self, 
                       category: Optional[str] = None,
                       status: Optional[FactorStatus] = None,
                       source: Optional[str] = None,
                       data_requirements: Optional[List[str]] = None) -> List[FactorMeta]:
        """多条件组合过滤"""
        result = list(self._registry.values())
        
        if category:
            result = [m for m in result if m.category == category]
        if status:
            result = [m for m in result if m.status == status]
        if source:
            result = [m for m in result if m.source == source]
        if data_requirements:
            result = [m for m in result if all(req in m.data_requirements for req in data_requirements)]
        
        return result
    
    def get_categories(self) -> List[str]:
        """获取所有类别"""
        return list(self._category_index.keys())
    
    def count_by_category(self) -> Dict[str, int]:
        """统计各类别因子数量"""
        return {k: len(v) for k, v in self._category_index.items()}
    
    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    
    def save_to_disk(self):
        """保存到磁盘"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        data = {name: meta.to_dict() for name, meta in self._registry.items()}
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 因子注册表已保存: {self.config_path}")
    
    def _load_from_disk(self):
        """从磁盘加载"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for name, d in data.items():
            if name not in self._registry:
                meta = FactorMeta.from_dict(d)
                self._add_to_indices(meta)
                self._registry[name] = meta
        print(f"[OK] 因子注册表已加载: {len(data)} 条记录")
    
    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    
    def generate_summary(self) -> str:
        """生成因子库摘要报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("Factor Registry Summary")
        lines.append("=" * 70)
        lines.append(f"Total Factors: {len(self._registry)}")
        lines.append("")
        lines.append("By Category:")
        for cat, count in sorted(self.count_by_category().items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:<20s}: {count:3d}")
        lines.append("")
        lines.append("By Status:")
        for status in FactorStatus:
            count = len(self._status_index.get(status.value, []))
            lines.append(f"  {status.value:<12s}: {count:3d}")
        lines.append("")
        lines.append("By Source:")
        for src, names in sorted(self._source_index.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {src:<25s}: {len(names):3d}")
        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# 快捷入口
# =============================================================================

def get_registry() -> FactorRegistry:
    """获取全局注册表实例"""
    return FactorRegistry()


if __name__ == "__main__":
    reg = get_registry()
    print(reg.generate_summary())
