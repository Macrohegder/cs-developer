#!/usr/bin/env python3
"""
Batch Factors — 批量因子向量化计算实现

所有函数签名统一为：
    def calc_XXX(engine: FactorEngine, **params) -> pd.DataFrame:
        # 返回 wide-format DataFrame: index=datetime, columns=dominant_symbols

核心原则：
1. 纯向量化计算，不使用循环
2. 统一处理 NaN，避免除零
3. 每个函数自包含，不依赖外部状态
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factor_engine import FactorEngine


# =============================================================================
# 工具函数
# =============================================================================

def safe_div(a, b, fill=0.0):
    """安全除法，避免除零，保持 pandas 结构"""
    result = np.where(np.abs(b) > 1e-12, a / b, fill)
    if hasattr(b, 'index') and hasattr(b, 'columns'):
        return pd.DataFrame(result, index=b.index, columns=b.columns)
    if hasattr(a, 'index') and hasattr(a, 'columns'):
        return pd.DataFrame(result, index=a.index, columns=a.columns)
    return result


def safe_log_ratio(a, b, fill=0.0):
    """安全对数比率 ln(a/b)，保持 pandas 结构"""
    result = np.where((a > 0) & (b > 0), np.log(a / b), fill)
    if hasattr(b, 'index') and hasattr(b, 'columns'):
        return pd.DataFrame(result, index=b.index, columns=b.columns)
    if hasattr(a, 'index') and hasattr(a, 'columns'):
        return pd.DataFrame(result, index=a.index, columns=a.columns)
    return result


def roll_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动Z-Score"""
    mean = df.rolling(window=window, min_periods=window).mean()
    std = df.rolling(window=window, min_periods=window).std()
    return safe_div(df - mean, std, fill=0.0)


# =============================================================================
# 动量类 (Momentum)
# =============================================================================

def calc_momentum(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """N日收益率动量 = close_t / close_{t-N} - 1"""
    return safe_div(engine.close, engine.close.shift(cycle), fill=0.0) - 1.0


def calc_stable_momentum(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """稳健动量：每日收益率截面排名均值的时序平均"""
    ret = engine.returns
    # 每日截面排名 (0~1)
    rank_ret = ret.rank(axis=1, pct=True)
    # 滚动平均
    return rank_ret.rolling(window=cycle, min_periods=cycle).mean()


def calc_trend_coeff(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """趋势效率系数 = 总收益 / 收益绝对值之和"""
    ret = engine.returns
    total_ret = ret.rolling(window=cycle, min_periods=cycle).sum()
    abs_sum = ret.abs().rolling(window=cycle, min_periods=cycle).sum()
    return safe_div(total_ret, abs_sum, fill=0.0)


def calc_intraday_momentum(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """日内动量 = mean(close/open - 1)"""
    id_ret = safe_div(engine.close, engine.open_price, fill=1.0) - 1.0
    return id_ret.rolling(window=cycle, min_periods=cycle).mean()


def calc_overnight_momentum(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """隔夜动量 = mean(open/close.shift(1) - 1)"""
    on_ret = safe_div(engine.open_price, engine.close.shift(1), fill=1.0) - 1.0
    return on_ret.rolling(window=cycle, min_periods=cycle).mean()


def calc_rsi_momentum(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """RSI动量 = 上涨日收益和 / 总收益绝对值和"""
    ret = engine.returns
    up_sum = ret.clip(lower=0).rolling(window=cycle, min_periods=cycle).sum()
    abs_sum = ret.abs().rolling(window=cycle, min_periods=cycle).sum()
    return safe_div(up_sum, abs_sum, fill=0.5)


def calc_bias_indicator(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """乖离率 = close / SMA(close) - 1"""
    sma = engine.close.rolling(window=cycle, min_periods=cycle).mean()
    return safe_div(engine.close, sma, fill=1.0) - 1.0


# =============================================================================
# 反转类 (Reversal)
# =============================================================================

def calc_reversal(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """反转因子 = -N日收益率"""
    return -(safe_div(engine.close, engine.close.shift(cycle), fill=1.0) - 1.0)


# =============================================================================
# 期限结构类 (Carry)
# =============================================================================

def calc_carry_ret(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """精确年化展期收益 = (F2-F1)/F2 / 真实到期日差 × 365
    
    与 /root/long-short-term-strategy-revise/carry_factor.py 原始定义一致：
    - 使用真实主力/次主力合约的到期日差（非固定60天）
    - carry > 0 表示 Contango（次主力>主力，远期升水）
    - carry < 0 表示 Backwardation（次主力<主力，远期贴水）
    标准策略：做多低 carry（Backwardation），做空高 carry（Contango）
    """
    f1 = engine.close
    f2 = engine.f2_close
    if f2 is None or f2.empty:
        return pd.DataFrame(index=f1.index, columns=f1.columns)
    
    common_cols = f1.columns.intersection(f2.columns)
    f1 = f1[common_cols]
    f2 = f2[common_cols]
    
    # 获取真实到期日差
    dominant_df = getattr(engine, '_dominant_df', None)
    expiry_map = getattr(engine, '_expiry_map', {})
    
    if dominant_df is not None and expiry_map:
        delta_df = pd.DataFrame(index=f1.index, columns=common_cols, dtype=float)
        for col in common_cols:
            key1 = f"{col}@1"
            key2 = f"{col}@2"
            if key1 not in dominant_df.columns or key2 not in dominant_df.columns:
                continue
            # 提取合约代码（去掉交易所后缀）
            s1 = dominant_df[key1].astype(str).str.split('.').str[0]
            s2 = dominant_df[key2].astype(str).str.split('.').str[0]
            # 映射到期日
            exp1 = pd.to_datetime(s1.map(expiry_map))
            exp2 = pd.to_datetime(s2.map(expiry_map))
            delta = (exp2 - exp1).dt.days
            delta_df[col] = delta.where(delta > 0, 60.0)
        delta_df = delta_df.fillna(60.0)
    else:
        delta_df = pd.DataFrame(60.0, index=f1.index, columns=common_cols)
    
    carry = safe_div(f2 - f1, f2, fill=0.0) / delta_df * 365.0
    
    result = pd.DataFrame(index=engine.close.index, columns=engine.close.columns)
    result[common_cols] = carry
    return result


def calc_carry_momentum(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """展期动量 = F1区间收益 - F2区间收益"""
    f1 = engine.close
    f2 = engine.f2_close
    f1_ret = safe_div(f1, f1.shift(cycle), fill=1.0) - 1.0
    if f2 is None or f2.empty:
        return f1_ret
    
    common_cols = f1.columns.intersection(f2.columns)
    f2_common = f2[common_cols]
    f2_ret = safe_div(f2_common, f2_common.shift(cycle), fill=1.0) - 1.0
    
    result = pd.DataFrame(index=f1.index, columns=f1.columns)
    result[common_cols] = f1_ret[common_cols] - f2_ret
    return result


# =============================================================================
# 波动率类 (Volatility)
# =============================================================================

def calc_basic_volatility(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """实现波动率 = std(日收益率) × √252"""
    return engine.returns.rolling(window=cycle, min_periods=cycle).std() * np.sqrt(252)


def calc_parkinson_volatility(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """Parkinson波动率 = √(Σln(high/low)² / 4Nln2) × √252"""
    hl_ratio = safe_log_ratio(engine.high, engine.low, fill=0.0)
    var = (hl_ratio ** 2).rolling(window=cycle, min_periods=cycle).mean()
    return np.sqrt(var / (4.0 * np.log(2.0))) * np.sqrt(252)


def calc_gk_volatility(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """Garman-Klass波动率（简化版）"""
    log_hl = safe_log_ratio(engine.high, engine.low, fill=0.0) ** 2
    log_co = safe_log_ratio(engine.close, engine.open_price, fill=0.0) ** 2
    
    var = (0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co)
    var = var.rolling(window=cycle, min_periods=cycle).mean()
    return np.sqrt(var.clip(lower=0)) * np.sqrt(252)


def calc_duvol(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """DUVOL = 上行标准差 / 下行标准差"""
    ret = engine.returns
    mean_ret = ret.rolling(window=cycle, min_periods=cycle).mean()
    
    up_ret = ret.where(ret >= mean_ret, 0)
    down_ret = ret.where(ret < mean_ret, 0)
    
    up_std = up_ret.rolling(window=cycle, min_periods=cycle).std()
    down_std = down_ret.rolling(window=cycle, min_periods=cycle).std()
    
    return safe_div(up_std, down_std, fill=1.0)


def calc_timevol(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """时序波动率 = 近期std / 远期std"""
    ret = engine.returns
    short_std = ret.rolling(window=cycle, min_periods=cycle).std()
    long_std = ret.rolling(window=cycle * 2, min_periods=cycle * 2).std()
    return safe_div(short_std, long_std, fill=1.0)


def calc_coef_of_variation(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """变异系数 = std(return) / |mean(return)|"""
    ret = engine.returns
    std = ret.rolling(window=cycle, min_periods=cycle).std()
    mean = ret.rolling(window=cycle, min_periods=cycle).mean()
    return safe_div(std, mean.abs(), fill=0.0)


# =============================================================================
# 偏度类 (Skewness)
# =============================================================================

def calc_skew(engine: "FactorEngine", cycle: int = 60) -> pd.DataFrame:
    """收益率偏度"""
    return engine.returns.rolling(window=cycle, min_periods=cycle).skew()


# =============================================================================
# 流动性类 (Liquidity)
# =============================================================================

def calc_amivest(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """Amivest流动性 = Σ(return / volume)"""
    ret = engine.returns
    vol = engine.volume.replace(0, np.nan)
    liq = safe_div(ret, vol, fill=0.0)
    return liq.rolling(window=cycle, min_periods=cycle).sum()


def calc_abs_amivest(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """绝对Amivest = Σ(|return| / volume)"""
    ret_abs = engine.returns.abs()
    vol = engine.volume.replace(0, np.nan)
    liq = safe_div(ret_abs, vol, fill=0.0)
    return liq.rolling(window=cycle, min_periods=cycle).sum()


# =============================================================================
# 资金流向类 (Cash Flow)
# =============================================================================

def calc_cashflow(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """资金流向 = -return + volume_ret - turnover_ret"""
    ret = engine.returns
    vol_ret = safe_div(engine.volume, engine.volume.shift(cycle), fill=1.0) - 1.0
    turn_ret = safe_div(engine.turnover, engine.turnover.shift(cycle), fill=1.0) - 1.0
    return -ret + vol_ret - turn_ret


def calc_cf_rsi(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """资金流RSI = 基于日度cashflow的RSI"""
    ret = engine.returns
    vol_ret = safe_div(engine.volume, engine.volume.shift(1), fill=1.0) - 1.0
    turn_ret = safe_div(engine.turnover, engine.turnover.shift(1), fill=1.0) - 1.0
    cf = (-ret + vol_ret - turn_ret).fillna(0)
    
    up_sum = cf.clip(lower=0).rolling(window=cycle, min_periods=cycle).sum()
    abs_sum = cf.abs().rolling(window=cycle, min_periods=cycle).sum()
    return safe_div(up_sum, abs_sum, fill=0.5)


# =============================================================================
# 持仓/套保压力类 (Open Interest)
# =============================================================================

def calc_oi_change(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """持仓变化 = ln(OI_t / OI_{t-N})"""
    oi = engine.open_interest.replace(0, np.nan)
    return safe_log_ratio(oi, oi.shift(cycle), fill=0.0)


def calc_hedging_pressure(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """套保压力 = (OI_t / OI_{t-N}) / volume_{t-N}"""
    oi = engine.open_interest.replace(0, np.nan)
    oi_ratio = safe_div(oi, oi.shift(cycle), fill=1.0)
    vol = engine.volume.replace(0, np.nan)
    return safe_div(oi_ratio, vol.shift(cycle), fill=0.0)


# =============================================================================
# 技术指标类 (Technical)
# =============================================================================

def calc_aroon_osc(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """Aroon Oscillator = AroonUp - AroonDown"""
    high = engine.high
    low = engine.low
    
    aroon_up = high.rolling(window=cycle, min_periods=cycle).apply(
        lambda x: ((np.argmax(x) + 1) / cycle) * 100, raw=True
    )
    aroon_down = low.rolling(window=cycle, min_periods=cycle).apply(
        lambda x: ((np.argmin(x) + 1) / cycle) * 100, raw=True
    )
    return aroon_up - aroon_down


def calc_cci(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """CCI = (TP - MA) / (0.015 × MD)"""
    tp = (engine.high + engine.low + engine.close) / 3.0
    ma = tp.rolling(window=cycle, min_periods=cycle).mean()
    md = (tp - ma).abs().rolling(window=cycle, min_periods=cycle).mean()
    return safe_div(tp - ma, md * 0.015, fill=0.0)


def calc_williams_r(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """Williams %R = (HH - Close) / (HH - LL) × -100"""
    hh = engine.high.rolling(window=cycle, min_periods=cycle).max()
    ll = engine.low.rolling(window=cycle, min_periods=cycle).min()
    return safe_div(hh - engine.close, hh - ll, fill=-50.0) * -100.0


def calc_force_index(engine: "FactorEngine", cycle: int = 13) -> pd.DataFrame:
    """Force Index = EMA(close_diff × volume, cycle)"""
    close_diff = engine.close.diff()
    fi = close_diff * engine.volume
    return fi.ewm(span=cycle, adjust=False).mean()


def calc_adx(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """ADX 简化版"""
    high = engine.high
    low = engine.low
    close = engine.close
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).groupby(level=1, axis=1).max()
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = (-down_move).where((down_move > up_move) & (down_move > 0), 0.0)
    
    tr_smooth = tr.ewm(alpha=1/cycle, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/cycle, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/cycle, adjust=False).mean()
    
    plus_di = 100.0 * safe_div(plus_dm_smooth, tr_smooth, fill=0.0)
    minus_di = 100.0 * safe_div(minus_dm_smooth, tr_smooth, fill=0.0)
    
    dx = 100.0 * safe_div((plus_di - minus_di).abs(), plus_di + minus_di, fill=0.0)
    return dx.ewm(alpha=1/cycle, adjust=False).mean()


def calc_mfi(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """MFI = Money Flow Index"""
    tp = (engine.high + engine.low + engine.close) / 3.0
    rmf = tp * engine.volume
    tp_prev = tp.shift(1)
    
    pos_flow = rmf.where(tp > tp_prev, 0).rolling(window=cycle, min_periods=cycle).sum()
    neg_flow = rmf.where(tp < tp_prev, 0).rolling(window=cycle, min_periods=cycle).sum()
    
    mfi = 100.0 - safe_div(100.0, 1.0 + safe_div(pos_flow, neg_flow, fill=np.inf), fill=100.0)
    return mfi.fillna(50)


def calc_chaikin_osc(engine: "FactorEngine", cycle: int = 10) -> pd.DataFrame:
    """Chaikin Oscillator = EMA(short, ADL) - EMA(long, ADL)"""
    high = engine.high
    low = engine.low
    close = engine.close
    vol = engine.volume
    
    mf_mult = safe_div((close - low) - (high - close), high - low, fill=0.0)
    mf_vol = mf_mult * vol
    adl = mf_vol.cumsum()
    
    short_p = max(3, int(cycle / 3))
    long_p = cycle
    
    ema_short = adl.ewm(span=short_p, adjust=False).mean()
    ema_long = adl.ewm(span=long_p, adjust=False).mean()
    return ema_short - ema_long


def calc_ulcer_index(engine: "FactorEngine", cycle: int = 14) -> pd.DataFrame:
    """Ulcer Index = √(mean(drawdown²))"""
    rolling_max = engine.close.rolling(window=cycle, min_periods=cycle).max()
    drawdown = safe_div(engine.close - rolling_max, rolling_max, fill=0.0) * 100.0
    return np.sqrt((drawdown ** 2).rolling(window=cycle, min_periods=cycle).mean())


def calc_trix(engine: "FactorEngine", cycle: int = 15) -> pd.DataFrame:
    """TRIX = EMA(EMA(EMA(close)))的1日变化率"""
    ema1 = engine.close.ewm(span=cycle, adjust=False).mean()
    ema2 = ema1.ewm(span=cycle, adjust=False).mean()
    ema3 = ema2.ewm(span=cycle, adjust=False).mean()
    return ema3.pct_change(1) * 100.0


# =============================================================================
# 统计类 (Statistical)
# =============================================================================

def calc_zscore_price(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """价格Z-Score = (close - rolling_mean) / rolling_std"""
    return roll_zscore(engine.close, cycle)


def calc_kurtosis(engine: "FactorEngine", cycle: int = 60) -> pd.DataFrame:
    """收益率峰度"""
    return engine.returns.rolling(window=cycle, min_periods=cycle).kurt()


def calc_win_rate(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """胜率 = 正收益天数 / 总天数"""
    pos = (engine.returns > 0).rolling(window=cycle, min_periods=cycle).sum()
    total = engine.returns.notna().rolling(window=cycle, min_periods=cycle).sum()
    return safe_div(pos, total, fill=0.5)


def calc_sharpe_ratio(engine: "FactorEngine", cycle: int = 60) -> pd.DataFrame:
    """滚动夏普比率 = mean(return) / std(return) × √252"""
    ret = engine.returns
    mean = ret.rolling(window=cycle, min_periods=cycle).mean()
    std = ret.rolling(window=cycle, min_periods=cycle).std()
    return safe_div(mean, std, fill=0.0) * np.sqrt(252)


def calc_max_drawdown(engine: "FactorEngine", cycle: int = 60) -> pd.DataFrame:
    """滚动最大回撤 = max((peak - current) / peak)"""
    close = engine.close
    rolling_max = close.rolling(window=cycle, min_periods=cycle).max()
    return safe_div(rolling_max - close, rolling_max, fill=0.0)


# =============================================================================
# 现有 cs_developer 因子映射
# =============================================================================

def calc_spread_zscore(engine: "FactorEngine", lookback: int = 20) -> pd.DataFrame:
    """期限结构价差Z-Score = (spread - mean) / std"""
    f1 = engine.close
    f2 = engine.f2_close
    if f2 is None or f2.empty:
        return pd.DataFrame(index=f1.index, columns=f1.columns)
    
    spread = f1 - f2
    mean = spread.rolling(window=lookback, min_periods=lookback).mean()
    std = spread.rolling(window=lookback, min_periods=lookback).std()
    return safe_div(spread - mean, std, fill=0.0)


def calc_spread_return(engine: "FactorEngine", lookback: int = 10) -> pd.DataFrame:
    """价差动量 = (spread_t - spread_{t-N}) / |spread_{t-N}|"""
    f1 = engine.close
    f2 = engine.f2_close
    if f2 is None or f2.empty:
        return pd.DataFrame(index=f1.index, columns=f1.columns)
    
    spread = f1 - f2
    spread_prev = spread.shift(lookback)
    return safe_div(spread - spread_prev, spread_prev.abs(), fill=0.0)


# =============================================================================
# 新增因子 — 快速验证
# =============================================================================

def calc_speculation_ratio(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """投机度 = 成交量 / 持仓量。值越高投机情绪越浓，预期收益越低（IC-）"""
    volume = engine.volume
    oi = engine.open_interest
    if oi is None or oi.empty:
        return pd.DataFrame(index=engine.close.index, columns=engine.close.columns)
    ratio = safe_div(volume, oi, fill=np.nan)
    return ratio.rolling(window=cycle, min_periods=cycle).mean()


def calc_term_structure_slope(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """期限结构斜率动量 = carry_ret 的时序动量。捕捉期限结构变化的持续性"""
    # 复用 calc_carry_ret 的精确 carry 计算（真实到期日差）
    carry = calc_carry_ret(engine, cycle=5)
    # carry 的时序动量
    return carry - carry.shift(cycle)


def calc_opening_gap_reversal(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """开盘跳空反转 = 隔夜跳空幅度。大幅跳空后预期反向修复（IC-）"""
    open_p = engine.open_price
    close_prev = engine.close.shift(1)
    gap = safe_div(open_p, close_prev, fill=1.0) - 1.0
    # 取绝对跳空幅度（不区分方向，因子值越大表示跳空越剧烈）
    return gap.abs().rolling(window=cycle, min_periods=cycle).mean()


# =============================================================================
# VIP33 因子集 (Idiosyncratic Asymmetry & Cross-Sectional)
# 学术来源: Han et al. (2022) "Is idiosyncratic asymmetry priced in commodity futures?"
# =============================================================================

def _calc_idio_residuals(ret_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    二次项市场模型滚动回归残差。
    模型: r_i,t = alpha + beta1 * r_m,t + beta2 * r_m,t^2 + epsilon_i,t
    市场基准 r_m = 各品种日收益率的截面等权均值（商品等权基准）
    
    返回:
        DataFrame: 每个时间点的当期残差 epsilon（index=datetime, columns=symbols）
    """
    market_ret = ret_df.mean(axis=1).fillna(0).values
    market_ret2 = market_ret ** 2
    
    residuals = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
    
    for col in ret_df.columns:
        y = ret_df[col].fillna(0).values
        n = len(y)
        if n < window:
            continue
        
        res_col = np.full(n, np.nan)
        for t in range(window - 1, n):
            y_win = y[t - window + 1:t + 1]
            x1_win = market_ret[t - window + 1:t + 1]
            x2_win = market_ret2[t - window + 1:t + 1]
            
            if np.std(y_win) < 1e-12:
                continue
            
            X = np.column_stack([np.ones(window), x1_win, x2_win])
            try:
                beta = np.linalg.lstsq(X, y_win, rcond=None)[0]
                res_col[t] = y_win[-1] - (beta[0] + beta[1] * x1_win[-1] + beta[2] * x2_win[-1])
            except Exception:
                continue
        
        residuals[col] = res_col
    
    return residuals


def calc_amplitude(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """
    日内振幅因子 = mean((high - low) / close)
    
    直觉: 振幅太大的品种噪音太多，后续收益偏低（IC-）
    """
    amp = safe_div(engine.high - engine.low, engine.close, fill=0.0)
    return amp.rolling(window=cycle, min_periods=cycle).mean()


def calc_volume_momentum(engine: "FactorEngine", cycle: int = 20) -> pd.DataFrame:
    """
    成交量变化 = ln(volume_t / volume_{t-N})
    
    直觉: 成交量放大代表资金流入，有持续性（IC+）
    """
    vol = engine.volume.replace(0, np.nan)
    return safe_log_ratio(vol, vol.shift(cycle), fill=0.0)


def calc_ivol(engine: "FactorEngine", cycle: int = 60) -> pd.DataFrame:
    """
    特质波动率 (Idiosyncratic Volatility)
    
    构建规则:
        1. 二次项市场模型: r_i = alpha + beta1*r_m + beta2*r_m^2 + epsilon
        2. 市场基准 r_m = 商品等权基准（截面均值）
        3. IVOL = std(残差) * sqrt(252)
    
    直觉: 特质风险高 = 被过度投机，后续收益偏低（IC-）
    """
    residuals = _calc_idio_residuals(engine.returns, cycle)
    ivol = residuals.rolling(window=cycle, min_periods=cycle).std() * np.sqrt(252)
    return ivol


def calc_iskew(engine: "FactorEngine", cycle: int = 60) -> pd.DataFrame:
    """
    残差偏度 (Idiosyncratic Skewness)
    
    构建规则:
        1. 二次项市场模型回归取残差（同 IVOL）
        2. ISKEW = skew(残差)
    
    直觉: "彩票性"高的品种被追捧高估，后续收益偏低（IC-）
    """
    residuals = _calc_idio_residuals(engine.returns, cycle)
    return residuals.rolling(window=cycle, min_periods=cycle).skew()


def calc_ie(engine: "FactorEngine", cycle: int = 126, threshold: float = 0.5) -> pd.DataFrame:
    """
    特异非对称性 (Idiosyncratic Entropy, IE)
    
    学术来源: Han et al. (2022) SSRN.3391784
    
    构建规则:
        1. 二次项市场模型回归取残差 epsilon（同 IVOL）
        2. 残差标准化: z = (epsilon - mu_epsilon) / sigma_epsilon
        3. IE = P(z > threshold) - P(z < -threshold)
           （滚动窗口内，标准化残差大于正阈值的比例 减去 小于负阈值的比例）
    
    参数:
        cycle: 126天（论文标准窗口）
        threshold: 0.5（半个标准差的阈值）
    
    直觉: IE 高 = 暴涨概率比暴跌大 = 投资者争抢 = 被高估 = 后续收益低（IC-）
    """
    residuals = _calc_idio_residuals(engine.returns, cycle)
    
    # 滚动标准化残差
    mu = residuals.rolling(window=cycle, min_periods=cycle).mean()
    sigma = residuals.rolling(window=cycle, min_periods=cycle).std()
    z_score = safe_div(residuals - mu, sigma, fill=0.0)
    
    # 尾部概率差异
    upper_tail = (z_score > threshold).rolling(window=cycle, min_periods=cycle).mean()
    lower_tail = (z_score < -threshold).rolling(window=cycle, min_periods=cycle).mean()
    
    return upper_tail - lower_tail


# =============================================================================
# 文章因子 (Article / 研报复现)
# =============================================================================

def calc_cycle_reversion(engine: "FactorEngine", cycle: int = 5) -> pd.DataFrame:
    """
    周而复始因子 — 期货品种交易"过热"程度的微观结构因子
    
    来源: 中泰期货《周而复始因子研究报告》（微信公众号）
    
    构建规则:
        1. N日收益率绝对值: |close_t / close_{t-N} - 1|
        2. N日成交量对数变化率: ln(volume_t / volume_{t-N})
        3. N日持仓量对数变化率: ln(oi_t / oi_{t-N})
        4. 对三个指标分别做截面rank（0~1），解决正负值无法线性组合的问题
        5. 热度得分 = rank(|ret|) + rank(vol_chg) + rank(oi_chg)
    
    经济学逻辑:
        当品种出现"收益率绝对值、成交量、持仓量齐升"时，市场呈现"过热"状态。
        期货作为套期保值工具，在过热行情中会发挥对冲功能，导致价格朝相反方向回复。
        因此该因子为负向因子：高热度品种未来收益低，低热度品种未来收益高。
    
    参数:
        cycle: 回看周期（日），默认5天（对应周度调仓）
    
    交易方向: ic_direction = -1（做多低热度，做空高热度）
    """
    close = engine.close
    volume = engine.volume.replace(0, np.nan)
    oi = engine.open_interest.replace(0, np.nan)
    
    # 1. N日收益率绝对值
    ret = safe_div(close, close.shift(cycle), fill=1.0) - 1.0
    abs_ret = ret.abs()
    
    # 2. N日成交量对数变化率
    vol_chg = safe_log_ratio(volume, volume.shift(cycle), fill=0.0)
    
    # 3. N日持仓量对数变化率
    oi_chg = safe_log_ratio(oi, oi.shift(cycle), fill=0.0)
    
    # 4. 截面rank（0~1），NaN保持NaN
    def cross_sectional_rank(df):
        return df.rank(axis=1, pct=True, na_option='keep')
    
    rank_abs_ret = cross_sectional_rank(abs_ret)
    rank_vol_chg = cross_sectional_rank(vol_chg)
    rank_oi_chg = cross_sectional_rank(oi_chg)
    
    # 5. 热度得分 = 三者之和（越大表示越"过热"）
    # 对NaN用0.5填充（中性值），确保即使某个指标缺失也能计算
    heat_score = rank_abs_ret.fillna(0.5) + rank_vol_chg.fillna(0.5) + rank_oi_chg.fillna(0.5)
    
    return heat_score
