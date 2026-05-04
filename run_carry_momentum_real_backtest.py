#!/usr/bin/env python3
"""对 carry_momentum_real 跑完整回测并与 carry_momentum_889 对比"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "factor_system"))

from run_backtest_unified import run_single_backtest, check_factor_exists
from factor_system.factor_registry import get_registry, FactorMeta, FactorStatus

registry = get_registry()
meta = FactorMeta(
    name="carry_momentum_real",
    category="carry",
    sub_category="spread_momentum_real",
    description="展期动量（真实合约版）= 真实主力合约20日收益 - 真实次主力合约20日收益",
    params={"cycle": 20},
    data_requirements=["close_price"],
    author="cs_developer",
    source="cs_developer",
    lookback_days=20,
    ic_direction=1,
)
if registry.get("carry_momentum_real") is None:
    registry.register(meta)

symbols = [
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

start = datetime(2020, 1, 1)
end = datetime(2024, 12, 31)

param = "cycle20"
if not check_factor_exists("carry_momentum_real", param, "factor_system", symbols, start, end):
    print("[ERROR] 因子数据不存在")
    sys.exit(1)

print("=" * 70)
print("跑 carry_momentum_real 完整回测")
print("=" * 70)

result = run_single_backtest(
    factor_name="carry_momentum_real",
    parameter=param,
    factor_author="factor_system",
    long_low=False,
    holding_period=5,
    trading_signal=0.2,
    aggregation="sum",
    leverage=2.0,
    start=start,
    end=end,
    capital=10_000_000,
    commission=0.0001,
    risk_free=0.0,
    plot_chart=False,
    symbols=symbols,
    verbose=True
)

if result:
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 70)
    print("对比: carry_momentum_real vs carry_momentum_889")
    print("=" * 70)
    
    orig_pnl = pd.read_csv("/root/cs_developer/result_carry_momentum_889_pnl.csv", parse_dates=["datetime"], index_col="datetime")
    new_pnl = pd.read_csv("/root/cs_developer/result_carry_momentum_real_pnl.csv", parse_dates=["datetime"], index_col="datetime")
    
    common_idx = orig_pnl.index.intersection(new_pnl.index)
    orig_nav = orig_pnl.loc[common_idx, "balance"] / 10_000_000
    new_nav = new_pnl.loc[common_idx, "balance"] / 10_000_000
    
    def calc_metrics(nav):
        ret = nav.pct_change().dropna()
        annual_ret = (nav.iloc[-1] ** (240 / len(nav))) - 1
        vol = ret.std() * np.sqrt(240)
        sharpe = annual_ret / vol if vol > 0 else 0
        running_max = nav.cummax()
        max_dd = ((nav - running_max) / running_max).min()
        return annual_ret, sharpe, max_dd
    
    orig_ar, orig_sr, orig_dd = calc_metrics(orig_nav)
    new_ar, new_sr, new_dd = calc_metrics(new_nav)
    
    print(f"\n{'指标':<20s} {'carry_momentum_889':>20s} {'carry_momentum_real':>24s} {'变化':>10s}")
    print("-" * 80)
    print(f"{'年化收益':<20s} {orig_ar:>+19.2%} {new_ar:>+23.2%} {(new_ar-orig_ar)*100:>+9.1f}pp")
    print(f"{'夏普比率':<20s} {orig_sr:>+19.3f} {new_sr:>+23.3f} {new_sr-orig_sr:>+9.3f}")
    print(f"{'最大回撤':<20s} {orig_dd:>19.2%} {new_dd:>23.2%} {(new_dd-orig_dd)*100:>+9.1f}pp")
    
    orig_ret = orig_nav.pct_change().dropna()
    new_ret = new_nav.pct_change().dropna()
    corr = np.corrcoef(orig_ret, new_ret)[0, 1]
    print(f"\n日收益相关性: {corr:.3f}")
    
    print("\n[OK] 全部完成!")
