"""
回测纯 889 动量因子 (F1 的 20 日收益，不含 F2)
"""
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, "/root/cs_developer")
sys.path.insert(0, "/tmp/trading-system--master-vnpy_alpharesearch/vnpy_alpharesearch")

from factor_system.factor_registry import FactorRegistry, FactorMeta
from factor_system.factor_engine import FactorEngine
from run_backtest_unified import run_backtest_unified

# 注册因子
registry = FactorRegistry()

# 纯 889 动量
registry.register(FactorMeta(
    name="momentum_889",
    category="momentum",
    sub_category="trend",
    description="纯889主力合约20日动量，不含次主力",
    params={"cycle": 20},
    data_requirements=["close_price"],
    author="cs_developer",
    source="cs_developer",
    lookback_days=20,
    ic_direction=1,
))

# 从 CSV 加载预计算的因子值
factor_df = pd.read_csv("factor_momentum_889.csv", parse_dates=["datetime"], index_col="datetime")

# 运行回测
run_backtest_unified(
    factor_name="momentum_889",
    factor_df=factor_df,
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    output_prefix="result_momentum_889",
    top_n=10,
    use_adaptive_weight=True,
)
