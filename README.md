# 期货期限结构策略回测 — 标准化流程

> 本目录严格按照 `/root/long-short-term-strategy-revise/` 的 AlphaResearch 标准化模板执行。
> 除策略核心逻辑（双合约交易、期限结构 Z-Score 因子）外，所有流程均遵循模板规范。

## 项目结构

```
.
├── README.md                       # 本文件：执行指南
├── spread_zscore_factor.py         # Step 1：因子定义（继承 FactorTemplate）
├── term_structure_strategy.py      # Step 2：策略定义（继承 StrategyTemplate）
├── run_backtest.py                 # Step 1/2/3 统一执行脚本
├── notebook/                       # Jupyter Notebook（可选，用于交互式分析）
└── archive/                        # 历史版本与日志备份（已归档）
```

## 依赖前提

1. **数据已入库**（由数据准备脚本完成，不在本项目中重复）：
   - 合约信息表 → `DataCenter.save_contract_df()`
   - 主力映射表 → `vnpy_dominant_contract`（@1 主力、@2 次主力）
   - 日线数据 → `get_database().save_bar_data()`（含具体合约、88、88A2）

2. **Python 环境**：已安装 `vnpy_alpharesearch` 及其依赖。

## 标准三步执行流程

### 方式一：命令行一键执行（推荐）

```bash
cd /root/cs_developer
python3 run_backtest.py
```

脚本内部自动完成：
- **Step 1**：因子计算 → `FactorGenerator.generate_factor()` → 保存到 DataCenter
- **Step 2**：策略回测 → `StrategyBacktester.run_backtesting()` → 生成 target_df
- **Step 3**：绩效统计 → `calculate_portfolio_performance()` → 输出绩效指标与图表

### 方式二：Jupyter Notebook 交互式执行

参考 `/root/long-short-term-strategy-revise/script/` 下的 notebook 模式：
1. `5 - CarryFactor计算生成.ipynb` 模式 → 对应本项目的 `spread_zscore_factor.py`
2. `6 - CarryStrategy策略回测.ipynb` 模式 → 对应本项目的 `term_structure_strategy.py`

## 核心组件说明

### 1. 因子：`SpreadZScoreFactor`

- **继承**：`FactorTemplate`
- **核心逻辑**：计算 (F1_close - F2_close) 的 Z-Score
- **输入**：88 + 88A2 的行情数据（由 `FactorGenerator.load_data()` 提供）
- **输出**：每个 dominant_symbol（88）上的因子值
- **保存**：`DataCenter.save_factor_df(name="spread_zscore", parameter="lookback20", author="futures_term_structure")`

### 2. 策略：`TermStructureStrategy`

- **继承**：`StrategyTemplate`
- **核心逻辑**：
  - 根据 spread_zscore 因子排序，做多尾部、做空头部
  - 每个品种同时交易 F1（主力）和 F2（次主力），方向相反
  - 持仓周期 `holding_period` 天，每日分仓新开（滚动持仓）
- **标准化组件**：
  - `DominantManager`：映射 88 → 具体主力合约，计算交易手数
  - `FactorManager`：加载因子数据（含未来函数检查）
  - `DataCenter`：额外加载 @2 次主力映射
- **输出**：`target_df`，列名为**具体合约代码**（如 `RB2610.SHFE`），非 88

### 3. 回测执行脚本：`run_backtest.py`

严格遵循模板的三段式：

```python
# Step 1: 因子计算
fg = FactorGenerator(vt_symbols, Interval.DAILY, start, end)
fg.load_data()
factor_df = fg.generate_factor(SpreadZScoreFactor, factor_setting)
dc.save_factor_df(factor_df, ...)

# Step 2: 策略回测
backtester = StrategyBacktester(dominant_symbols, Interval.DAILY, start, end, capital)
backtester.load_data()
target_df = backtester.run_backtesting(TermStructureStrategy, strategy_setting)

# Step 3: 绩效分析
result = calculate_portfolio_performance(target_df, Interval.DAILY, commission, capital)
```

## 参数配置

所有可调参数集中在 `run_backtest.py` 顶部的 `CONFIG` 字典中：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `dominant_symbols` | 品种池（88 合约） | 51 个活跃商品期货 |
| `start` / `end` | 回测区间 | 2011-01-01 ~ 2026-04-30 |
| `capital` | 初始资金 | 10,000,000 |
| `lookback` | 因子回看周期 | 20 交易日 |
| `holding_period` | 持仓天数 | 5 交易日 |
| `trading_signal` | 多空两端各选比例 | 0.1（每端约 5 个品种） |
| `commission` | 手续费率（双边） | 0.0001（万 1） |

## 注意事项

1. **不要修改 `vnpy_alpharesearch` 框架源码**。所有自定义逻辑应在因子/策略类中实现。
2. **不要使用自定义的绩效计算函数**。统一使用框架内置的 `calculate_portfolio_performance`。
3. **不要在策略类中使用 `empty_output` 等 hack**。框架的 `output` 函数已正确处理 `newline` 参数。
4. **target_df 的列必须是具体合约代码**，不能是 88 连续合约。绩效计算基于具体合约的 K 线数据。
5. **如需调整策略逻辑**，仅修改 `term_structure_strategy.py` 中的 `calculate_target()` 和 `calculate_daily_target_series()`，其余保持不动。
