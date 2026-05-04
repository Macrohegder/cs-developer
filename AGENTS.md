# CS Developer — Agent 开发规范

## 项目背景

基于 `vnpy_alpharesearch` 的 CTA 商品期货量化研究框架，支持多因子横截面多空策略的完整研究链路：数据加载 → 因子计算 → IC 分析 → 回测验证 → 绩效评估。

## 核心规则（强制执行）

### 规则 1：因子值必须统一用 88 指数构建

> **无论简化版回测还是完整版回测，因子值的计算基础必须完全一致：主力连续合约使用 `88`，次主力连续合约使用 `88A2`。这是标准化模板写死的规范，不可更改。**

**禁止事项：**
- ❌ 禁止使用 `99`、`888`、`889` 或其他连续合约指数计算因子值
- ❌ 禁止因"回测效果不佳"而擅自更换因子计算的数据基础
- ❌ 禁止为不同因子类别使用不同的数据基础

**允许事项：**
- ✅ 简化版回测的**价格数据**（returns）可以使用 `99` 指数，以避免 88 的展期跳价影响收益计算
- ✅ 完整版回测（StrategyBacktester）使用具体合约价格，不受此限制
- ✅ 研究性质的指数对比测试（如 `test_index_comparison.py`）可以临时使用其他指数，但**不得将结果作为生产因子值**

**原因：**
- 标准化模板（CSstrategy_summary-master）中所有因子的定义均基于 88/88A2 数据构建
- 因子值是策略信号的"源 truth"，必须在所有回测版本中保持一致
- 如果 88A2 数据质量导致某些因子（如 carry_ret）简化回测效果不佳，应在报告中注明，而非修改因子计算基础

### 规则 2：交易方向必须与因子注册表一致

> **每个因子的交易方向（long_low）由 `factor_registry.py` 中的 `ic_direction` 字段唯一确定，不可通过 auto_direction 机制覆盖。**

**方向映射规则：**
| `ic_direction` | 含义 | `long_low` 值 | 交易逻辑 |
|:---|:---|:---|:---|
| `+1` | 因子值越高，未来收益越高 | `False` | 做多高因子值，做空低因子值 |
| `-1` | 因子值越低，未来收益越高 | `True` | 做多低因子值，做空高因子值 |

**禁止事项：**
- ❌ 禁止在批量回测中使用 `auto_direction=True` 让系统根据数据"自动选择"方向
- ❌ 禁止因"回测夏普更高"而反转注册表规定的交易方向

**允许事项：**
- ✅ 用户通过命令行参数 `--long-low True/False` 显式覆盖（需自行承担后果）
- ✅ 研究性质的对比测试可以临时测试反向，但**不得修改注册表默认方向**

**原因：**
- 交易方向是因子的核心定义的一部分，与经济学逻辑绑定
- carry_ret: ic_direction=+1 → long_low=False（做多 backwardation）
- skew: ic_direction=-1 → long_low=True（做多负偏度）
- 方向一致性确保简化版与完整版回测的可比性

### 规则 3：回测引擎分层规范

| 回测类型 | 引擎 | 价格数据 | 因子数据来源 | 适用场景 |
|:---|:---|:---|:---|:---|
| **完整版** | `StrategyBacktester` | 具体主力合约（DominantManager 映射） | DataCenter（88 计算） | 最终验证、生产部署 |
| **简化版** | `BatchBacktestEngine` | `99` 指数（避免跳价） | parquet / DataCenter（88 计算） | 快速筛选、批量对比 |

**注意：** 简化版回测的**因子值**仍来自 88 计算的 parquet，只是**价格数据**用 99 加载。

## 代码修改约束

### FactorEngine 默认值

```python
# factor_engine.py — get_engine() 默认值
primary_suffix: str = "88"      # 主力连续
secondary_suffix: str = "88A2"  # 次主力连续
```

**禁止**将默认值改为 `99` 或 `889`。

### BatchBacktestEngine 方向选择

```python
# backtest_engine.py — run_factor() 默认行为
auto_direction: bool = False  # 禁用自动方向选择
```

`long_low` 必须从 `FactorRegistry` 的 `ic_direction` 推断。

## 数据质量处理

如果某些品种/因子的 88/88A2 数据存在质量问题（如跳价过大、缺失严重）：
1. **不得**修改因子计算基础
2. **可以**在因子计算中增加异常值过滤（如 `safe_div`、截断、winsorize）
3. **可以**在回测报告中标注"该因子简化回测与完整版存在偏差"
4. **可以**向数据提供方反馈质量问题

## 文件结构规范

```
cs_developer/
├── factor_system/           # 核心框架（新增）
│   ├── factor_engine.py     # 数据加载（默认 88+88A2）
│   ├── backtest_engine.py   # 简化回测（价格可用 99）
│   ├── factor_registry.py   # 因子元数据与方向定义
│   ├── factor_monitor.py    # IC 分析
│   ├── factors/
│   │   └── batch_factors.py # 39 个向量化因子（基于 88）
│   ├── run_factor_pipeline.py
│   └── run_batch_backtest.py
├── strategies/
│   └── cross_sectional_strategy.py  # 完整版策略
├── runners/
│   ├── run_backtest_skew.py         # 完整版回测（保留）
│   └── ...
├── run_backtest_unified.py  # 统一回测入口（完整版）
├── archive/                 # 过时脚本归档（gitignore）
└── AGENTS.md                # 本文件
```

## 修订历史

- **2026-05-04**: 初版 — 确立因子值必须用 88 指数、交易方向必须与注册表一致的核心规则
