# CS Developer — Agent 开发规范

## 项目背景

基于 `vnpy_alpharesearch` 的 CTA 商品期货量化研究框架，支持多因子横截面多空策略的完整研究链路：数据加载 → 因子计算 → IC 分析 → 回测验证 → 绩效评估。

## 核心规则（强制执行）

### 规则 1：因子值必须统一用 88 指数构建

> **因子值的计算基础必须完全一致：主力连续合约使用 `88`，次主力连续合约使用 `88A2`。这是标准化模板写死的规范，不可更改。**

**禁止事项：**
- ❌ 禁止使用 `99`、`888`、`889` 或其他连续合约指数计算因子值
- ❌ 禁止因"回测效果不佳"而擅自更换因子计算的数据基础
- ❌ 禁止为不同因子类别使用不同的数据基础

**允许事项：**
- ✅ 完整版回测（StrategyBacktester）使用具体合约价格，不受此限制
- ✅ 研究性质的指数对比测试可以临时使用其他指数，但**不得将结果作为生产因子值**

**原因：**
- 标准化模板（CSstrategy_summary-master）中所有因子的定义均基于 88/88A2 数据构建
- 因子值是策略信号的"源 truth"，必须在所有回测版本中保持一致

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

### 规则 3：只使用完整回测

> **唯一可信的回测引擎是 `StrategyBacktester` + `CrossSectionalStrategy`，使用真实合约映射（DominantManager）进行逐日交易。**

**原因：**
- 连续合约指数（88/888/889/99）均存在结构性偏差
- 88：存在换月跳价，但价格水平真实
- 888/889：后复权隐藏换月成本（0.5%~3%/次），系统性高估收益约 20%
- 99：无负价格，但换月行为与真实交易不一致
- 只有真实合约映射才能准确反映交易成本、换月滑点和持仓连续性

## 代码修改约束

### FactorEngine 默认值

```python
# factor_engine.py — get_engine() 默认值
primary_suffix: str = "88"      # 主力连续
secondary_suffix: str = "88A2"  # 次主力连续
```

**禁止**将默认值改为 `99` 或 `889`。

## 数据质量处理

如果某些品种/因子的 88/88A2 数据存在质量问题（如跳价过大、缺失严重）：
1. **不得**修改因子计算基础
2. **可以**在因子计算中增加异常值过滤（如 `safe_div`、截断、winsorize）
3. **可以**向数据提供方反馈质量问题

## 文件结构规范

```
cs_developer/
├── factor_system/           # 核心框架
│   ├── factor_engine.py     # 数据加载（默认 88+88A2）
│   ├── factor_registry.py   # 因子元数据与方向定义
│   ├── factor_monitor.py    # IC 分析
│   ├── factors/
│   │   └── batch_factors.py # 向量化因子（基于 88）
│   └── run_factor_pipeline.py
├── strategies/
│   └── cross_sectional_strategy.py  # 完整版策略
├── runners/
│   ├── run_backtest_skew.py         # 完整版回测
│   └── ...
├── run_backtest_unified.py  # 统一回测入口（完整版）
├── archive/                 # 过时脚本归档（gitignore）
└── AGENTS.md                # 本文件
```

## 编码规范与 Skill 引用

- 因子编写请加载 `vnpy-coding-standard` Skill
- Git 与协作规范请加载 `quant-workflow` Skill
- 截面因子回测/优化/参数扫描详见 `.kimi/skills/cs-pipeline/SKILL.md`
- 从文章生成截面因子代码详见 `cs-strategy-factory` Skill

:root/quant/cs_developer/AGENTS.md

## 与其他 Agent 的协作

| 协作对象 | 关系 | 说明 |
|---------|------|------|
| `strategy_factory` | 上游 | 接收生成的截面因子策略代码或 YAML |
| `portfolio_optimizer` | 下游 | 输出单策略最优参数 JSON |
| `llm-wiki` | 下游 | 批量回测达标后发布报告 |
| `data_operator` | 依赖 | 数据质量问题转交 data_operator，禁止直接修改数据源 |
| `cta_live_deploy` | 下游 | 策略源码和参数的最终消费方 |

**数据问题上报**：若 88/88A2 数据缺失、延迟或异常，转交 `data_operator` 处理，禁止自行切换数据源或修改因子计算基础。

## 修订历史

- **2026-05-04**: 初版 — 确立因子值必须用 88 指数、交易方向必须与注册表一致的核心规则
- **2026-05-04**: 移除所有简化回测代码 — 确认 888/889/99 指数均存在不可弥补的结构性偏差，只保留 StrategyBacktester 完整回测
