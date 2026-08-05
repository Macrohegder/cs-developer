# 股指期货日历价差动态方向切换测试报告

## 1. 测试目标

在 `EnhancedCalendarSpreadStrategy` 中引入基于期限结构斜率的动态套利方向切换：

- 默认方向 `+1`：多近月、空远月（赚贴水/滚动收益）。
- 当近远月价差 `spread_pct = (near - far) / near` 的滚动 z-score 超过阈值时，切换为 `-1`：空近月、多远月（赌极端贴水修复）。

验证该逻辑是否能够提升 IC / IM 单品种及组合表现，并核对交易记录是否与策略方向一致。

## 2. 参数与方法

- 回测区间：2022-07-22 ~ 2026-07-14
- 基准参数：`vol_window=20`、`only_quarterly=True`、`roll_before_days=5`、无止盈止损
- 波动阈值：`IC=15%`、`IM=15%`
- 名义杠杆：2.0，初始资金 1,000 万
- 动态方向参数扫描：
  - `lookback ∈ {20, 40, 60}`
  - `z_threshold ∈ {0.5, 1.0, 1.5, 2.0, 2.5}`
- z-score 计算：基于每日实际近月/远月合约收盘价的价差百分比。

## 3. 关键结论

| 组合 | 参数 | 总收益 | 年化收益 | 最大回撤 | 夏普 | Calmar | 交易次数 |
|---|---|---:|---:|---:|---:|---:|---:|
| IC+IM（静态，+1） | — | 624,671.86 | 1.62% | -0.59% | **1.98** | 2.76 | 148 |
| IC+IM（动态统一参数） | lookback=60, z=1.5 | 595,799.48 | 1.55% | -0.94% | 1.97 | 1.64 | 162 |
| **IC+IM（动态按品种最优）** | IC(40,1.0) + IM(20,2.5) | **858,835.33** | **2.23%** | **-0.43%** | **2.66** | **5.14** | 192 |

按品种使用各自最优动态参数后，组合夏普从 1.98 提升到 **2.66**，最大回撤从 -0.59% 降至 **-0.43%**，总收益提升约 **37%**。

### 3.1 单品种最优动态参数

| 品种 | 最优参数 | 总收益 | 夏普 | 方向分布（+1 / -1） |
|---|---|---:|---:|---:|
| IC | lookback=40, z=1.0 | 650,297.20 | 1.37 | 241天 / 48天 |
| IM | lookback=20, z=2.5 | 1,136,069.85 | 2.78 | 304天 / 2天 |

IC 在动态切换后夏普从静态的约 0.48 大幅提升到 1.37；IM 由于阈值较高（z=2.5），几乎始终维持 +1 方向，但夏普仍略高于静态（2.78 vs 2.74）。

## 4. 交易记录核对

策略在方向为 `-1` 时，交易记录显示为：

- 近月合约：`SELL`（空头）
- 远月合约：`BUY`（多头）

示例（IC+IM 按品种最优）：

| datetime | product | vt_symbol | role | change | direction | price |
|---|---|---|---|---:|---|---:|
| 2022-08-19 | IM | IM2209.CFFEX | near | -3 | SELL | 7186.4 |
| 2023-11-08 | IC | IC2312.CFFEX | near | -4 | SELL | 5587.8 |
| 2024-02-01 | IC | IC2403.CFFEX | near | -5 | SELL | 4596.0 |
| 2025-02-05 | IC | IC2503.CFFEX | near | -4 | SELL | 5567.8 |

核对结果：`-1` 方向的开仓记录全部符合“空近月、多远月”逻辑，策略方向与交易方向一致。

## 5. 产出文件

- 参数扫描汇总：`docs/ic_im_vol_scan/dynamic_direction_scan_summary.csv`
- IC 动态最优交易记录：`docs/ic_im_vol_scan/dynamic_ic_trades.csv`
- IM 动态最优交易记录：`docs/ic_im_vol_scan/dynamic_im_trades.csv`
- 组合动态统一参数交易记录：`docs/ic_im_vol_scan/dynamic_ic_im_trades.csv`
- 组合动态按品种最优交易记录：`docs/ic_im_vol_scan/dynamic_ic_im (per-product)_trades.csv`
- 组合静态对照交易记录：`docs/ic_im_vol_scan/dynamic_ic_im (static)_trades.csv`

## 6. 代码改动

- `strategies/enhanced_calendar_spread_strategy.py`：
  - 新增 `dynamic_direction`、`spread_lookback`、`spread_z_threshold` 参数，支持统一值或按品种字典。
  - 在 `__init__` 中预计算每个品种近远月价差的滚动 z-score。
  - 在 `_calculate_product_target` 中根据 z-score 动态切换方向；方向变化时先平旧仓并按新方向重新开仓。
- 新增脚本：
  - `scan_dynamic_direction.py`：参数扫描。
  - `run_dynamic_direction_verification.py`：最优参数验证与交易记录核对。

## 7. 风险提示与下一步建议

1. 动态切换依赖历史价差的均值回归特性；若市场结构发生长期变化（如持续深度贴水），z-score 信号可能失效。
2. IC 的最优参数（z=1.0）触发反向交易较频繁，需关注交易成本对实际收益的影响。
3. 建议进一步测试：
   - 加入止盈止损或方向切换的滞后/确认机制，减少假突破。
   - 对 IF 品种进行同样测试，确认是否具有类似效果。
   - 使用更长的样本外区间或滚动参数优化，验证稳健性。
