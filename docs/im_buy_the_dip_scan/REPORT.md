# IM Buy-The-Dip 对比报告

- 回测区间：2022-07-22 ~ 2026-07-14
- 框架：StrategyBacktester + EnhancedCalendarSpreadStrategy
- 执行时序：当日收盘生成信号，下一根 bar 生效

## 最优结果

| 模式 | total_pnl | total_return | max_dd | sharpe | trades_count | 关键参数 |
|---|---:|---:|---:|---:|---:|---|
| static | 886725.11 | 8.87% | -1.47% | 1.8089 | 126 | - |
| dynamic | 800927.28 | 8.01% | -1.48% | 1.6275 | 128 | spread_lookback=60.0, spread_z_threshold=2.5 |
| buy_the_dip | 94705.48 | 0.95% | -0.49% | 0.7334 | 12 | spread_lookback=20.0, dip_entry_z=-1.5, dip_exit_z=-0.5, regime_lookback=60.0 |
