# 商品期货日历价差扫描报告（多近空远）

- 回测区间：2020-01-02 ~ 2026-07-22
- 策略：EnhancedCalendarSpreadStrategy，direction=+1 多近空远，vol_threshold=0.15（20 日年化波动，88 主连收盘价），roll_before_days=5，only_quarterly=False，use_quarterly_pair=False，dynamic_direction=False，无止盈止损，max_month_span=12
- 资金：10,000,000，leverage=2.0，capital_ratio=1.0，commission=万1（单边按成交额）
- 口径：信号 T 日收盘计算，T+1 日开盘价成交；日盈亏 = start_pos×(close−pre_close) + pos_change×(close−open) − 佣金；夏普 = 日频收益均值/标准差 × √240
- 胜率/持有天数基于价差组合层面往返交易（pnl 为收盘价近似，不含日内成交价与手续费，仅用于统计）

## 汇总指标

| 品种 | 交易所 | 总盈亏 | 总收益率 | 年化 | 夏普 | 最大回撤 | 交易对数 | 胜率 | 平均持有天数 | 持仓天数占比 | 当前持仓 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I | DCE | 2,635,190 | 26.35% | 4.05% | 0.54 | -11.54% | 23 | 52.2% | 60.3 | 88.7% | 否 |
| UR | CZCE | -1,564,809 | -15.65% | -2.45% | -0.23 | -25.14% | 32 | 37.5% | 33.9 | 70.9% | 否 |
| P | DCE | 2,697,154 | 26.97% | 4.14% | 0.68 | -8.36% | 29 | 51.7% | 43.8 | 81.2% | 否 |
| RB | SHFE | -650,238 | -6.50% | -1.07% | -0.22 | -11.96% | 31 | 45.2% | 23.9 | 47.4% | 否 |
| JM | DCE | -2,068,475 | -20.68% | -3.18% | -0.20 | -32.38% | 21 | 33.3% | 61.8 | 83.0% | 否 |

## 数据质量备注

以下合约出现在主力映射 @1/@2 中，但 vnpy.bar_data 无日线数据，已从回测合约池剔除；映射引用这些合约的交易日策略自动空仓（近/远腿无有效收盘价即不开仓）：

- I：I2202.DCE,I2203.DCE,I2204.DCE,I2701.DCE
- UR：UR2007.CZCE,UR2112.CZCE,UR2203.CZCE,UR2204.CZCE,UR2208.CZCE,UR2212.CZCE,UR2302.CZCE,UR2303.CZCE,UR2306.CZCE,UR2310.CZCE,UR2311.CZCE,UR2402.CZCE,UR2404.CZCE,UR2606.CZCE,UR2607.CZCE,UR2610.CZCE,UR2611.CZCE,UR2701.CZCE
- P：P2110.DCE,P2111.DCE,P2202.DCE,P2206.DCE,P2207.DCE,P2210.DCE,P2302.DCE,P2306.DCE,P2310.DCE,P2701.DCE
- RB：RB2311.SHFE,RB2403.SHFE,RB2407.SHFE,RB2411.SHFE,RB2503.SHFE,RB2607.SHFE,RB2611.SHFE,RB2701.SHFE
- JM：JM2102.DCE,JM2106.DCE,JM2110.DCE,JM2111.DCE,JM2202.DCE,JM2701.DCE

## 文件清单

每品种输出（`commodity_<product>_*.csv`）：
- `_pnl.csv`：每日 balance/drawdown/net_pnl/return
- `_target.csv`：每日目标仓位
- `_trades.csv`：逐合约仓位变动明细
- `_trade_pairs.csv`：价差组合层面往返交易（含移仓标记）

注：`_trade_pairs.csv` 中 `rolled=True` 表示该笔以移仓（平旧对新开新对）结束，策略在主力映射 @1/@2 切换或近月到期前 5 天时自动移仓。
