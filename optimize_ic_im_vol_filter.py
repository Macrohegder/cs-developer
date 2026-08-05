#!/usr/bin/env python3
"""
IC/IM 日历价差套利回测（单品种 / vol 过滤）

策略逻辑：
- 反向套利：做多近月（@1），做空远月（@2）
- 波动过滤：当该品种 88 指数的 20 日年化波动 >= vol_threshold 时才开仓/持仓
- 数据清洗：
    1) 过滤 @1 == @2 的异常映射
    2) 过滤近远月到期月份跨度 > 3 个月的异常映射
- 持仓：每个套利对建立后一直持有，直到 @1 或 @2 合约切换时平仓并开新对

用法：
    python optimize_ic_im_vol_filter.py --product IM --vol-threshold 0.15
    python optimize_ic_im_vol_filter.py --product IC --vol-threshold 0.15
"""

import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import numpy as np
from clickhouse_driver import Client

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "factor_system"))

from vnpy.trader.constant import Interval
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_bar_df


# 合约乘数
MULTIPLIER_MAP = {"IC": 200, "IM": 200, "IF": 300}
# 初始资金
CAPITAL = 10_000_000
# 单边手续费率
COMMISSION = 0.0001
# 名义杠杆（用于计算每日目标名义市值）
LEVERAGE = 2.0
# 每日使用资金比例（在 leverage 基础上再分配）
DAILY_CAPITAL_RATIO = 1.0


def get_multiplier(product: str) -> int:
    return MULTIPLIER_MAP.get(product, 200)


def parse_args():
    parser = argparse.ArgumentParser(description="IC/IM/IF 日历价差套利回测")
    parser.add_argument("--product", type=str, default="IM", choices=["IC", "IM", "IF"],
                        help="品种：IC、IM 或 IF")
    parser.add_argument("--vol-threshold", type=float, default=0.15,
                        help="20 日年化波动阈值（默认 0.15=15%%）")
    parser.add_argument("--start", type=str, default="2022-07-22",
                        help="回测开始日期")
    parser.add_argument("--end", type=str, default="2026-07-14",
                        help="回测结束日期")
    parser.add_argument("--capital", type=int, default=CAPITAL,
                        help="初始资金")
    parser.add_argument("--commission", type=float, default=COMMISSION,
                        help="单边手续费率")
    parser.add_argument("--output-dir", type=str,
                        default="/root/quant/cs_developer/docs/ic_im_vol_filtered",
                        help="输出目录")
    parser.add_argument("--plot", action="store_true", help="是否绘制图表")
    return parser.parse_args()


def load_dominant_mapping(product: str, start: datetime, end: datetime) -> pd.DataFrame:
    """从 ClickHouse 加载 @1/@2 主力映射"""
    client = Client(host='localhost')
    keys = [f"{product}88.CFFEX@1", f"{product}88.CFFEX@2"]
    result = client.execute(
        'SELECT datetime, key, value FROM vnpy.vnpy_dominant_contract '
        'WHERE key IN %(keys)s AND datetime >= %(start)s AND datetime <= %(end)s '
        'ORDER BY datetime',
        {'keys': keys, 'start': start, 'end': end}
    )
    df = pd.DataFrame(result, columns=['datetime', 'key', 'value'])
    df['datetime'] = pd.to_datetime(df['datetime']).dt.normalize()
    df = df.drop_duplicates(subset=['datetime', 'key'])
    pivot = df.pivot(index='datetime', columns='key', values='value')
    return pivot


def load_contract_prices(contracts: Set[str], start: datetime, end: datetime) -> Dict[str, pd.Series]:
    """加载一组具体合约的收盘价序列"""
    prices = {}
    for vt_symbol in sorted(contracts):
        try:
            symbol, exchange = vt_symbol.split('.')
            df = load_bar_df(vt_symbol, Interval.DAILY, start, end)
            if df is None or df.empty:
                continue
            if hasattr(df.index, 'tz_localize'):
                df.index = df.index.tz_localize(None)
            prices[vt_symbol] = df['close_price']
        except Exception as e:
            print(f"  [WARN] 加载 {vt_symbol} 失败: {e}")
    return prices


def get_expiry_month(vt_symbol: str) -> Tuple[int, int]:
    """从合约代码解析到期年月，如 IM2609 -> (2026, 9)"""
    code = vt_symbol.split('.')[0]
    # 品种代码后的两位年份 + 两位月份
    # 例如 IM2609: 品种 IM, 年份 26, 月份 09
    if len(code) < 6:
        return (0, 0)
    year_short = int(code[-4:-2])
    month = int(code[-2:])
    year = 2000 + year_short
    return (year, month)


def month_span(m1: Tuple[int, int], m2: Tuple[int, int]) -> int:
    """计算两个年月之间的月份跨度"""
    return (m2[0] - m1[0]) * 12 + (m2[1] - m1[1])


def calculate_volatility(price_series: pd.Series, window: int = 20) -> pd.Series:
    """计算年化波动率（基于日收益率，250个交易日）"""
    ret = price_series.pct_change()
    vol = ret.rolling(window=window, min_periods=window//2).std() * np.sqrt(250)
    return vol


def run_backtest(product: str, vol_threshold: float, start: datetime, end: datetime,
                 capital: int = CAPITAL, commission: float = COMMISSION,
                 output_dir: str = None, plot: bool = False) -> Dict:
    """运行单品种的日历价差套利回测"""

    print(f"\n{'='*70}")
    print(f"{product}-only 日历价差套利回测 | vol >= {vol_threshold*100:.0f}%")
    print(f"{'='*70}")

    # 1. 加载 @1/@2 映射
    print("\n[1/5] 加载 dominant 映射...")
    mapping = load_dominant_mapping(product, start, end)
    k1 = f"{product}88.CFFEX@1"
    k2 = f"{product}88.CFFEX@2"
    if k1 not in mapping.columns or k2 not in mapping.columns:
        raise ValueError(f"映射中缺少 {k1} 或 {k2}")
    print(f"  映射记录: {mapping.shape[0]} 天")

    # 2. 加载 88 指数价格并计算波动
    print("\n[2/5] 加载 {product}88 指数并计算波动...")
    idx_symbol = f"{product}88.CFFEX"
    idx_df = load_bar_df(idx_symbol, Interval.DAILY, start, end)
    if hasattr(idx_df.index, 'tz_localize'):
        idx_df.index = idx_df.index.tz_localize(None)
    idx_close = idx_df['close_price']
    idx_vol = calculate_volatility(idx_close, window=20)
    print(f"  指数交易日: {len(idx_close)}")
    print(f"  波动均值: {idx_vol.mean():.2%}, 波动>=阈值天数: {(idx_vol>=vol_threshold).sum()}")

    # 3. 收集所有出现过的合约并加载价格
    print("\n[3/5] 加载具体合约价格...")
    contracts = set(mapping[k1].dropna().unique()) | set(mapping[k2].dropna().unique())
    # 扩展开始/结束日期以加载完整合约历史
    load_start = start - timedelta(days=60)
    load_end = end + timedelta(days=30)
    prices = load_contract_prices(contracts, load_start, load_end)
    print(f"  需加载合约: {len(contracts)}, 成功: {len(prices)}")

    # 4. 遍历日期，生成交易记录
    print("\n[4/5] 生成交易与计算 PnL...")

    # 统一日期索引
    all_dates = mapping.index.intersection(idx_close.index)
    all_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
    all_dates = sorted(all_dates)

    # 当前持仓: {vt_symbol: 手数（正为多，负为空）}
    position: Dict[str, int] = {}
    current_pair: Optional[Tuple[str, str]] = None  # (@1, @2)
    pair_start_date: Optional[datetime] = None
    current_pair_cum_pnl: float = 0.0  # 当前套利对累计净盈亏

    # 记录每日 PnL 和持仓市值
    daily_records = []
    pair_records = []

    # 每日目标名义市值
    daily_target_nominal = capital * LEVERAGE * DAILY_CAPITAL_RATIO
    multiplier = get_multiplier(product)

    for i, dt in enumerate(all_dates):
        near = mapping.loc[dt, k1] if pd.notna(mapping.loc[dt, k1]) else None
        far = mapping.loc[dt, k2] if pd.notna(mapping.loc[dt, k2]) else None
        vol = idx_vol.get(dt, np.nan)

        # 数据清洗：@1=@2 或缺失
        if near is None or far is None or near == far:
            # 如果有持仓，平仓
            if position:
                record = close_position(dt, position, prices, commission, multiplier)
                current_pair_cum_pnl += record['net_pnl']
                if current_pair and pair_start_date:
                    pair_records.append(finish_pair(current_pair, pair_start_date, dt, current_pair_cum_pnl))
                position = {}
                current_pair = None
                pair_start_date = None
                current_pair_cum_pnl = 0.0
            daily_records.append({
                'datetime': dt, 'pnl': 0.0, 'cost': 0.0, 'net_pnl': 0.0,
                'pair': None, 'position_value': 0.0
            })
            continue

        # 数据清洗：月份跨度 > 3
        near_month = get_expiry_month(near)
        far_month = get_expiry_month(far)
        span = month_span(near_month, far_month)
        if span <= 0 or span > 3:
            if position:
                record = close_position(dt, position, prices, commission, multiplier)
                current_pair_cum_pnl += record['net_pnl']
                if current_pair and pair_start_date:
                    pair_records.append(finish_pair(current_pair, pair_start_date, dt, current_pair_cum_pnl))
                position = {}
                current_pair = None
                pair_start_date = None
                current_pair_cum_pnl = 0.0
            daily_records.append({
                'datetime': dt, 'pnl': 0.0, 'cost': 0.0, 'net_pnl': 0.0,
                'pair': None, 'position_value': 0.0
            })
            continue

        # 波动过滤
        signal_active = (vol >= vol_threshold) if pd.notna(vol) else False

        # 检查是否需要换月（当前持仓与映射不一致）
        need_rebalance = False
        if current_pair != (near, far):
            need_rebalance = True

        if need_rebalance and position:
            # 平仓旧对
            record = close_position(dt, position, prices, commission, multiplier)
            current_pair_cum_pnl += record['net_pnl']
            if current_pair and pair_start_date:
                pair_records.append(finish_pair(current_pair, pair_start_date, dt, current_pair_cum_pnl))
            position = {}
            current_pair = None
            pair_start_date = None
            current_pair_cum_pnl = 0.0

        if signal_active and not position:
            # 开新仓：多 @1，空 @2
            near_price = prices.get(near, pd.Series()).get(dt)
            far_price = prices.get(far, pd.Series()).get(dt)
            if near_price is None or far_price is None or near_price <= 0:
                daily_records.append({
                    'datetime': dt, 'pnl': 0.0, 'cost': 0.0, 'net_pnl': 0.0,
                    'pair': None, 'position_value': 0.0
                })
                continue

            # 每手名义市值 = 价格 * 乘数
            # 目标手数 = 每日目标名义市值 / 每手名义市值
            # 多空各一半资金
            target_lots = int(daily_target_nominal / 2 / near_price / multiplier)
            if target_lots < 1:
                target_lots = 1

            position[near] = target_lots
            position[far] = -target_lots
            current_pair = (near, far)
            pair_start_date = dt
            current_pair_cum_pnl = 0.0

            # 开仓手续费
            open_cost = target_lots * near_price * multiplier * commission + \
                        target_lots * far_price * multiplier * commission

            daily_records.append({
                'datetime': dt, 'pnl': 0.0, 'cost': -open_cost, 'net_pnl': -open_cost,
                'pair': f"{near} / {far}", 'position_value': daily_target_nominal
            })
            continue

        if not signal_active and position:
            # 信号消失，平仓
            record = close_position(dt, position, prices, commission, multiplier)
            current_pair_cum_pnl += record['net_pnl']
            if current_pair and pair_start_date:
                pair_records.append(finish_pair(current_pair, pair_start_date, dt, current_pair_cum_pnl))
            position = {}
            current_pair = None
            pair_start_date = None
            current_pair_cum_pnl = 0.0
            daily_records.append(record)
            continue

        if position:
            # 持有中：计算当日盈亏
            record = calculate_daily_pnl(dt, position, prices, multiplier)
            record['pair'] = f"{current_pair[0]} / {current_pair[1]}"
            record['position_value'] = daily_target_nominal
            current_pair_cum_pnl += record['net_pnl']
            daily_records.append(record)
        else:
            daily_records.append({
                'datetime': dt, 'pnl': 0.0, 'cost': 0.0, 'net_pnl': 0.0,
                'pair': None, 'position_value': 0.0
            })

    # 最后一天平仓
    if position:
        dt = all_dates[-1]
        record = close_position(dt, position, prices, commission, multiplier)
        current_pair_cum_pnl += record['net_pnl']
        if current_pair and pair_start_date:
            pair_records.append(finish_pair(current_pair, pair_start_date, dt, current_pair_cum_pnl))
        daily_records[-1] = record

    daily_df = pd.DataFrame(daily_records)
    daily_df.set_index('datetime', inplace=True)
    pair_df = pd.DataFrame(pair_records)

    # 5. 绩效统计
    print("\n[5/5] 计算绩效指标...")
    stats = calculate_stats(daily_df, capital)

    # 6. 保存结果
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(f"{output_dir}/{product.lower()}_only_vol{int(vol_threshold*100)}_pnl.csv")
        if not pair_df.empty:
            pair_df.to_csv(f"{output_dir}/{product.lower()}_only_pair_summary.csv", index=False)
        metrics = pd.DataFrame([{
            'name': f"{product}-only vol>={int(vol_threshold*100)}%",
            **stats
        }])
        metrics.to_csv(f"{output_dir}/{product.lower()}_only_metrics.csv", index=False)
        print(f"\n[OK] 结果保存到 {output_dir}")

    return {
        'daily_df': daily_df,
        'pair_df': pair_df,
        'stats': stats,
    }


def calculate_daily_pnl(dt: datetime, position: Dict[str, int], prices: Dict[str, pd.Series],
                        multiplier: int) -> Dict:
    """计算当日持仓盈亏（不计手续费）"""
    pnl = 0.0
    for vt_symbol, lots in position.items():
        price_series = prices.get(vt_symbol, pd.Series())
        if dt not in price_series.index:
            continue
        prev_dt = price_series.index[price_series.index < dt]
        if len(prev_dt) == 0:
            continue
        prev_price = price_series.loc[prev_dt[-1]]
        curr_price = price_series.loc[dt]
        pnl += lots * (curr_price - prev_price) * multiplier
    return {
        'datetime': dt,
        'pnl': pnl,
        'cost': 0.0,
        'net_pnl': pnl,
        'pair': None,
        'position_value': 0.0,
    }


def close_position(dt: datetime, position: Dict[str, int], prices: Dict[str, pd.Series],
                   commission: float, multiplier: int) -> Dict:
    """平仓并计算当日盈亏 + 平仓手续费"""
    record = calculate_daily_pnl(dt, position, prices, multiplier)
    close_cost = 0.0
    for vt_symbol, lots in position.items():
        price_series = prices.get(vt_symbol, pd.Series())
        if dt not in price_series.index:
            continue
        curr_price = price_series.loc[dt]
        close_cost += abs(lots) * curr_price * multiplier * commission
    record['cost'] = -close_cost
    record['net_pnl'] = record['pnl'] - close_cost
    record['pair'] = None
    return record


def finish_pair(pair: Tuple[str, str], start_dt: datetime, end_dt: datetime,
                cum_pnl: float) -> Dict:
    """记录一个套利对的汇总信息"""
    return {
        'product': pair[0][:2],
        'pair': f"{pair[0]} / {pair[1]}",
        'start': start_dt.strftime('%Y-%m-%d'),
        'end': end_dt.strftime('%Y-%m-%d'),
        'days': (end_dt - start_dt).days + 1,
        'total_pnl': cum_pnl,
    }


def calculate_stats(daily_df: pd.DataFrame, capital: int) -> Dict:
    """计算回测绩效指标"""
    net_pnl = daily_df['net_pnl']
    total_pnl = net_pnl.sum()
    total_return = total_pnl / capital

    # 日收益率
    daily_ret = net_pnl / capital
    active_days = daily_ret[daily_ret != 0]

    # 年化（按 250 日）
    annual_return = daily_ret.mean() * 250
    volatility = daily_ret.std() * np.sqrt(250)
    sharpe = annual_return / volatility if volatility > 0 else 0

    # 最大回撤
    cum = daily_ret.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()
    max_dd_pct = max_dd

    calmar = annual_return / abs(max_dd_pct) if max_dd_pct < 0 else 0

    profit_days = (net_pnl > 0).sum()
    loss_days = (net_pnl < 0).sum()
    win_rate = profit_days / (profit_days + loss_days) if (profit_days + loss_days) > 0 else 0

    # 保证金视角收益（IC/IM 保证金约 12%）
    margin_ratio = 0.12
    margin_total_return = total_return / margin_ratio if margin_ratio > 0 else 0
    margin_max_dd = max_dd_pct / margin_ratio if margin_ratio > 0 else 0

    return {
        'total_pnl': round(total_pnl, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd_pct,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_days': int(profit_days),
        'loss_days': int(loss_days),
        'days': int(len(daily_df)),
        'margin_total_return': margin_total_return,
        'margin_max_dd': margin_max_dd,
    }


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    result = run_backtest(
        product=args.product,
        vol_threshold=args.vol_threshold,
        start=start,
        end=end,
        capital=args.capital,
        commission=args.commission,
        output_dir=args.output_dir,
        plot=args.plot,
    )

    stats = result['stats']
    print("\n" + "="*70)
    print(f"【{args.product}-only vol>={args.vol_threshold*100:.0f}% 绩效】")
    print("="*70)
    for k, v in stats.items():
        if isinstance(v, float):
            if k in ['sharpe', 'calmar']:
                print(f"  {k:<25s}: {v:.4f}")
            elif k in ['win_rate']:
                print(f"  {k:<25s}: {v:.4%}")
            elif k in ['total_return', 'annual_return', 'max_dd', 'margin_total_return', 'margin_max_dd']:
                print(f"  {k:<25s}: {v:.4%}")
            else:
                print(f"  {k:<25s}: {v:.2f}")
        else:
            print(f"  {k:<25s}: {v}")
    print("="*70)


if __name__ == "__main__":
    main()
