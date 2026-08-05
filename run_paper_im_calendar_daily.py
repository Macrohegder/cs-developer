#!/usr/bin/env python3
"""
IM 日历价差策略 — paper 日度信号与盈亏跟踪（半规模 5M）

每日流程：
1. 数据保障：通过 RQData（vnpy datafeed，唯一授权入口）补齐当前合约对的日线到最新，
   并校验主力映射表新鲜度（映射由 rq_data/download_cn_rqdata.py 每日 23:00 任务写入）。
2. 以半规模资金（500 万）重放 EnhancedCalendarSpreadStrategy 全历史，取最新交易日目标仓位。
3. paper 记账：以 config.paper_start_date 为起点累计 net_pnl（与回测口径、手续费一致）。
4. 输出：
   - 状态:  paper/im_calendar_spread/state.json
   - 明细:  paper/im_calendar_spread/daily_pnl.csv
   - 日报:  tasks/results/{信号日期}-daily_pnl-paper_im_calendar_spread.md

调度：系统 crontab 每日 01:30（在 23:00 RQData 更新任务完成之后）。
"""

import json
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

import pytz

_original_timezone = pytz.timezone


def _patched_timezone(zone):
    if zone == "Asia/Beijing":
        return _original_timezone("Asia/Shanghai")
    return _original_timezone(zone)


pytz.timezone = _patched_timezone

CS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CS_ROOT))
sys.path.insert(0, str(CS_ROOT / "factor_system"))

from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.utility import ZoneInfo
from vnpy_alpharesearch import DataCenter
from vnpy_alpharesearch.utility import load_history_df
from vnpy_alpharesearch.strategy.backtester import StrategyBacktester
from vnpy_alpharesearch.strategy.analysis import calculate_portfolio_performance

import run_latest_calendar_signal as sig
from strategies.enhanced_calendar_spread_strategy import EnhancedCalendarSpreadStrategy

CHINA_TZ = ZoneInfo("Asia/Shanghai")

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=str(CS_ROOT / "paper" / "im_calendar_spread" / "config.json"),
                 help="paper 账户配置 json 路径")
_args = _ap.parse_args()

CONFIG_PATH = Path(_args.config).resolve()
PAPER_DIR = CONFIG_PATH.parent
STATE_PATH = PAPER_DIR / "state.json"
DAILY_CSV = PAPER_DIR / "daily_pnl.csv"
REPORT_DIR = Path("/root/quant/tasks/results")

CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CAPITAL = CONFIG["capital"]
PRODUCTS = CONFIG["products"]
COMMISSION = CONFIG["commission"]
PAPER_START = pd.Timestamp(CONFIG["paper_start_date"])
START = datetime.strptime(CONFIG["data_start"], "%Y-%m-%d")
ACCOUNT_ID = CONFIG.get("account_id", PAPER_DIR.name)

# 研究口径 10M → 半规模 5M：覆盖被复用模块中的资金常量
sig.CAPITAL = CAPITAL
sig.PRODUCTS = PRODUCTS


# ---------------------------------------------------------------------------
# 数据保障
# ---------------------------------------------------------------------------

def ensure_contract_daily_bars(symbols: list[str]) -> dict:
    """通过 RQData 补齐指定合约的日线到最新，返回 {symbol: 写入条数}"""
    import rqdatac as rq
    from clickhouse_driver import Client

    client = Client(host="localhost")
    db = get_database()
    written = {}
    fields = ["open", "high", "low", "close", "volume", "total_turnover", "open_interest"]
    for vt_symbol in symbols:
        symbol, exchange = vt_symbol.split(".")
        rows = client.execute(
            "SELECT max(datetime) FROM vnpy.bar_data WHERE symbol=%(s)s AND interval='d'",
            {"s": symbol},
        )
        last_dt = rows[0][0] if rows and rows[0][0] and rows[0][0].year > 1971 else None
        start_d = (last_dt.date() + timedelta(days=1)) if last_dt else START.date()
        end_d = date.today()
        if start_d > end_d:
            written[vt_symbol] = 0
            continue
        df = rq.get_price(symbol, start_date=start_d.strftime("%Y%m%d"),
                          end_date=end_d.strftime("%Y%m%d"), frequency="1d", fields=fields)
        if df is None or df.empty:
            written[vt_symbol] = 0
            continue
        df = df.fillna(0)
        if isinstance(df.index, pd.MultiIndex):
            dts = df.index.get_level_values(1)
        else:
            dts = df.index
        bars = []
        for (_, row), idx in zip(df.iterrows(), dts):
            dt = pd.Timestamp(idx).to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CHINA_TZ)
            else:
                dt = dt.astimezone(CHINA_TZ)
            bars.append(BarData(
                symbol=symbol, exchange=Exchange(exchange), interval=Interval.DAILY,
                datetime=dt,
                open_price=round(float(row["open"]), 6),
                high_price=round(float(row["high"]), 6),
                low_price=round(float(row["low"]), 6),
                close_price=round(float(row["close"]), 6),
                volume=float(row["volume"]), turnover=float(row["total_turnover"]),
                open_interest=float(row["open_interest"]), gateway_name="RQ",
            ))
        db.save_bar_data(bars)
        written[vt_symbol] = len(bars)
    return written


def check_mapping_freshness(latest_bar_date: pd.Timestamp) -> tuple[bool, str]:
    """校验主力映射表是否覆盖到最新交易日"""
    from clickhouse_driver import Client
    client = Client(host="localhost")
    keys = [f"{p}88.CFFEX@{r}" for p in PRODUCTS for r in (1, 2)]
    rows = client.execute(
        "SELECT key, max(datetime) FROM vnpy.vnpy_dominant_contract "
        "WHERE key IN %(k)s GROUP BY key", {"k": keys},
    )
    stale = [f"{k} 映射仅到 {v.date()}" for k, v in rows if v.date() < latest_bar_date.date()]
    return (not stale), "; ".join(stale)


# ---------------------------------------------------------------------------
# 回测与信号
# ---------------------------------------------------------------------------

def run_strategy(latest: datetime):
    """重放全历史，返回 (target_df, overall_df, history_df, mappings, idx_closes)"""
    dc = DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    data_cache = {p: sig.prepare_product_data(p, contract_df, latest) for p in PRODUCTS}
    all_contracts = sorted(set(c for p in PRODUCTS for c in data_cache[p]["contracts"]))
    history_df = load_history_df(all_contracts, Interval.DAILY, START, latest)

    backtester = StrategyBacktester(vt_symbols=all_contracts, interval=Interval.DAILY,
                                    start=START, end=latest, capital=CAPITAL)
    backtester.history_df = history_df
    backtester.contract_df = contract_df
    symbol_ix = contract_df.index + "." + contract_df["exchange"]
    backtester.multiplier_series = pd.Series(contract_df["size"].values, index=symbol_ix)

    dominant_symbols = history_df.columns.get_level_values(0).drop_duplicates()
    intraday_list, close_list = [], []
    for vt in dominant_symbols:
        o = history_df[(vt, "open_price")]
        c = history_df[(vt, "close_price")]
        intraday_list.append(c - o)
        close_list.append(c - c.shift(1))
    backtester.intraday_change_df = pd.concat(intraday_list, axis=1)
    backtester.intraday_change_df.columns = dominant_symbols
    backtester.close_change_df = pd.concat(close_list, axis=1)
    backtester.close_change_df.columns = dominant_symbols

    mappings, idx_closes = {}, {}
    for p in PRODUCTS:
        mappings[p] = data_cache[p]["mapping"].copy().reindex(history_df.index)
        idx_closes[p] = data_cache[p]["idx_close"]

    setting = {
        "products": PRODUCTS,
        "leverage": CONFIG["leverage"],
        "capital_ratio": CONFIG["capital_ratio"],
        "mapping": mappings,
        "idx_close": idx_closes,
        "contract_df": contract_df,
        "history_df": history_df,
        "vol_threshold": CONFIG["vol_threshold"],
        "vol_window": CONFIG["vol_window"],
        "stop_loss_pct": CONFIG["stop_loss_pct"],
        "take_profit_pct": CONFIG["take_profit_pct"],
        "only_quarterly": CONFIG["only_quarterly"],
        "roll_before_days": CONFIG["roll_before_days"],
        "use_quarterly_pair": CONFIG["use_quarterly_pair"],
        "dynamic_direction": CONFIG["dynamic_direction"],
        "direction": CONFIG["direction"],
    }
    target_df = backtester.run_backtesting(EnhancedCalendarSpreadStrategy, setting)
    result = calculate_portfolio_performance(
        target_df=target_df, interval=Interval.DAILY,
        commission=COMMISSION, capital=CAPITAL, plot_chart=False,
    )
    return target_df, result["overall"], history_df, mappings, idx_closes


def compute_vol20(idx_close: pd.Series, end: pd.Timestamp) -> float:
    """88 指数 20 日年化波动率"""
    ret = idx_close.pct_change().dropna()
    ret = ret[ret.index <= end].tail(CONFIG["vol_window"])
    return float(ret.std() * np.sqrt(240)) if len(ret) else float("nan")


# ---------------------------------------------------------------------------
# 风控警报（三条警报线，口径见 tasks/results 基差报告 / OPTIMIZATION_REPORT）
# ---------------------------------------------------------------------------

# 阈值可通过 config.json 覆盖
ALERT_PCT_FLOOR = CONFIG.get("alert_pct_floor", 0.10)    # 价差率分位 <10% → 🔴
ALERT_PCT_WATCH = CONFIG.get("alert_pct_watch", 0.25)    # 分位 <25% → 🟡
ALERT_CARRY_FLOOR = CONFIG.get("alert_carry_floor", 0.04)  # 年化 carry <4% → 🔴
ALERT_CARRY_WATCH = CONFIG.get("alert_carry_watch", 0.06)  # carry <6% → 🟡
ALERT_DD_MULT = CONFIG.get("alert_dd_mult", 2.0)         # paper 回撤 > 2×回测 maxDD → 🔴
SPREAD_LOOKBACK = 250  # 分位窗口（交易日）


def _third_friday(year: int, month: int) -> date:
    """合约到期日 = 合约月份第三个周五（即 15~21 日中的那个周五）"""
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _contract_expiry(vt_symbol: str) -> date:
    """IM2609.CFFEX → 2026-09 第三个周五"""
    code = vt_symbol.split(".")[0]
    yy, mm = int(code[-4:-2]), int(code[-2:])
    return _third_friday(2000 + yy, mm)


def compute_spread_metrics(mapping: pd.DataFrame, close_df: pd.DataFrame,
                           k1: str, k2: str, last_dt: pd.Timestamp) -> dict:
    """主力对(@1/@2)价差率近 250 日分位 + 按两腿到期间隔年化的毛 carry"""
    idx = close_df.index[close_df.index <= last_dt][-SPREAD_LOOKBACK:]
    rates = []
    for d in idx:
        near, far = mapping.loc[d, k1], mapping.loc[d, k2]
        if not (isinstance(near, str) and isinstance(far, str)):
            continue
        if near not in close_df.columns or far not in close_df.columns:
            continue
        np_, fp = close_df.loc[d, near], close_df.loc[d, far]
        if pd.isna(np_) or pd.isna(fp) or not np_:
            continue
        rates.append((np_ - fp) / np_)
    if not rates:
        return {"percentile": float("nan"), "carry": float("nan"), "n": 0}
    cur = rates[-1]
    percentile = float(np.mean([r <= cur for r in rates]))
    near_now, far_now = mapping.loc[last_dt, k1], mapping.loc[last_dt, k2]
    days = (_contract_expiry(far_now) - _contract_expiry(near_now)).days
    carry = float(cur * 365 / days) if days > 0 else float("nan")
    return {"percentile": percentile, "carry": carry, "n": len(rates)}


def _max_drawdown_pct(net_pnl: pd.Series, capital: float) -> float:
    """最大回撤（相对本金口径，与 OPTIMIZATION_REPORT 一致）"""
    if not len(net_pnl):
        return 0.0
    balance = capital + net_pnl.cumsum()
    return float(((balance.cummax() - balance) / capital).max())


def evaluate_alerts(spread_pct: float, spread_metrics: dict, vol20: float,
                    holding: bool, overall: pd.DataFrame,
                    paper_pnl: pd.Series, vol_threshold: float) -> list[str]:
    """返回警报消息列表（🟡 关注 / 🔴 警报），空列表 = 正常"""
    alerts = []
    pct, carry = spread_metrics["percentile"], spread_metrics["carry"]

    # 1. carry 厚度：分位 + 年化 carry 双口径
    if not pd.isna(pct) and not pd.isna(carry):
        if pct < ALERT_PCT_FLOOR or carry < ALERT_CARRY_FLOOR:
            alerts.append(f"🔴 carry 枯竭：价差率 {spread_pct:.2%}（分位 {pct:.0%} < {ALERT_PCT_FLOOR:.0%}）"
                          f"，毛 carry {carry:.1%}（警戒线 {ALERT_CARRY_FLOOR:.0%}）→ 停止新开仓")
        elif pct < ALERT_PCT_WATCH or carry < ALERT_CARRY_WATCH:
            alerts.append(f"🟡 carry 变薄：价差率 {spread_pct:.2%}（分位 {pct:.0%}），"
                          f"毛 carry {carry:.1%}，新开仓性价比偏低")

    # 2. 波动率 regime：策略开仓前提（20 日年化波动 ≥ vol_threshold）
    if not pd.isna(vol20) and vol20 < vol_threshold:
        tag = "持仓中，注意信号可能失效" if holding else "当前空仓，属正常过滤"
        alerts.append(f"🔴 波动率跌破阈值：20 日年化 {vol20:.1%} < {vol_threshold:.0%}（{tag}）")

    # 3. 回撤超设计：paper 回撤 > 2× 全历史回测最大回撤
    bt_dd = _max_drawdown_pct(overall["net_pnl"], CAPITAL)
    paper_dd = _max_drawdown_pct(paper_pnl, CAPITAL)
    if bt_dd > 0 and paper_dd > ALERT_DD_MULT * bt_dd:
        alerts.append(f"🔴 回撤超设计：paper 回撤 {paper_dd:.2%} > {ALERT_DD_MULT:.0f}×回测 maxDD "
                      f"{bt_dd:.2%} → 市场结构偏离回测样本，停止加仓并复查")
    return alerts


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 66)
    print(f"日历价差 paper 跟踪（{ACCOUNT_ID}，{','.join(PRODUCTS)}，本金 {CAPITAL/1e4:.0f} 万）")
    print("=" * 66)

    # 0. RQData 初始化
    feed = get_datafeed()
    feed.init(output=lambda x: print(f"[datafeed] {x}"))

    # 0.5 数据保障（傍晚运行场景）：先从 RQData 补齐 88 指数日线到最新，
    # 使信号日期不受 23:00 批量下载任务时序约束（RQData 无今日数据时自动
    # 回退到最近可用交易日，行为与原来一致）
    written_88 = ensure_contract_daily_bars([f"{p}88.CFFEX" for p in PRODUCTS])
    print(f"88 日线补齐: {written_88}")

    # 1. 确定最新可用交易日（88 日线）
    latest = sig.get_latest_common_date()
    latest_ts = pd.Timestamp(latest)
    print(f"\n数据库最新交易日: {latest_ts.date()}")

    # 2. 数据保障：补齐当前映射合约对的日线
    recent_contracts = set()
    for p in PRODUCTS:
        mapping_now = sig.load_dominant_mapping(p, latest - timedelta(days=10), latest)
        for col in mapping_now.columns:
            recent_contracts.update(v for v in mapping_now[col].dropna().unique() if isinstance(v, str))
    recent_contracts = sorted(c for c in recent_contracts if "." in c)
    print(f"近期映射合约: {recent_contracts}")
    written = ensure_contract_daily_bars(recent_contracts)
    print(f"日线补齐: {written}")

    ok_map, map_msg = check_mapping_freshness(latest_ts)
    if not ok_map:
        print(f"[警告] 主力映射不新鲜: {map_msg}")

    # 3. 重放策略取最新目标
    target_df, overall, history_df, mappings, idx_closes = run_strategy(latest)
    last_dt = target_df.index[-1]
    close_df = history_df.xs("close_price", axis=1, level=1)

    # 4. paper 记账（从 paper_start_date 起累计）
    paper = overall[overall.index >= PAPER_START].copy()
    paper_equity = CAPITAL + paper["net_pnl"].cumsum()
    today_pnl = float(overall.loc[last_dt, "net_pnl"]) if last_dt >= PAPER_START else 0.0
    cum_pnl = float(paper["net_pnl"].sum()) if len(paper) else 0.0
    equity = float(paper_equity.iloc[-1]) if len(paper_equity) else CAPITAL

    # 5. 提取信号
    p0 = PRODUCTS[0]
    k1, k2 = f"{p0}88.CFFEX@1", f"{p0}88.CFFEX@2"
    mapping = mappings[p0]
    near = mapping.loc[last_dt, k1] if last_dt in mapping.index else None
    far = mapping.loc[last_dt, k2] if last_dt in mapping.index else None
    near_lots = int(target_df.loc[last_dt, near]) if near in target_df.columns else 0
    far_lots = int(target_df.loc[last_dt, far]) if far in target_df.columns else 0
    near_px = float(close_df.loc[last_dt, near]) if near in close_df.columns else float("nan")
    far_px = float(close_df.loc[last_dt, far]) if far in close_df.columns else float("nan")
    spread_pct = (near_px - far_px) / near_px if near_px and not pd.isna(near_px) else float("nan")
    vol20 = compute_vol20(idx_closes[p0], last_dt)

    # 5.5 风控警报（carry 厚度 / 波动率 regime / 回撤超设计）
    spread_metrics = compute_spread_metrics(mapping, close_df, k1, k2, last_dt)
    holding_for_alert = bool(near_lots or far_lots)
    paper_pnl = paper["net_pnl"]
    vol_thr_cfg = CONFIG["vol_threshold"]
    vol_thr = float(vol_thr_cfg.get(p0, 0.15)) if isinstance(vol_thr_cfg, dict) else float(vol_thr_cfg)
    alerts = evaluate_alerts(spread_pct, spread_metrics, vol20,
                             holding_for_alert, overall, paper_pnl, vol_thr)

    # 6. 与前一日状态对比
    prev = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    prev_pos = prev.get("position", {})
    cur_pos = {"near": near, "far": far, "near_lots": near_lots, "far_lots": far_lots}
    holding = bool(near_lots or far_lots)
    if not prev:
        action = "首次建仓" if holding else "空仓等待（首日）"
    else:
        changed = cur_pos != {k: prev_pos.get(k) for k in cur_pos}
        action = ("开仓/调仓" if holding else "平仓") if changed else ("持仓不变" if holding else "空仓等待")

    # 7. 落盘
    state = {
        "account_id": CONFIG["account_id"],
        "signal_date": str(last_dt.date()),
        "position": cur_pos,
        "prices": {"near": near_px, "far": far_px, "spread_pct": spread_pct},
        "vol20": vol20,
        "spread_percentile": spread_metrics["percentile"],
        "carry_annual": spread_metrics["carry"],
        "alerts": alerts,
        "paper_equity": equity,
        "cum_pnl": cum_pnl,
        "updated_at": datetime.now(CHINA_TZ).isoformat(),
    }
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    row = pd.DataFrame([{
        "date": str(last_dt.date()), "near": near, "far": far,
        "near_lots": near_lots, "far_lots": far_lots,
        "near_px": near_px, "far_px": far_px, "spread_pct": spread_pct,
        "spread_percentile": spread_metrics["percentile"],
        "carry_annual": spread_metrics["carry"],
        "vol20": vol20, "daily_pnl": today_pnl, "cum_pnl": cum_pnl, "equity": equity,
    }])
    if DAILY_CSV.exists():
        old = pd.read_csv(DAILY_CSV)
        old = old[old["date"] != str(last_dt.date())]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(DAILY_CSV, index=False)

    # 8. 日报
    data_alert = "" if ok_map else f"\n> ⚠️ **数据滞后：{map_msg}。今日信号不可信，请勿据此交易，先修复数据。**\n"
    if alerts:
        alert_section = "\n".join(f"- {a}" for a in alerts)
    else:
        alert_section = (f"- 🟢 正常：价差率分位 {spread_metrics['percentile']:.0%}，"
                         f"毛 carry {spread_metrics['carry']:.1%}，波动率与回撤均在设计范围内")
    report = f"""# Daily P&L 报告：{ACCOUNT_ID}（{','.join(PRODUCTS)} 跨期 · {CAPITAL/1e4:.0f} 万）
**信号日期**: {last_dt.date()}
**生成时间**: {datetime.now(CHINA_TZ).isoformat(timespec="seconds")}
{data_alert}
## 风控警报

{alert_section}

## 数据 freshness

| 项 | 状态 |
|---|---|
| 88 日线 | 至 {latest_ts.date()} |
| 主力映射 | {"✅ 正常" if ok_map else "⚠️ " + map_msg} |
| 合约日线补齐 | {written} |

## 最新信号

| 项 | 值 |
|---|---|
| 近月合约 | {near} @ {near_px:.2f} |
| 远月合约 | {far} @ {far_px:.2f} |
| spread_pct | {spread_pct:.4%} |
| {p0}88 20日年化波动 | {vol20:.2%}（开仓阈值 ≥15%） |
| 方向 | {"+1 多近空远" if near_lots > 0 else "空仓"} |
| 近月目标手数 | {near_lots} |
| 远月目标手数 | {far_lots} |
| 动作 | **{action}** |

## paper 账户（{CAPITAL/1e4:.0f} 万，自 {PAPER_START.date()} 起）

- **当日盈亏**: {today_pnl:,.2f} CNY
- **累计盈亏**: {cum_pnl:,.2f} CNY
- **paper 权益**: {equity:,.2f} CNY

---
*由 cs_developer/run_paper_im_calendar_daily.py 生成；策略口径见 {CONFIG_PATH.relative_to(CS_ROOT)}*
"""
    report_path = REPORT_DIR / f"{last_dt.date()}-daily_pnl-{ACCOUNT_ID}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n信号: {near} {near_lots}手 / {far} {far_lots}手  动作={action}")
    print(f"当日盈亏 {today_pnl:,.0f} | 累计 {cum_pnl:,.0f} | 权益 {equity:,.0f}")
    print(f"风控警报: {len(alerts)} 条" + ("" if not alerts else " — " + "；".join(alerts)))
    print(f"日报: {report_path}")


if __name__ == "__main__":
    main()
