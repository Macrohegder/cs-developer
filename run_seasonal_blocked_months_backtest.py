#!/usr/bin/env python3
"""
季节性月份过滤增强回测验证
============================
基于 /tmp/all_spread_pairs_v2.pkl 的季节性分析结论：
- 12 月跨年代/跨品种稳定为负 carry（2015-2021: -0.49bp/d, 2022+: -0.91bp/d; IC -0.78, IM -0.53）
- 1 月、4 月在 2022+ 年代为负（但 2015-2021 为正，稳定性差）
- 11 月稳定为正

变体：
- base      : 当前最优参数（OPTIMAL_PARAMS）
- dec       : base + blocked_months=[12]
- dec_jan_apr: base + blocked_months=[1,4,12]

口径与 run_optimal_calendar_spread.py 完全一致（capital 10M, commission 1e-4），
END 统一到 2026-06-30 与组合分析窗口对齐。
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, '/root/quant/cs_developer')
sys.path.insert(0, '/root/quant')

import run_optimal_calendar_spread as R

OUT = Path('/root/quant/cs_developer/docs/ic_im_vol_scan')

VARIANTS = {
    'base': {},
    'dec': {'blocked_months': [12]},
    'dec_jan_apr': {'blocked_months': [1, 4, 12]},
}

R.END = datetime(2026, 6, 30)


def main():
    from cs_developer.run_optimal_calendar_spread import DataCenter  # noqa
    dc = R.DataCenter()
    contract_df = dc.load_contract_df()
    contract_df.index = contract_df.index.astype(str)
    contract_df["exchange"] = contract_df["exchange"].astype(str)

    data_cache = {
        'IC': R.prepare_product_data('IC', contract_df),
        'IM': R.prepare_product_data('IM', contract_df),
    }

    rows = []
    overalls = {}
    for vname, extra in VARIANTS.items():
        R.OPTIMAL_PARAMS = {
            "vol_threshold": 0.15,
            "vol_window": 20,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "only_quarterly": True,
            "roll_before_days": 5,
            **extra,
        }
        for product in ['IC', 'IM']:
            print(f"\n>>> {vname} / {product} ...")
            res = R.run_backtest(product, data_cache, contract_df)
            key = f'{vname}_{product.lower()}'
            overalls[key] = res['overall']
            res['overall'][["balance", "drawdown", "ddpercent", "net_pnl", "return"]].to_csv(
                OUT / f"seasonal_{key}_pnl.csv")
            exposure = (res['target_df'].abs().sum(axis=1) > 0).mean()
            rows.append({
                'variant': vname, 'product': product,
                'annual_return': res['annual_return'],
                'max_dd': res['max_dd'], 'sharpe': res['sharpe'],
                'calmar': res['calmar'], 'exposure': exposure,
            })
            print(f"  ann={res['annual_return']:.2%} sharpe={res['sharpe']:.3f} "
                  f"dd={res['max_dd']:.2%} expo={exposure:.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'seasonal_blocked_months_metrics.csv', index=False)
    print('\n=== 汇总 ===')
    print(df.pivot(index='variant', columns='product',
                   values=['annual_return', 'sharpe', 'max_dd']).round(4).to_string())


if __name__ == '__main__':
    main()
