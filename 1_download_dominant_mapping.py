#!/usr/bin/env python3
"""
Step 3: RQData主力信息下载（标准化流程）

严格参照 /root/long-short-term-strategy-revise/script/3 - RQData主力信息下载.ipynb

功能：
1. 从 RQData 下载所有期货品种的主力(rank=1)和次主力(rank=2)合约映射
2. 保存到 vnpy_dominant_contract 表（ClickHouse）

命名规范：
- @1 = 主力合约映射
- @2 = 次主力合约映射
- key = "{product}88.{exchange}@{rank}"
- value = 具体合约代码，如 "RB2310.SHFE"
"""

import rqdatac
from vnpy.trader.setting import SETTINGS
from vnpy_alpharesearch import DataCenter


def main():
    print("=" * 70)
    print("Step 3: RQData 主力信息下载（标准化流程）")
    print("=" * 70)

    # 实例化数据中心
    dc = DataCenter()

    # 登录 RQData
    print("\n登录 RQData...")
    rqdatac.init(
        username=SETTINGS["datafeed.username"],
        password=SETTINGS["datafeed.password"],
        addr=("212.64.120.155", 16011)
    )
    print("RQData 登录成功")

    # 创建/确认主力合约表
    print("\n确认 vnpy_dominant_contract 表...")
    dc.create_reference_table(["vnpy_dominant_contract"])
    print("表已就绪")

    # 获取期货品种信息
    print("\n获取期货品种列表...")
    contract_df = rqdatac.all_instruments(type="Future")
    products = set()
    for tp in contract_df.itertuples():
        products.add((tp.underlying_symbol, tp.exchange))
    print(f"共 {len(products)} 个期货品种")

    # 主力合约后缀
    DOMINANT_SUFFIX = 88

    # 遍历下载入库
    print("\n开始下载主力/次主力映射...")
    success = 0
    skipped = 0
    for product, exchange in sorted(products):
        data1 = rqdatac.futures.get_dominant(product, rank=1)
        data2 = rqdatac.futures.get_dominant(product, rank=2)

        if data1 is None:
            print(f"  ⚠️ {product}.{exchange}: 无主力(rank=1)数据")
            skipped += 1
            continue

        if data2 is None:
            print(f"  ⚠️ {product}.{exchange}: 无次主力(rank=2)数据")
            skipped += 1
            continue

        # 添加交易所后缀
        data1 = data1 + "." + exchange
        data2 = data2 + "." + exchange

        # 写入主力映射关系
        key1 = f"{product}{DOMINANT_SUFFIX}.{exchange}@1"
        key2 = f"{product}{DOMINANT_SUFFIX}.{exchange}@2"

        dc.save_reference_series("vnpy_dominant_contract", key1, data1)
        dc.save_reference_series("vnpy_dominant_contract", key2, data2)

        print(f"  ✅ {product}.{exchange}: {len(data1)} days (rank1), {len(data2)} days (rank2)")
        success += 1

    print(f"\n{'=' * 70}")
    print(f"下载完成: 成功={success}, 跳过={skipped}")
    print(f"{'=' * 70}")

    # 验证
    print("\n验证数据...")
    for key in ["RB88.SHFE@1", "RB88.SHFE@2", "CU88.SHFE@1", "CU88.SHFE@2"]:
        s = dc.load_reference_series("vnpy_dominant_contract", key, None, None)
        if s is not None:
            print(f"  {key}: {len(s)} days, first={s.index[0].date()}, last={s.index[-1].date()}")
        else:
            print(f"  {key}: NOT FOUND")


if __name__ == "__main__":
    main()
