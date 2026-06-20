import numpy as np
import pandas as pd
from typing import Dict

def compute(cache: 'DataCache', params: Dict) -> pd.DataFrame:
    """
    Nelson-Siegel Slope Factor - Enhanced Version
    利用888(主力)、889(次主力)、99(指数)三个合约构建期限结构因子
    
    核心逻辑：
    1. 计算888 vs 889的价差（期限结构斜率代理）
    2. 计算99 vs 888的偏离（指数与主力偏离）
    3. 结合两者构建综合斜率因子
    4. 使用Z-Score标准化捕捉极端偏离
    
    论文参考：Bianchi et al. (2023) - Slope Factor Strategy
    """
    cycle = params.get('cycle', 5)
    zscore_lookback = params.get('zscore_lookback', 60)
    
    # 获取三个合约的价格
    # 888 = 主力连续合约
    # 889 = 次主力连续合约  
    # 99 = 指数合约（全期限加权平均）
    
    # 从cache获取数据
    f1_close = cache._f1_close  # 888合约
    f2_close = cache._f2_close  # 889合约
    
    # 尝试获取99合约（指数）
    # 99合约的命名规则：品种代码 + '99'
    symbols = cache._symbols
    f3_close = None
    
    # 构建99合约的symbol映射
    # 888合约的symbol格式如：'RB888.SHFE'
    # 对应的99合约应该是：'RB99.SHFE'
    f3_symbols = []
    for sym in symbols:
        # 提取品种代码（去掉888）
        if '888' in sym:
            base = sym.replace('888', '99')
            f3_symbols.append(base)
        else:
            f3_symbols.append(sym)
    
    # 尝试从数据库加载99合约数据
    try:
        from vnpy.trader.database import get_database
        from vnpy.trader.constant import Exchange, Interval
        import pandas as pd
        
        db = get_database()
        f3_data = {}
        
        for i, sym in enumerate(symbols):
            # 解析symbol和exchange
            parts = sym.split('.')
            if len(parts) == 2:
                symbol_code, exchange_str = parts
                # 构建99合约代码
                symbol_99 = symbol_code.replace('888', '99')
                
                # 映射交易所字符串到Exchange枚举
                exchange_map = {
                    'SHFE': Exchange.SHFE,
                    'DCE': Exchange.DCE,
                    'CZCE': Exchange.CZCE,
                    'CFFEX': Exchange.CFFEX,
                    'INE': Exchange.INE,
                }
                exchange = exchange_map.get(exchange_str, Exchange.SHFE)
                
                # 查询数据库
                bars = db.load_bar_data(symbol_99, exchange, interval=Interval.DAILY, start=cache._start, end=cache._end)
                if bars:
                    df = pd.DataFrame({
                        'datetime': [b.datetime for b in bars],
                        'close': [b.close_price for b in bars]
                    })
                    df.set_index('datetime', inplace=True)
                    f3_data[sym] = df['close']
        
        if f3_data:
            f3_close = pd.DataFrame(f3_data)
            f3_close.index = pd.to_datetime(f3_close.index)
        else:
            f3_close = None
            
    except Exception as e:
        print(f"[NS-Enhanced] 无法加载99合约数据: {e}")
        f3_close = None
    
    # 计算基础价差（888 vs 889）
    spread = f1_close - f2_close
    spread_pct = spread / f2_close * 100  # 百分比价差
    
    # 如果99合约数据可用，计算额外信息
    if f3_close is not None:
        # 对齐索引
        common_idx = f1_close.index.intersection(f3_close.index)
        if len(common_idx) > 0:
            f1_aligned = f1_close.loc[common_idx]
            f3_aligned = f3_close.loc[common_idx]
            
            # 指数 vs 主力偏离
            index_deviation = (f3_aligned - f1_aligned) / f1_aligned * 100
            
            # 综合斜率：价差 + 指数偏离
            combined_slope = spread_pct.loc[common_idx] + index_deviation
        else:
            combined_slope = spread_pct
            index_deviation = None
    else:
        combined_slope = spread_pct
        index_deviation = None
    
    # 计算价差的Z-Score（捕捉极端偏离）
    # 使用滚动窗口计算均值和标准差
    spread_mean = combined_slope.rolling(window=zscore_lookback, min_periods=zscore_lookback//2).mean()
    spread_std = combined_slope.rolling(window=zscore_lookback, min_periods=zscore_lookback//2).std()
    
    # 避免除以零
    spread_std = spread_std.replace(0, np.nan)
    
    zscore = (combined_slope - spread_mean) / spread_std
    
    # 计算斜率变化率（动量）
    slope_change = combined_slope.diff(cycle)
    
    # 综合因子：Z-Score + 斜率变化
    # 当Z-Score为正（斜率陡峭）且斜率在变陡 → 因子值高
    # 当Z-Score为负（斜率平坦）且斜率在变平 → 因子值低
    
    # 对Z-Score和斜率变化进行排名标准化
    zscore_rank = zscore.rank(axis=1, pct=True)
    slope_change_rank = slope_change.rank(axis=1, pct=True)
    
    # 综合因子（等权）
    factor = 0.6 * zscore_rank + 0.4 * slope_change_rank
    
    # 处理缺失值
    factor = factor.fillna(method='ffill', limit=5).fillna(0.5)
    
    # 确保输出格式与输入一致
    factor = factor.loc[f1_close.index]
    
    print(f"[NS-Enhanced] 计算完成: cycle={cycle}, zscore_lookback={zscore_lookback}")
    print(f"[NS-Enhanced] 使用合约: 888(主力), 889(次主力), 99(指数)")
    if f3_close is not None:
        print(f"[NS-Enhanced] 99合约数据: 已加载")
    else:
        print(f"[NS-Enhanced] 99合约数据: 未加载，仅使用888+889")
    
    return factor
