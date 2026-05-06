#!/usr/bin/env python3
import subprocess
import pandas as pd

holding_periods = [5, 10, 20]
trading_signals = [0.1, 0.2, 0.3]

results = []
total = len(holding_periods) * len(trading_signals)
count = 0

for hp in holding_periods:
    for ts in trading_signals:
        count += 1
        print(f"\n{'='*60}")
        print(f"[{count}/{total}] hp={hp}, ts={ts:.0%}")
        print(f"{'='*60}")
        
        cmd = [
            "python3", "run_carry_exact_backtest.py",
            "--start", "2015-01-01",
            "--end", "2024-12-31",
            "--commission", "0.0001",
            "--holding-period", str(hp),
            "--trading-signal", str(ts),
            "--skip-compute",
        ]
        
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=300)
            
            total_ret = None
            annual_ret = None
            sharpe = None
            max_dd = None
            for line in output.split('\n'):
                if '总收益:' in line:
                    total_ret = line.split(':')[1].strip()
                elif '年化收益:' in line:
                    annual_ret = line.split(':')[1].strip()
                elif '夏普比率:' in line:
                    sharpe = line.split(':')[1].strip()
                elif '最大回撤:' in line:
                    max_dd = line.split(':')[1].strip()
            
            results.append({
                'hp': hp, 'ts': ts,
                'total_return': total_ret,
                'annual_return': annual_ret,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
            })
            
            print(f"  结果: 总收益={total_ret}, 年化={annual_ret}, 夏普={sharpe}, 回撤={max_dd}")
            
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] 超时")
            results.append({'hp': hp, 'ts': ts, 'total_return': 'TIMEOUT', 'annual_return': '', 'sharpe': '', 'max_drawdown': ''})
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({'hp': hp, 'ts': ts, 'total_return': 'ERROR', 'annual_return': '', 'sharpe': '', 'max_drawdown': ''})

print(f"\n{'='*60}")
print("参数优化结果汇总")
print(f"{'='*60}")

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv('/root/cs_developer/carry_grid_results.csv', index=False)
print(f"\n[OK] 结果已保存到 carry_grid_results.csv")
