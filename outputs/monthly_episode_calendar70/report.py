from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
r,f,v=[read(n) for n in ('RESULTS.json','FROZEN.json','VALIDATION.json')]
lines=['月末历史回撤与真实持有周期','2016-08-17至2026-09-24；同完整收盘信号/成交、零成本理想化回放。','使用同一月末信号历史，配对比较21交易日与下一实际月末的反弹效用；两臂共同样本、共同成熟日，保持固定查询与原评分预算。',
       '先用前88期及分段门槛选择并冻结，再检查后34期。研究历史已被反复使用，不是新的独立样本外验证。',
       '名称 | 全期年化 | Sharpe | 最大回撤 | 前88期年化 | 后34期年化 | 次日开盘每边5bp年化']
for name,z in r.items():
 a,c=z['metrics'],z['slices']
 lines.append(f"{name} | {a['cagr']:.2%} | {a['sharpe']:.3f} | {a['max_drawdown']:.2%} | {c['training2016_2023']['cagr']:.2%} | {c['later34']['cagr']:.2%} | {z['next_open_cost']['metrics']['cagr']:.2%}")
lines += ['',f"主要候选：{f['selected']['primary']['spec']}。新通过门槛：{f['new_admitted']}。",
          f"主要候选达到70%：{r['primary']['metrics']['cagr']>=.70}。",
          '核心长期动量、原股票池、月内固定单位和已有观察账户不变。没有拟合作者持仓，没有根据后34期重新选择。',
          '股票池和ETF名单仍存在原代理假设；同收盘信号/成交不代表可保证真实执行，另外报告次日开盘及成本敏感性。',
          f"受保护文件：{v['protected_files']}；完整校验结果见VALIDATION.json。"]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
