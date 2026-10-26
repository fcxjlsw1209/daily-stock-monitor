from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
r,f,v=[read(n) for n in ('RESULTS.json','FROZEN.json','VALIDATION.json')]
lines=['历史市场敏感度调整的月收益标签：相同样本双目标对照。', '历史beta仅用各训练信号日及以前252个日对数收益；两臂预测下一月对数收益减去1倍或历史beta倍SPY对数收益，先截断再在共同训练样本内中心化。', '模型只换原有Ridge评分槽：弱市5%，平静且非弱市1%。不添加beta特征，不把组合变成市场中性。', '配对训练样本可能因beta缺失缩小，因此与同样本beta=1模型另作对照；原完整样本模型独立重放核验。查询股票池不因beta或未来标签改变。']
lines += ['2016-08-17至2026-09-24；同完整收盘信号/成交、零成本理想化回放。',
'长期动量保持50%/52%且跳过42日，月末三只等权、月内固定单位。保持原资格和真实边界选择流程。',
'前88期及早/近分段门槛选择，先冻结候选再读后34期；所有固定候选均报告，包括未通过者。',
'名称 | 全期年化 | Sharpe | 最大回撤 | 前88期年化 | 后34期年化 | 次日开盘每边5bp年化']
for name,z in r.items():
 a,c=z['metrics'],z['slices']
 lines.append(f"{name} | {a['cagr']:.2%} | {a['sharpe']:.3f} | {a['max_drawdown']:.2%} | {c['training2016_2023']['cagr']:.2%} | {c['later34']['cagr']:.2%} | {z['next_open_cost']['metrics']['cagr']:.2%}")
lines += ['',f"主要候选：{f['selected']['primary']['spec']}；新通过门槛：{f['new_admitted']}。",
 f"主要候选达到70%：{r['primary']['metrics']['cagr']>=.70}。",
 '所有历史反复用于开发，非未见样本外；估算Top32、ETF假设、行业与历史数据时点、同收盘执行限制保留。不是作者原公式或未来收益保证。',
 f"受保护文件：{v['protected_files']}；观察账户未修改。"]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
