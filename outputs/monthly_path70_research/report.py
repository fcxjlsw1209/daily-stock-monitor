from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda x:json.loads((P/x).read_text())
r,f,v=read('RESULTS.json'),read('FROZEN.json'),read('VALIDATION.json')
lines=['月策略70%目标：长期涨势的阶段结构',
       '2016-08-17至2026-09-24；同完整收盘信号/成交、零成本理想化回放。',
       f"固定26配置；新通过训练门槛{f['new_admitted']}项；全期主要候选达到70%：{v['primary_target70_reached']}。",
       '名称 | 年化 | Sharpe | 最大回撤 | 早53期年化 | 后34期年化 | 次日开盘每边5bp年化']
for name,z in r.items():
    if name=='robust' and z['selection']['spec']==r['primary']['selection']['spec']:continue
    s=z['metrics'];cuts=z['slices'];cost=z['next_open_cost']['metrics']['cagr']
    lines.append(f"{name} | {s['cagr']:.2%} | {s['sharpe']:.3f} | {s['max_drawdown']:.2%} | {cuts['early2016_2020']['cagr']:.2%} | {cuts['later34']['cagr']:.2%} | {cost:.2%}")
lines+=['','原50/52%长期动量核心、跳过最近42交易日、月内固定复权单位，股票池和资格不变。',
 '四种手工特征：最弱70日阶段、阶段均值减波动、最坏阶段下行风险、中心绝对偏差风险分母。',
 '比较原65.34评分与真实66.36选择器。手工预算最多10%，旧校正槽仍最多10%，合计最多20%，仅替换原辅助分。',
 '前88期冻结规则，后34期检验；历史已经多次研究，不能称新的样本外结果。',
 '历史Top32和ETF17仍有代理假设；不声称复现原作者。',
 f"独立固定单位回放、原控制校验通过；{v['protected_files']}个受保护文件未改变。",'所有观察账户保持原规则。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
