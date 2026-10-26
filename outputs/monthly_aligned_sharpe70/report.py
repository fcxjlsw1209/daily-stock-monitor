from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda x:json.loads((P/x).read_text())
r,f,v=read('RESULTS.json'),read('FROZEN.json'),read('VALIDATION.json')
lines=['月策略70%目标：收益与波动使用同一形成期的动量核心',
       '2016-08-17至2026-09-24；同完整收盘信号/成交、零成本理想化回放。',
       f"固定14配置；新通过训练门槛{f['new_admitted']}项；全期主要候选达到70%：{v['primary_target70_reached']}。",
       '名称 | 年化 | Sharpe | 最大回撤 | 早53期年化 | 后34期年化 | 次日开盘每边5bp年化']
for name,z in r.items():
    if name=='robust' and z['selection']['spec']==r['primary']['selection']['spec']:continue
    s=z['metrics'];cuts=z['slices'];cost=z['next_open_cost']['metrics']['cagr']
    lines.append(f"{name} | {s['cagr']:.2%} | {s['sharpe']:.3f} | {s['max_drawdown']:.2%} | {cuts['early2016_2020']['cagr']:.2%} | {cuts['later34']['cagr']:.2%} | {cost:.2%}")
lines+=['','原50/52%长期动量核心权重不变；新因子一律用t−294至t−42的253个正价格、252个收益，月内固定复权单位，股票池和原资格不变。',
 '三种核心：算术日收益均值/简单收益标准差、累计收益/简单收益标准差、对数日收益均值/对数收益标准差。波动与收益形成期对齐。前两者是旧公式的匹配复测，不称新发现。原calm核心形成期210日，这里延长为252日；差异不能全归因于波动分母。',
 '比较原65.34评分与真实66.36选择器；核心权重50/52%不变，替换一半或全部核心rank，辅助分不变。',
 '前88期冻结规则，后34期检验；历史已经多次研究，不能称新的样本外结果。',
 '历史Top32和ETF17仍有代理假设；不声称复现原作者。',
 f"独立固定单位回放、原控制校验通过；{v['protected_files']}个受保护文件未改变。",'所有观察账户保持原规则。']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
