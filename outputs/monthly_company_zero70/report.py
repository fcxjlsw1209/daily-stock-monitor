from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda x:json.loads((P/x).read_text())
r,f,v,a=read('RESULTS.json'),read('FROZEN.json'),read('VALIDATION.json'),read('FEATURE_AUDIT.json')
lines=['月策略70%目标：同公司月度历史收益向零收缩',
 '2016-08-17至2026-09-24；同完整收盘信号/成交、零成本理想化回放。',
 f"六个固定配置；新候选通过训练门槛{f['new_admitted']}项；主要候选达到70%：{v['primary_target70_reached']}。",
 '名称 | 年化 | Sharpe | 最大回撤 | 早53期年化 | 后34期年化 | 次日开盘每边5bp年化']
for name,z in r.items():
 met=z['metrics'];cuts=z['slices'];cost=z['next_open_cost']['metrics']['cagr']
 lines.append(f"{name} | {met['cagr']:.2%} | {met['sharpe']:.3f} | {met['max_drawdown']:.2%} | {cuts['early2016_2020']['cagr']:.2%} | {cuts['later34']['cagr']:.2%} | {cost:.2%}")
lines+=['','原pre-score前两名锁定；只在原排名3至5且距第三名不超过.02的候选之间调整。原辅助预算不扩大。',
 '同公司既有月度超额SPY收益，按过去5年/24个月半衰期加权，以6或12有效月的零先验收缩；分别测试普通历史和相同回撤类型/形态/市场状态历史。变换固定为额外effect=.25*tanh(raw/.05)，缺失保持旧effect。',
 '最多只改变总分.0125；原clip可能抑制部分改变。未知与真实零分开记录，不使用同公司全历史未来结果。',
 '前88期按预先固定门槛冻结，后34期检验；没有共享Ridge先验，也不引入新的事件资料。',
 '历史已经反复研究，不是独立样本外；同收盘成交和历史股票池代理假设仍存在。',
 f"原控制及独立固定单位回放通过，{v['protected_files']}个受保护文件未改变。",'',json.dumps(dict(available_cells=a['available_cells'],effect_audit=a['effect_audit']),ensure_ascii=False,indent=2)]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
