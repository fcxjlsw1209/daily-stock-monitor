from pathlib import Path
import json
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
r,f,v=[read(n) for n in ('RESULTS.json','FROZEN.json','VALIDATION.json')]
lines=['Matched training anchor density experiment.',
'Both arms use original plain10 and matured 21-official-session log excess. Month21 uses original monthly anchors; union21 adds weekly anchors from latest known pools. Historical pool proxies remain unverified PIT estimates.',
'Union arm is restricted to the same annual mature calendar months as monthly arm. Each month has the same age weight and both annual sums equal mature A row count, fixing effective alpha10.',
'Both candidates replace existing 5% weak/1% calm-nonweak ML slot. Longmomentum remains 50%/52% with skip42, monthly three equal positions, fixed units within month.',
'2016-08-17..2026-09-24; same final close signal/fill and zero cost idealization. Next-open and costs reported separately.',
'First88 guard versus local67.666 control, both diagnostics frozen before later34; candidates start from scope67.4929 without local-model stacking.',
'A vs original also changes label horizon; B vs A is the matched anchor-density comparison. Overlapping 21-session labels are not independent.',
'Name | Full CAGR | Sharpe | MaxDD | First88 CAGR | Later34 CAGR | Nextopen 5bp/side CAGR']
for name,z in r.items():
 a,c=z['metrics'],z['slices']
 lines.append(f"{name} | {a['cagr']:.2%} | {a['sharpe']:.3f} | {a['max_drawdown']:.2%} | {c['training2016_2023']['cagr']:.2%} | {c['later34']['cagr']:.2%} | {z['next_open_cost']['metrics']['cagr']:.2%}")
lines += ['',f"主要候选：{f['selected']['primary']['spec']}；新通过门槛：{f['new_admitted']}。",
 f"主要候选达到70%：{r['primary']['metrics']['cagr']>=.70}。",
 '所有历史反复用于开发，非未见样本外；估算Top32、ETF假设、历史数据时点、同收盘执行限制保留。不是作者原公式或未来收益保证。',
 f"受保护文件：{v['protected_files']}；观察账户未修改。"]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
