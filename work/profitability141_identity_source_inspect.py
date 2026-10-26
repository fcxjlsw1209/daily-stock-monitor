from pathlib import Path
import json
R=Path.cwd()
for s in ['work/monthly_profitability141_local_annual_feasibility.json','outputs/monthly_earnings_repair/berkshire/DISCOVERY.json','outputs/monthly_earnings_repair/berkshire/manifest.json','outputs/monthly_earnings_repair/batch_health_energy_foreign/GE_REPORTED_PERIOD_POLICY.json']:
 p=R/s
 if not p.exists():continue
 d=json.loads(p.read_text());print('FILE',s,'KEYS',list(d) if isinstance(d,dict) else ['list',len(d)])
 if s.endswith('GE_REPORTED_PERIOD_POLICY.json'):print(json.dumps(d)[:2000])
 elif isinstance(d,list): print('FIRST',json.dumps(d[0])[:1800] if d else '')
 else:
  for k,v in d.items():
   if isinstance(v,list):print(k,len(v),json.dumps(v[0])[:1800] if v else '')
