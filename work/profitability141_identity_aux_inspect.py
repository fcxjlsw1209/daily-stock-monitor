from pathlib import Path
import json
R=Path.cwd();plan=json.loads((R/'outputs/monthly_profitability141_sources/DATA_PLAN.json').read_text());symbols={x['symbol'] for x in plan['candidate_cik_locating_evidence'] if x['candidate_ciks']};symbols|={'META','FB','BRK-B'}
for name in ['VERIFIED_IDENTITY_EVENTS.json','IDENTITY_EPISODE_GAPS.json','REGISTRY_IDENTITY_AUDIT.json']:
 p=R/'outputs/dynamic_top100'/name;v=json.loads(p.read_text());print(name, list(v) if isinstance(v,dict) else ['list',len(v)])
 def walk(x):
  if isinstance(x,dict):
   if any(x.get(k) in symbols for k in ['ticker','symbol','research_ticker']):print(json.dumps(x)[:2500]);return
   for y in x.values():walk(y)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(v)
