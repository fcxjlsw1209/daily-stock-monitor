"""Finite checks declared after stage1. Scores cached, portfolio holdings carried causally."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import adaptation as a
P=Path(__file__).resolve().parent/'robustness'

def main():
 P.mkdir(exist_ok=True)
 resume='--resume' in __import__('sys').argv
 if (P/'SUMMARY.json').exists() and not resume:raise RuntimeError('Use --resume only for unchanged declared cases')
 cases=[]
 for name,kind,windows,limit in [('short_window','walk_gap',(63,126),None),('long_window','walk_gap',(252,504),None),('limit025','entry_limit',None,.25),('limit075','entry_limit',None,.75),('walk_plus_limit','walk_gap',(126,252),.5)]:
  for cost in [10,30]:cases.append(dict(key=f'{name}_cost{cost}',kind=kind,cost=cost,windows=windows,limit=limit,start='2019-01-02'))
 for kind in ['fixed10','walk_gap','entry_limit']:
  cases.append(dict(key=f'{kind}_cost50',kind=kind,cost=50,windows=(126,252) if kind=='walk_gap' else None,limit=None,start='2019-01-02'))
  for cost in [10,30]:cases.append(dict(key=f'fresh2024_{kind}_cost{cost}',kind=kind,cost=cost,windows=(126,252) if kind=='walk_gap' else None,limit=None,start='2024-01-02'))
 config=dict(cases=cases,declared_after_stage1=True,role='Sensitivity checks, not recent-return reselection',performance_publishable=False,
 cost_policy='Reference gap selection stays based on original10bp reference strategy returns in every cost stress case, so policy choices are held fixed.',
 data_policy='All reviewed history remains exploratory; post2024 fresh-cash is not untouchedOOS')
 if resume:assert json.loads((P/'CONFIG.json').read_text())==json.loads(json.dumps(config))
 else:(P/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 protect=json.loads((a.P/'SOURCE_HASHES.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in protect.items())
 codehash=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
 m,d=a.r.prepare();d=a.r.apply(m,d,P/'pool');f,_=a.r.features(d)
 axes=json.loads((a.P/'AXES.json').read_text());assert axes['dates']==[str(x.date()) for x in d['close'].index] and axes['symbols']==list(d['close'].columns)
 with np.load(a.P/'SCORES.npz',allow_pickle=False) as z:score=pd.DataFrame(z['base'],index=d['close'].index,columns=d['close'].columns)
 mask=f['event']&score.ge(.75);refs={}
 for gap,file in [(.1,a.P/'fixed10_cost10.json'),(.15,a.P/'fixed15_cost10.json'),(.2,a.old.P/'robustness/gap20_cost10.json')]:
  rows=json.loads(file.read_text())['daily'];refs[gap]=pd.Series({pd.Timestamp(v['date']):v['return'] for v in rows})
 summary={};gap_audits={}
 for cfg in cases:
  if cfg['windows']:gaps,gap_audits[cfg['key']]=a.monthly_gap(score.index,refs,tuple(cfg['windows']))
  else:gaps,_=a.gap_schedule(score.index,score,cfg['kind'])
  key=cfg['key'];path=P/(key+'.json')
  out=json.loads(path.read_text()) if resume and path.exists() else a.run(cfg['kind'],cfg['cost'],d,f,score,mask,gaps,start=cfg['start'],limit_atr=cfg['limit'])
  if not path.exists():path.write_text(json.dumps(out,allow_nan=False)+'\n')
  summary[key]=a.old.prior.metrics(out);summary[key]['diagnostics']=a.old.diagnostics(out);summary[key]['cancellations']=len(out['cancellations'])
  (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(key,'CAGR',summary[key]['full']['cagr'],'DEV',(summary[key]['development'] or {}).get('cagr'),'LAST',summary[key]['last252']['return_'],flush=True)
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in protect.items());assert codehash==hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
 (P/'GAP_CHOICES.json').write_text(json.dumps(gap_audits,indent=2)+'\n')
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,code_sha256=codehash,data_certified=False),indent=2)+'\n');print('COMPLETE',flush=True)

if __name__=='__main__':main()
