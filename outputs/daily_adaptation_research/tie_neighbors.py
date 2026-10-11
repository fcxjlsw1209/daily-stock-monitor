from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import adaptation as a
from stable_ties import schedule
P=Path(__file__).resolve().parent/'tie_neighbors'

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite neighbors')
 (P/'CONFIG.json').write_text(json.dumps(dict(windows=[[63,126],[252,504]],cost_bps=[10,30],tie_tolerance=1e-10,declared_after_tie_result=True,selection='No selection; retain126/252 window',performance_publishable=False),indent=2)+'\n')
 protect=json.loads((a.P/'SOURCE_HASHES.json').read_text());m,d=a.r.prepare();d=a.r.apply(m,d,P/'pool');f,_=a.r.features(d)
 axes=json.loads((a.P/'AXES.json').read_text());assert axes['symbols']==list(d['close'].columns);assert axes['dates']==[str(v.date()) for v in d['close'].index]
 with np.load(a.P/'SCORES.npz',allow_pickle=False) as z:score=pd.DataFrame(z['base'],index=d['close'].index,columns=d['close'].columns)
 refs={}
 for gap,file in [(.1,a.P/'fixed10_cost10.json'),(.15,a.P/'fixed15_cost10.json'),(.2,a.old.P/'robustness/gap20_cost10.json')]:
  refs[gap]=pd.Series({pd.Timestamp(v['date']):v['return'] for v in json.loads(file.read_text())['daily']})
 summary={}
 for windows in [(63,126),(252,504)]:
  gaps,audit=schedule(score.index,refs,windows=windows)
  (P/f'choices_{windows[0]}_{windows[1]}.json').write_text(json.dumps(audit,indent=2)+'\n')
  for cost in [10,30]:
   key=f'w{windows[0]}_{windows[1]}_cost{cost}';out=a.run('walk_gap',cost,d,f,score,f['event']&score.ge(.75),gaps)
   (P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n');summary[key]=a.old.prior.metrics(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(key,summary[key]['full'],summary[key]['last252'],flush=True)
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in protect.items())
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=4,protected_unchanged=True),indent=2)+'\n');print('COMPLETE',flush=True)

if __name__=='__main__':main()
