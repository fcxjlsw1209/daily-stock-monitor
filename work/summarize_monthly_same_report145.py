"""Read existing registered experiment outputs; no replay or modifications."""
from pathlib import Path
import json
from collections import Counter
root=Path(__file__).resolve().parents[1]
p=root/'outputs/monthly_same_report145'
def read(n): return json.loads((p/(n+'.json')).read_text())
f=read('FEATURE_AUDIT')
print(json.dumps({'cells':len(f['cells']),'available':dict(Counter(c['symbol'] for c in f['cells'] if c['available']))},sort_keys=True))
if (p/'FROZEN.json').exists():
 print(json.dumps({'selection':read('FROZEN')['selection']}))
for n in ['TRAIN_TRIALS','TRAIN_OPEN_TRIALS','TRAIN_REFERENCE_OPEN']:
 if (p/(n+'.json')).exists(): print(json.dumps({n:read(n)}))
if (p/'RESULTS.json').exists():
 for name,paths in read('RESULTS').items():
  print(json.dumps({'name':name,'paths':{mode:{'metrics':{k:v['metrics'][k] for k in ['cagr','sharpe','max_drawdown','start','end']},'slices':{k:{m:s[m] for m in ['cagr','sharpe','max_drawdown']} for k,s in v['slices'].items()}} for mode,v in paths.items()}}))
 a=read('ATTRIBUTION')
 print(json.dumps({'changes':{name:{'changed_months':v['changed_months'],'changes':[{k:x[k] for k in ['date','old','new','log_contribution']} for x in v['changes']]} for name,v in a['comparisons'].items()}}))
