"""Numerical tie audit, declared after tiny reference-score differences were observed."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import adaptation as a
P=Path(__file__).resolve().parent/'stable_ties'

def schedule(index,refs,tolerance=1e-10,windows=(126,252)):
 log=np.log1p(pd.DataFrame(refs).reindex(index));out=pd.Series(.15,index=index);audit=[];last=.15
 for month in index[index>='2019-01-01'].to_period('M').unique():
  ids=np.flatnonzero(index.to_period('M')==month);prior=index[ids[0]-1];past=log.loc[:prior].tail(max(windows))
  values={float(k):min(float(past[k].tail(w).mean()) for w in windows) for k in past if past[k].notna().sum()==max(windows)}
  tied=[]
  if values:
   best=max(values.values());tied=[k for k in values if best-values[k]<=tolerance]
   last=last if last in tied else min(tied,key=lambda k:(abs(k-.15),k))
  else:last=.15
  out.iloc[ids]=last;audit.append(dict(month=str(month),selected_on=str(prior.date()),first_signal=str(index[ids[0]].date()),gap=last,tied_gaps=tied,criterion=values))
 return out,audit


def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite tie audit')
 config=dict(tolerance=1e-10,unit='mean daily log return',tie_rule='Keep previous month gap among ties; otherwise nearest.15 then smaller gap',
  rationale='Observed floating differences around1e-17 in raw monthly chooser; numerical determinism test, not maximizing return',cost_bps=[10,30],starts=['2019-01-02','2024-01-02'],data_certified=False)
 (P/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 protect=json.loads((a.P/'SOURCE_HASHES.json').read_text());m,d=a.r.prepare();d=a.r.apply(m,d,P/'pool');f,_=a.r.features(d)
 axes=json.loads((a.P/'AXES.json').read_text());assert axes['symbols']==list(d['close'].columns);assert axes['dates']==[str(v.date()) for v in d['close'].index]
 with np.load(a.P/'SCORES.npz',allow_pickle=False) as z:score=pd.DataFrame(z['base'],index=d['close'].index,columns=d['close'].columns)
 refs={}
 for gap,file in [(.1,a.P/'fixed10_cost10.json'),(.15,a.P/'fixed15_cost10.json'),(.2,a.old.P/'robustness/gap20_cost10.json')]:
  rows=json.loads(file.read_text())['daily'];refs[gap]=pd.Series({pd.Timestamp(v['date']):v['return'] for v in rows})
 gaps,audit=schedule(score.index,refs);(P/'CHOICES.json').write_text(json.dumps(audit,indent=2)+'\n');summary={}
 for start in config['starts']:
  for cost in config['cost_bps']:
   key=f'{start[:4]}_cost{cost}';out=a.run('walk_gap',cost,d,f,score,f['event']&score.ge(.75),gaps,start=start)
   (P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n');summary[key]=a.old.prior.metrics(out);summary[key]['diagnostics']=a.old.diagnostics(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(key,'CAGR',summary[key]['full']['cagr'],'LAST',summary[key]['last252']['return_'],flush=True)
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in protect.items())
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=4,protected_unchanged=True),indent=2)+'\n')
 print('COMPLETE',flush=True)

if __name__=='__main__':main()
