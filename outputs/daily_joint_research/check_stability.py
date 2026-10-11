"""Neighbor-window check after adaptive candidate discovery; keep original candidate."""
from pathlib import Path
import json,functools
import numpy as np
import pandas as pd
import experiment as e
import adaptive as a

P=e.P/'stability'
CONFIG=dict(windows={'shorter':[84,168,30,40],'longer':[168,336,50,80]},
 candidate_kept=[126,252,40,60],cost_bps=[10,30],introduced_after_adaptive_results=True,
 purpose='Sensitivity check, not a new winner selection round',no_live_changes=True)


def skill(pred,outcome,event,window):
 good=event&pred.notna()&outcome.notna();x=pred.where(good).rank(axis=1);y=outcome.where(good).rank(axis=1)
 x=x.sub(x.mean(axis=1),axis=0);y=y.sub(y.mean(axis=1),axis=0)
 den=np.sqrt((x*x).sum(axis=1,min_count=1)*(y*y).sum(axis=1,min_count=1)).replace(0,np.nan)
 ic=((x*y).sum(axis=1,min_count=1)/den).where(good.sum(axis=1)>=5).shift(6)
 short,long,nshort,nlong=window
 return pd.concat([ic.rolling(short,min_periods=nshort).mean(),ic.rolling(long,min_periods=nlong).mean()],axis=1).min(axis=1,skipna=False)


def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite sensitivity results')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 m,d=e.r.prepare();d=e.r.apply(m,d,P/'pool');f,_=e.r.features(d);pred,_,audit=e.r.model(f,6)
 pure=sum(f['groups'])/4;base=a.blended_score(f,pred,.5);scores={};choices={}
 for name,window in CONFIG['windows'].items():
  selected,choices[name]=a.choices(pred.index,{'baseline':skill(pred,f['y'],f['event'],window)},['baseline'])
  sc=pure.copy();yes=selected.eq('baseline');sc.loc[yes]=base.loc[yes];scores[name]=sc
 (P/'CHOICES.json').write_text(json.dumps(choices,indent=2)+'\n');summary={}
 for cost in CONFIG['cost_bps']:
  for name,sc in scores.items():
   ff=dict(score=sc,mask=f['event']&sc.ge(.75),atr=f['atr'],ma5=f['ma5']);e.h.plan=e.prior.daily_plan(.15);e.h.execute=functools.partial(e.prior.relaxed_daily,cost_bps=cost)
   ot=e.h.terminal_baskets.apply;os=e.h.spin_distributions.apply
   try:
    e.h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);e.h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
    out=e.h.run_account('baseline',d,'2019-01-02','2026-10-06',{'baseline':ff})
   finally:e.h.terminal_baskets.apply=ot;e.h.spin_distributions.apply=os
   assert out['status']=='complete';assert all(v['signal_date']<v['session'] for v in out['decisions'])
   key=f'{name}_cost{cost}';(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n');summary[key]=e.prior.metrics(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(key,json.dumps(summary[key]),flush=True)


if __name__=='__main__':main()
