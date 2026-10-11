"""Post-diagnostic experiment: shrink or validate ML before giving it a score weight."""
from pathlib import Path
import json,functools,hashlib
import numpy as np
import pandas as pd
import experiment as e

P=Path(__file__).resolve().parent/'adaptive'
KINDS=['baseline','cost_margin','rules_only','half_ml','validated_base','validated_pool']
CONFIG=dict(variants=KINDS,cost_bps=[10,30],start='2019-01-02',end='2026-10-06',
 declared_before_adaptive_results=True,added_after_initial_calibration=True,
 no_grid=True,group_weights=[.25]*4,selection_rule='Every first signal session of month use prior completed session validation; per-date Spearman correlation among>=5 event candidates, matured by6sessions; positive mean both126(min40observations) and252(min60); pool chooses largest min(two means); otherwise use pure historical four groups. This is a heuristic, not a significance claim.',
 half_ml='Model prediction25% of rebound group, i.e.6.25% total score; baseline12.5%',
 rules_only='No fitted return prediction; original four groups25% each',
 data_certified=False,all_history_previously_inspected=True,no_live_changes=True)


def matured_skill(prediction,outcome,event):
 good=event&prediction.notna()&outcome.notna()
 x=prediction.where(good).rank(axis=1);y=outcome.where(good).rank(axis=1)
 x=x.sub(x.mean(axis=1),axis=0);y=y.sub(y.mean(axis=1),axis=0)
 denominator=np.sqrt((x*x).sum(axis=1,min_count=1)*(y*y).sum(axis=1,min_count=1)).replace(0,np.nan)
 ic=((x*y).sum(axis=1,min_count=1)/denominator).where(good.sum(axis=1)>=5).shift(6)
 a=ic.rolling(126,min_periods=40).mean();b=ic.rolling(252,min_periods=60).mean()
 return pd.concat([a,b],axis=1).min(axis=1,skipna=False)


def choices(index,skills,allowed):
 chosen=pd.Series(None,index=index,dtype=object);audit=[]
 for month in index[index>='2019-01-01'].to_period('M').unique():
  loc=np.flatnonzero(index.to_period('M')==month);prior=index[loc[0]-1]
  values={k:float(skills[k].loc[prior]) for k in allowed if pd.notna(skills[k].loc[prior]) and skills[k].loc[prior]>0}
  selected=max(values,key=values.get) if values else None
  chosen.iloc[loc]=selected
  audit.append(dict(month=str(month),selected_on=str(prior.date()),first_signal=str(index[loc[0]].date()),selected=selected,validation=values))
 return chosen,audit


def blended_score(f,prediction,weight):
 g=f['groups'];rank=prediction.where(f['valid']).rank(axis=1,pct=True)
 return (g[0]+(1-weight)*g[1]+weight*rank+g[2]+g[3])/4


def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite adaptive experiment')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 m,d=e.r.prepare();d=e.r.apply(m,d,P/'pool');f,_=e.r.features(d);base,_,training=e.r.model(f,6)
 axes=json.loads((e.P/'PREDICTION_AXES.json').read_text())
 assert list(f['valid'].columns)==axes['symbols'] and [str(x.date()) for x in f['valid'].index]==axes['dates']
 predictions={'baseline':base}
 for name in ['context','split','context_split']:
  with np.load(e.P/(name+'_predictions.npz'),allow_pickle=False) as saved:
   predictions[name]=pd.DataFrame(saved['mu'],index=base.index,columns=base.columns)
 skill={name:matured_skill(pred,f['y'],f['event']) for name,pred in predictions.items()}
 pure=sum(f['groups'])/4;scores={'baseline':blended_score(f,base,.5),'cost_margin':blended_score(f,base,.5),
  'rules_only':pure,'half_ml':blended_score(f,base,.25)}
 audits={}
 for kind,allowed in [('validated_base',['baseline']),('validated_pool',list(predictions))]:
  selected,audits[kind]=choices(base.index,skill,allowed);sc=pure.copy()
  for name,pred in predictions.items():
   chosen=selected.eq(name);sc.loc[chosen]=blended_score(f,pred,.5).loc[chosen]
  scores[kind]=sc
 (P/'CHOICES.json').write_text(json.dumps(audits,indent=2)+'\n')
 summary={}
 for cost in CONFIG['cost_bps']:
  for kind in KINDS:
   sc=scores[kind];mask=f['event']&sc.ge(.75)
   if kind=='cost_margin':mask &= base.gt(.003)
   ff=dict(score=sc,mask=mask,atr=f['atr'],ma5=f['ma5']);e.h.plan=e.prior.daily_plan(.15);e.h.execute=functools.partial(e.prior.relaxed_daily,cost_bps=cost)
   ot=e.h.terminal_baskets.apply;os=e.h.spin_distributions.apply
   try:
    e.h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);e.h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
    out=e.h.run_account('baseline',d,CONFIG['start'],CONFIG['end'],{'baseline':ff})
   finally:e.h.terminal_baskets.apply=ot;e.h.spin_distributions.apply=os
   assert out['status']=='complete',out.get('error')
   assert all(v['signal_date']<v['session'] for v in out['decisions']);assert all(v['cash']>=-1e-6 for v in out['daily'])
   if kind in ['baseline','cost_margin']:
    old=json.loads((e.P/f'{kind}_cost{cost}.json').read_text());assert old['orders']==out['orders']
    np.testing.assert_allclose([v['equity'] for v in old['daily']],[v['equity'] for v in out['daily']],rtol=1e-12,atol=1e-7)
   key=f'{kind}_cost{cost}';(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
   summary[key]=e.prior.metrics(out);summary[key]['diagnostics']=e.diagnostic(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(key,json.dumps(summary[key]),flush=True)
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,controls_reproduced=True,decisions_before_fill=True,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_certified=False),indent=2)+'\n')


if __name__=='__main__':main()
