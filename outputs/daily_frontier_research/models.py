"""Third disclosed batch: target construction and learned fraction of rebound group."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import frontier as z
P=z.P/'models'
VARIANTS={'original50':('original',.5),'original75':('original',.75),'original100':('original',1.),
 'tree50':('tree',.5),'tree75':('tree',.75),'rank50':('rank',.5),'rank75':('rank',.75),
 'clipped50':('clipped',.5),'clipped75':('clipped',.75),'positive50':('positive',.5),'positive75':('positive',.75),
 'rank_ensemble50':('ensemble',.5)}
CONFIG=dict(declared_after_96_runs=True,declared_before_this_batch_results=True,variants=VARIANTS,cost_bps=[10,30],
 all_history_reused=True,group_weights=[.25]*4,no_leverage=True,no_observer_promotion=True,
 data_certified=False,performance_publishable=False,
 base='Same shock pause2 entry gate and stable monthly replacement; no parameter changes from prior control.',
 targets={'original':'Net5session forward open/open return; existing frozen model recipe.',
 'tree':'Original return Ridge plus15% one best depth1 residual stump; original r.model implementation.',
 'rank':'Within-signal-session percentile rank of realized net5session return among original pullback event candidates minus.5; need>=3 observed candidates, excluded until all labels mature.',
 'clipped':'Net5session return clipped to [-.20,.20], fixed bounds not fitted to recent results.',
 'positive':'Indicator of positive net5session return; Ridge ranking score, NOT a calibrated probability.',
 'ensemble':'Average within-valid-universe ranks of original, rank and clipped predictions; needs all3.'},
 validation='Each model separately:126/252 matured per-date Spearman with actual5session returns, positive both, min40/60 observations; choose use/off from prior close at first signal session of month.',
 weights='Model occupies50/75/100% of rebound group ONLY when its own historical skill gate passes; off months use historical rebound group. Overall four groups remain25%.',
 training='Monthly fit, last1008sessions, at least300 finite event observations, Ridge alpha10, embargo6sessions. Whole date label ranking still fully mature before training.',
 target_policy=z.CONFIG['target_policy'],development_selection=z.CONFIG['development_selection'])

def prepare_models(f):
 models={};audits={}
 original,tree,training=z.r.model(f,6);models['original']=original;models['tree']=tree;audits['original']=training
 for name in ['rank','clipped','positive']:
  ff=dict(f)
  if name=='rank':
   target=f['y'].where(f['event']);target=target.rank(axis=1,pct=True)-.5;target=target.where((f['event']&f['y'].notna()).sum(1).ge(3),axis=0)
  elif name=='clipped':target=f['y'].clip(-.20,.20)
  else:target=f['y'].gt(0).astype(float).where(f['y'].notna())
  ff['y']=target;models[name],_,audits[name]=z.r.model(ff,6)
 models['ensemble']=sum(models[k].where(f['valid']).rank(axis=1,pct=True) for k in ['original','rank','clipped'])/3
 scores={};choices={}
 for name,pred in models.items():
  chosen,choice=z.t.a.old.adaptive.choices(pred.index,{name:z.t.a.old.adaptive.matured_skill(pred,f['y'],f['event'])},[name])
  choices[name]=choice
  for key,(source,weight) in VARIANTS.items():
   if source!=name:continue
   pure=sum(f['groups'])/4;blended=z.t.a.old.adaptive.blended_score(f,pred,weight)
   pure.loc[chosen.eq(name)]=blended.loc[chosen.eq(name)];scores[key]=pure
 return models,scores,audits,choices

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite model batch')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=z.protected_hashes()
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 models,scores,audits,choices=prepare_models(f)
 np.testing.assert_allclose(base.to_numpy(),scores['original50'].to_numpy(),equal_nan=True,rtol=1e-12,atol=1e-12)
 assert all(row['latest_label_session']<row['first_prediction'] for entries in audits.values() for row in entries)
 for file,x in [('TRAINING_AUDIT',audits),('MODEL_CHOICES',choices)]: (P/(file+'.json')).write_text(json.dumps(x,indent=2)+'\n')
 np.savez_compressed(P/'SCORES.npz',**{k:v.to_numpy() for k,v in scores.items()})
 np.savez_compressed(P/'PREDICTIONS.npz',**{k:v.to_numpy() for k,v in models.items()})
 (P/'AXES.json').write_text(json.dumps(dict(dates=[str(v.date()) for v in base.index],symbols=list(base.columns)))+'\n')
 x=z.extras(d,f);summary={};print('MODELS_PREPARED',flush=True)
 for name in VARIANTS:
  for cost in CONFIG['cost_bps']:
   out=z.run(z.BASE,cost,d,f,scores[name],gaps,x)
   if name=='original50':
    prior=json.loads((z.P/f'shock_control_cost{cost}.json').read_text());assert out['orders']==prior['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in prior['daily']],rtol=1e-12,atol=1e-7)
   z.save(P,f'{name}_cost{cost}',out,summary)
 original=json.loads((z.P/'SUMMARY.json').read_text());eligible=[k for k in VARIANTS if k!='original50' and summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=original['shock_control_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 hits=[k for k,v in summary.items() if v['last252']['return_']>=1]
 (P/'SELECTION.json').write_text(json.dumps(dict(development_selected=selected,eligible=eligible,in_sample_target_hits=hits,criterion=CONFIG['development_selection']),indent=2)+'\n')
 assert protected==z.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,control_scores_reproduced=True,control_orders_reproduced=True,labels_mature=True,no_leverage=True),indent=2)+'\n')
 print('MODELS_COMPLETE',len(summary),'selected',selected,'hits',hits,flush=True)

if __name__=='__main__':main()
