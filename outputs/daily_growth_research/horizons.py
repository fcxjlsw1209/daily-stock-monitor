"""Align mature rebound outcomes and model horizons with a multiweek holding policy."""
import json
import numpy as np
import pandas as pd
import growth as g
z=g.z;P=g.P/'horizons'
VARIANTS=['control','rules5','pred10','pred20','pred10_small','pred20_small','hist10_rules','hist20_rules','aligned10','aligned20','aligned10_tree','aligned20_tree','ensemble_scores']
CONFIG=dict(declared_after_partial_management_results=True,declared_before_this_batch_results=True,variants=VARIANTS,cost_bps=[10,30],
 holding='Unchanged30day cap,2.5ATR initial stop,no fixed profit,target2/3 positions,prior-close/next-open,original gap and shock gate',
 unchanged='Same pool and entry pullback eligibility; four factor groups25% each. No leverage or live changes.',
 labels='For h10/20: adjusted next-open to h+1 open net return minus.001; never usable before h+1 sessions. All training and skill gates use horizon-specific delay.',
 pred_only='Only Ridge label h10/20 changes; prior original feature groups retained. Model50% of rebound group (small25%) when own mature skill gate passes.',
 aligned='Historical rebound group rebuilt using current original pullback event; path group uses highest high + lowest low over next h sessions relative to next open, shifted h+1 before rolling756day mean with prior20 shrink. Update these two groups in features and fit the corresponding return target.',
 tree='Original simple residual stump,15% residual correction. Same label maturation as Ridge.',
 rules='No learned forecast; historical four group scores only.',
 ensemble='Arithmetic average of original control,pred10,pred20 scores; same four-group aggregate weights, model component diluted when gates are off.',
 training='Original monthly Ridge alpha10,1008session window,min300 observations. No hyperparameter tuning. Skill gate: per-date>=5 events,126/252 averages,min40/60,positive both,checked previous session of month.',
 development_selection=g.CONFIG['development_selection'],retrospective_growth_selection=g.CONFIG['retrospective_growth_selection'],all_history_reused=True,data_certified=False,performance_publishable=False)

def skill(prediction,outcome,event,delay):
 good=event&prediction.notna()&outcome.notna();x=prediction.where(good).rank(axis=1);y=outcome.where(good).rank(axis=1)
 x=x.sub(x.mean(axis=1),axis=0);y=y.sub(y.mean(axis=1),axis=0)
 denominator=np.sqrt((x*x).sum(axis=1,min_count=1)*(y*y).sum(axis=1,min_count=1)).replace(0,np.nan)
 ic=((x*y).sum(axis=1,min_count=1)/denominator).where(good.sum(axis=1)>=5).shift(delay)
 return pd.concat([ic.rolling(126,min_periods=40).mean(),ic.rolling(252,min_periods=60).mean()],axis=1).min(axis=1,skipna=False)

def features(d,f,h,aligned):
 ff=dict(f);ff['groups']=list(f['groups']);ff['X']=list(f['X']);a=d['adj'];c=d['close'];ao=d['opening']*a/c
 ff['y']=ao.shift(-(h+1))/ao.shift(-1)-1-.001
 if aligned:
  rank=lambda x:x.where(f['valid']).rank(axis=1,pct=True)
  hist=z.r.matured_average(ff['y'],f['event'],horizon=h+1)
  ah=d['high']*a/c;al=d['low']*a/c
  hi=pd.concat([ah.shift(-i) for i in range(1,h+1)],keys=range(h)).groupby(level=1).max().reindex(a.index)
  lo=pd.concat([al.shift(-i) for i in range(1,h+1)],keys=range(h)).groupby(level=1).min().reindex(a.index)
  path=z.r.matured_average(hi/ao.shift(-1)-1+lo/ao.shift(-1)-1,f['event'],horizon=h+1)
  ff['groups'][1]=rank(hist);ff['groups'][2]=.7*rank(path)+.3*rank(a.pct_change(fill_method=None).rolling(20).std())
  ff['X'][1]=ff['groups'][1];ff['X'][2]=ff['groups'][2]
 return ff

def score(ff,pred,h,weight):
 active,audit=z.t.a.old.adaptive.choices(pred.index,{'model':skill(pred,ff['y'],ff['event'],h+1)},['model'])
 sc=sum(ff['groups'])/4;blended=z.t.a.old.adaptive.blended_score(ff,pred,weight)
 sc.loc[active.eq('model')]=blended.loc[active.eq('model')]
 return sc,audit

def prepare_scores(d,f,base):
 scores={'control':base,'rules5':sum(f['groups'])/4};audits={};choices={};preds={}
 for h in [10,20]:
  for aligned in [False,True]:
   ff=features(d,f,h,aligned);pred,tree,training=z.r.model(ff,h+1);name=('aligned' if aligned else 'pred')+str(h)
   assert all(v['latest_label_session']<v['first_prediction'] for v in training)
   audits[name]=training;preds[name]=pred
   scores[name],choices[name]=score(ff,pred,h,.5)
   if aligned:
    scores[f'hist{h}_rules']=sum(ff['groups'])/4;scores[name+'_tree'],choices[name+'_tree']=score(ff,tree,h,.5)
   else:scores[name+'_small'],choices[name+'_small']=score(ff,pred,h,.25)
 scores['ensemble_scores']=(scores['control']+scores['pred10']+scores['pred20'])/3
 return scores,audits,choices,preds

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite horizons')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=g.protected_hashes()
 d,f,base,gaps,x,ctx,_=g.prepare(P);scores,audits,choices,pred=prepare_scores(d,f,base)
 for file,value in [('TRAINING_AUDIT',audits),('MODEL_CHOICES',choices)]: (P/(file+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 np.savez_compressed(P/'SCORES.npz',**{k:v.to_numpy() for k,v in scores.items()});np.savez_compressed(P/'PREDICTIONS.npz',**{k:v.to_numpy() for k,v in pred.items()})
 (P/'AXES.json').write_text(json.dumps(dict(dates=[str(v.date()) for v in base.index],symbols=list(base.columns)))+'\n')
 summary={};print('HORIZONS_PREPARED',flush=True)
 for name in VARIANTS:
  for cost in [10,30]:
   out=g.run(g.BASE,cost,d,f,scores[name],gaps,x,ctx)
   if name=='control':
    ref=json.loads((z.P/'robustness'/f'stop25_cost{cost}.json').read_text());assert out['orders']==ref['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in ref['daily']],rtol=1e-12,atol=1e-7)
   z.save(P,f'{name}_cost{cost}',out,summary)
 b10=summary['control_cost10'];b30=summary['control_cost30']
 dev=[k for k in VARIANTS if k!='control' and summary[k+'_cost30']['development']['max_drawdown']>=-.35 and summary[k+'_cost10']['development']['cagr']>=b10['development']['cagr'] and summary[k+'_cost30']['development']['cagr']>=b30['development']['cagr']]
 growth=[k for k in VARIANTS if k!='control' and summary[k+'_cost10']['full']['max_drawdown']>=-.35 and summary[k+'_cost30']['full']['cagr']>b30['full']['cagr'] and summary[k+'_cost30']['ex_recent']['cagr']>=b30['ex_recent']['cagr']]
 selection=dict(development_eligible=dev,development_selected=max(dev,key=lambda k:summary[k+'_cost30']['development']['cagr']) if dev else None,retrospective_eligible=growth,retrospective_growth_selected=max(growth,key=lambda k:summary[k+'_cost10']['full']['cagr']) if growth else None)
 (P/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n');assert protected==g.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,control_reproduced=True,horizon_labels_mature=True),indent=2)+'\n')
 print('HORIZONS_COMPLETE',len(summary),selection,flush=True)
if __name__=='__main__':main()
