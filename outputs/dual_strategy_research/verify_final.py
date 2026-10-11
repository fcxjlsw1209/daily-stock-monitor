"""Cost sensitivity of disclosed selected and retrospective candidates."""
import json,functools
import pandas as pd
import research as r
import extend as e
P=r.P

def main():
 m,d=r.prepare();d=r.apply(m,d,P/'pool');df,mf=r.features(d);dp,_,da=r.model(df,6);mp,_,ma=r.model(mf,22)
 assert all(x['latest_label_session']<x['first_prediction'] for x in da+ma)
 dates=d['close'].index;ends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);md=dict(d);md['opening']=d['close'];g=df['groups'];sc=(g[0]+.5*g[1]+.5*dp.where(df['valid']).rank(axis=1,pct=True)+g[2]+g[3])/4
 picks=json.loads((P/'SELECTION_EXTENDED.json').read_text());combos=json.loads((P/'COMBINATIONS_EXTENDED.json').read_text());names={v for k in [picks['selected'],picks['retrospective_best_recent']] for v in [combos[k]['daily'],combos[k]['monthly']]};results={}
 configs=json.loads((P/'CONFIG.json').read_text())['daily']+json.loads((P/'EXTENSION_CONFIG.json').read_text())['daily']
 for name in sorted(names):
  if name.startswith(('d_','x_d_')):
   cfg=next(c for c in configs if c['name']==name);mask=df['event']&(sc>=cfg['threshold'])
   if cfg['market_gate']:mask=mask.mul(df['market'],axis=0)
   ff=dict(score=sc,mask=mask,atr=df['atr'],ma5=df['ma5']);data=d;r.h.plan=r.daily_plan(cfg);ex=e.relaxed_daily if name.startswith('x_') else r.ORIGINAL_EXEC
  else:
   score=mf['base'];mask=mf['valid']&mf['positive'];sector=2
   if name.startswith('x_'):
    a=d['adj'];momentum=a.shift(42)/a.shift(252)-1;vol=a.pct_change(fill_method=None).rolling(63).std();rank=lambda x:x.where(mf['valid']).rank(axis=1,pct=True)
    score=.9*(.7*rank(momentum/vol)+.15*mf['X'][2]+.10*mf['X'][3]+.05*mf['X'][7])+.1*rank(mp);mask=mf['valid']&momentum.gt(0);sector=3
   ff=dict(score=score,mask=mask,atr=mf['atr'],ma5=mf['ma5']);data=md;r.h.plan=r.monthly_plan(ends,sector);ex=e.relaxed_monthly if sector==3 else r.execute_equal_close
  r.h.execute=functools.partial(ex,cost_bps=30);ot=r.h.terminal_baskets.apply;os=r.h.spin_distributions.apply
  try:
   r.h.terminal_baskets.apply=functools.partial(ot,cost_bps=30);r.h.spin_distributions.apply=functools.partial(os,cost_bps=30)
   result=r.h.run_account('baseline',data,'2019-01-02',m['end'],{'baseline':ff})
  finally:r.h.terminal_baskets.apply=ot;r.h.spin_distributions.apply=os
  assert result['status']=='complete',name
  results[name]=result;(P/(name+'_cost30.json')).write_text(json.dumps(result,allow_nan=False)+'\n');print(name,json.dumps(r.summarise(result)),flush=True)
 out={}
 for label,key in [('development_selected',picks['selected']),('retrospective_recent_best',picks['retrospective_best_recent'])]:
  c=combos[key];rows=r.combo_daily(results[c['daily']]['daily'],results[c['monthly']]['daily'],c['daily_weight']);x=pd.DataFrame(rows);out[label]=dict(name=key,full=r.period_metrics(rows,'2019-01-02',m['end']),last252_return=float(x.equity.iloc[-1]/x.equity.iloc[-253]-1))
 out['standalone']={k:r.summarise(v) for k,v in results.items()};out['training_label_checks']=len(da)+len(ma)
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print('DONE',flush=True)
if __name__=='__main__':main()
