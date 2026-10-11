"""Declared second batch: remove inherited sector cap; test 6/9/12-month skip2."""
import json,functools
import pandas as pd,numpy as np
import research as r
P=r.P

def relaxed_daily(name,state,decision,bars,dates,cost_bps=10,sector=None):
 return r.ORIGINAL_EXEC(name,state,decision,bars,dates,cost_bps=cost_bps,sector=None)
def relaxed_monthly(name,state,decision,bars,dates,cost_bps=10,sector=None):
 return r.execute_equal_close(name,state,decision,bars,dates,cost_bps=cost_bps,sector=None)
def main():
 dc=[dict(name=f'x_d_{mode}_{int(th*100)}_gate{gate}',mode=mode,threshold=th,cap=10,tp=3,sl=2,cash_exit=False,sector_cap=3,market_gate=gate) for mode in ['ridge','ridge_tree'] for th in [.65,.75] for gate in [False,True]]
 mc=[dict(name=f'x_m_{window}_{mode}_gate{gate}',window=window,mode=mode,market_gate=gate) for window in [126,189,252] for mode in ['raw','risk'] for gate in [False,True]]
 (P/'EXTENSION_CONFIG.json').write_text(json.dumps(dict(daily=dc,monthly=mc,reason='Reference has no sector cap; medium-term window unspecified; all trials disclosed',selection='same development-only criterion; no recent-return winner promotion'),indent=2)+'\n')
 m,d=r.prepare();d=r.apply(m,d,P/'pool');df,mf=r.features(d);dp,dt,_=r.model(df,6);mp,_,_=r.model(mf,22);summary=json.loads((P/'SUMMARY.json').read_text());cache={}
 def save(name,result):
  summary[name]=r.summarise(result);(P/(name+'.json')).write_text(json.dumps(result,allow_nan=False)+'\n');(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(name,json.dumps(summary[name]),flush=True)
 for cfg in dc:
  g=df['groups'];pred=dt if cfg['mode']=='ridge_tree' else dp;sc=(g[0]+.5*g[1]+.5*pred.where(df['valid']).rank(axis=1,pct=True)+g[2]+g[3])/4;mask=df['event']&(sc>=cfg['threshold'])
  if cfg['market_gate']:mask=mask.mul(df['market'],axis=0)
  ff=dict(score=sc,mask=mask,atr=df['atr'],ma5=df['ma5']);r.h.plan=r.daily_plan(cfg);r.h.execute=relaxed_daily;cache[cfg['name']]=(d,ff,r.h.plan,relaxed_daily);save(cfg['name'],r.h.run_account('baseline',d,'2019-01-02',m['end'],{'baseline':ff}))
 dates=d['close'].index;monthends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);md=dict(d);md['opening']=d['close'];a=d['adj'];vol=a.pct_change(fill_method=None).rolling(63).std();rank=lambda x:x.where(mf['valid']).rank(axis=1,pct=True)
 for cfg in mc:
  momentum=a.shift(42)/a.shift(cfg['window'])-1;core=rank(momentum/vol if cfg['mode']=='risk' else momentum)
  sc=.9*(.7*core+.15*mf['X'][2]+.10*mf['X'][3]+.05*mf['X'][7])+.1*rank(mp)
  mask=mf['valid']&momentum.gt(0)
  if cfg['market_gate']:mask=mask.mul(mf['market'],axis=0)
  ff=dict(score=sc,mask=mask,atr=mf['atr'],ma5=mf['ma5']);r.h.plan=r.monthly_plan(monthends,3);r.h.execute=relaxed_monthly;cache[cfg['name']]=(md,ff,r.h.plan,relaxed_monthly);save(cfg['name'],r.h.run_account('baseline',md,'2019-01-02',m['end'],{'baseline':ff}))
 results={k:json.loads((P/(k+'.json')).read_text()) for k,v in summary.items() if v['status']=='complete'};ds=[k for k in results if k.startswith(('d_','x_d_'))];ms=[k for k in results if k.startswith(('m_','x_m_'))];combinations={}
 for dn in ds:
  for mn in ms:
   for w in [.25,.5,.75]:
    rows=r.combo_daily(results[dn]['daily'],results[mn]['daily'],w);x=pd.DataFrame(rows);key=f'{dn}__{mn}__daily{w}';combinations[key]=dict(daily=dn,monthly=mn,daily_weight=w,development=r.period_metrics(rows,'2019-01-02','2023-12-31'),evaluation=r.period_metrics(rows,'2024-01-01',m['end']),full=r.period_metrics(rows,'2019-01-02',m['end']),last252_return=float(x.equity.iloc[-1]/x.equity.iloc[-253]-1),year2026=r.period_metrics(rows,'2026-01-01',m['end']))
 (P/'COMBINATIONS_EXTENDED.json').write_text(json.dumps(combinations,indent=2)+'\n');qualified={k:v for k,v in combinations.items() if v['development']['max_drawdown']>=-.4};selected=max(qualified,key=lambda k:qualified[k]['development']['cagr']);pick=dict(selected=selected,combination=qualified[selected],retrospective_best_recent=max(combinations,key=lambda k:combinations[k]['last252_return']),target_met_count=sum(v['last252_return']>=1 for v in combinations.values()),combinations_tested=len(combinations),standalone_target_met=[k for k,v in summary.items() if v['status']=='complete' and v['last252_return']>=1]);(P/'SELECTION_EXTENDED.json').write_text(json.dumps(pick,indent=2)+'\n');print('DONE',json.dumps(pick),flush=True)
 # Store selection-specific execution recipe for subsequent verification, not live use.
 (P/'RESEARCH_COMPLETE.json').write_text(json.dumps(dict(complete=all(v['status']=='complete' for v in summary.values()),standalone_count=len(summary),combination_count=len(combinations),original_accounts_changed=False),indent=2)+'\n')
if __name__=='__main__':main()
