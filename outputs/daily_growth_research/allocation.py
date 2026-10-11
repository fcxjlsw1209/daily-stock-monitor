"""Capital deployment ablations: original signals and exits, explicit net sizing."""
import json,functools
import numpy as np,pandas as pd
import growth as g
z=g.z;P=g.P/'allocation'
VARIANTS={'control':{},'net_change':dict(schedule='change'),'net_weekly':dict(schedule='weekly'),'net_monthly':dict(schedule='monthly'),'net_band10':dict(schedule='band'),
 'net_momentum':dict(schedule='change',weight='momentum'),'net_inversevol':dict(schedule='change',weight='inversevol'),'net_cash90':dict(schedule='change',exposure=.9)}
CONFIG=dict(declared_after_80_completed_runs=True,declared_before_this_batch_results=True,variants=VARIANTS,cost_bps=[10,30],
 hypothesis='More complete capital deployment may increase compound return but may trim winners, add trading cost or increase drawdown; test all outcomes.',
 scope='Unchanged30day cap,2.5ATR stop,original scoring and entries; only equal target weights on membership-change days, or extra Monday/first-session-of-month/10percentagepoint drift events.',
 weights='Default equal after costs. Momentum tilt=50%equal+50%normalized current momentumgroup. Inversevol uses63completed return std. Both cap any name50%. No leveraged weights.',
 execution='Reuse existing cost-aware net executor; actual open sizing from frozen weights. Partial cost basis and dividends proportional; original entry clock retained for topups; average entryprice and ATR update. This changes the future stop basis, disclosed.',
 band='Trigger when any prior-close holding weight differs from equal target by>10percentagepoints, or pool membership changes.',
 no_position_extension=True,no_added_tickers=True,data_certified=False,performance_publishable=False,all_history_reused=True,
 development_selection=g.CONFIG['development_selection'],retrospective_growth_selection=g.CONFIG['retrospective_growth_selection'])

def weights_for(targets,kind,mom,ret):
 if not targets:return {}
 n=len(targets)
 if kind=='momentum':
  signal=mom.reindex(targets).clip(lower=0).fillna(0);total=signal.sum();w=pd.Series(1/n,index=targets) if total<=0 else .5/n+.5*signal/total
 elif kind=='inversevol':
  sigma=ret[targets].tail(63).std();v=1/sigma.clip(lower=.001);w=v/v.sum() if v.notna().all() else pd.Series(1/n,index=targets)
 else:w=pd.Series(1/n,index=targets)
 if w.max()>.5:
  top=w.idxmax();rest=w.drop(top);w.loc[rest.index]=.5*rest/rest.sum();w.loc[top]=.5
 assert abs(w.sum()-1)<1e-10 and (w<=.5+1e-10).all()
 return {s:float(w[s]) for s in targets}

def decorate(dec,state,day,dates,close,atr,score,momentum,ctx,cfg):
 prev=dates[dates.get_loc(day)-1];sold={v['symbol'] for v in dec['sells']};targets=sorted((set(state['positions'])-sold)|{v['symbol'] for v in dec['buys']})
 changed=bool(dec['sells'] or dec['buys']);mode=cfg['schedule'];due=changed
 if mode=='weekly':due|=day.weekday()==0
 if mode=='monthly':due|=day.to_period('M')!=prev.to_period('M')
 if mode=='band' and len(targets)>=2:
  due|=any(abs(state['positions'][s]['qty']*close.at[prev,s]*ctx['raw_per_signal'].at[prev,s]/state['equity']-1/len(targets))>.1 for s in state['positions'] if s in targets)
 if not due:return dec
 if len(targets)<2:targets=[]
 w=weights_for(targets,cfg.get('weight','equal'),momentum.loc[prev],ctx['ret'].loc[:prev]);exposure=cfg.get('exposure',1.)
 dec['target_weights']={s:value*exposure for s,value in w.items()};dec['allocation_policy']=mode
 dec['buys']=[dict(symbol=s,budget=state['equity']*w[s]*exposure,score=float(score.at[prev,s]) if pd.notna(score.at[prev,s]) else None,
  atr=float(atr.at[prev,s]) if pd.notna(atr.at[prev,s]) else state['positions'][s]['atr'],reference_close=float(close.at[prev,s])) for s in targets]
 return dec

def run(cfg,cost,d,f,base,gaps,x,ctx,start='2019-01-02'):
 if not cfg:return g.run(g.BASE,cost,d,f,base,gaps,x,ctx,start)
 ctx=dict(ctx,raw_per_signal=d['raw_per_signal'])
 score,mask=z.signals(f,base,x,g.BASE);baseplan=g.planner(g.BASE,gaps,f['groups'][0],ctx)
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  out=baseplan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers)
  return decorate(out,state,day,dates,close,atr,score,f['groups'][0],ctx,cfg)
 def execute(name,state,decision,bars,dates,sector=None):return z.t.a.old.prior.net_execute(name,state,decision,bars,dates,cost_bps=cost,sector=None)
 h=z.h;oldplan,oldexec=h.plan,h.execute;ot,os=h.terminal_baskets.apply,h.spin_distributions.apply
 try:
  h.plan=plan;h.execute=execute;h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
  out=h.run_account('baseline',d,start,g.CONFIG['end'],{'baseline':dict(score=score,mask=mask,atr=f['atr'],ma5=f['ma5'])})
 finally:h.plan,h.execute=oldplan,oldexec;h.terminal_baskets.apply,h.spin_distributions.apply=ot,os
 if out['status']!='complete':raise ValueError(str({k:out.get(k) for k in ['status','error','blocked_session']}))
 assert all(v['signal_date']<v['session'] for v in out['decisions']);assert all(v['cash']>=-1e-6 and v['positions']<=3 and v['exposure']<=1+1e-8 for v in out['daily'])
 return out

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite allocation')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=g.protected_hashes();d,f,base,gaps,x,ctx,_=g.prepare(P);summary={}
 for name,cfg in VARIANTS.items():
  for cost in [10,30]:
   out=run(cfg,cost,d,f,base,gaps,x,ctx)
   if name=='control':
    ref=json.loads((g.P/f'control25_cost{cost}.json').read_text());assert out['orders']==ref['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in ref['daily']],rtol=1e-12,atol=1e-7)
   z.save(P,f'{name}_cost{cost}',out,summary)
 b10=summary['control_cost10'];b30=summary['control_cost30'];dev=[];growth=[]
 for k in VARIANTS:
  if k=='control':continue
  a=summary[k+'_cost10'];b=summary[k+'_cost30']
  if b['development']['max_drawdown']>=-.35 and a['development']['cagr']>=b10['development']['cagr'] and b['development']['cagr']>=b30['development']['cagr']:dev.append(k)
  if a['full']['max_drawdown']>=-.35 and b['full']['cagr']>b30['full']['cagr'] and b['ex_recent']['cagr']>=b30['ex_recent']['cagr']:growth.append(k)
 selection=dict(development_eligible=dev,development_selected=max(dev,key=lambda k:summary[k+'_cost30']['development']['cagr']) if dev else None,retrospective_eligible=growth,retrospective_growth_selected=max(growth,key=lambda k:summary[k+'_cost10']['full']['cagr']) if growth else None)
 (P/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n');assert protected==g.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,control_reproduced=True,no_leverage=True,partial_realizations_disclosed=True),indent=2)+'\n');print('ALLOCATION_COMPLETE',len(summary),selection,flush=True)
if __name__=='__main__':main()
