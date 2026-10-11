"""Finite, declared mechanism ablations; no live account or optimizer imports."""
from pathlib import Path
import sys, json, functools, hashlib, copy
import numpy as np
import pandas as pd
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / 'strategy_improvements'))
import research as prior
r = prior.r
h = r.h

CONFIG = {
 'start': '2019-01-02', 'end': '2026-10-06', 'cost_bps': [10, 30],
 'variants': ['baseline', 'five_day_clock', 'no_rank_replacement', 'recovery_exit',
              'event_ranking', 'conditional_rebound', 'vol_scaled_ridge', 'medium_momentum'],
 'declared_before_results': True, 'no_grid_search': True, 'no_live_changes': True,
 'four_groups_equal_weight': True,
 'hypotheses': {
  'five_day_clock': 'Only holding limit 10 to 5 sessions; all other exits unchanged.',
  'no_rank_replacement': 'Remove better-candidate replacement only; retain 10-session, ATR, and weak-rank exits.',
  'recovery_exit': 'Add prior-close recovery above SMA5 and above entry price; next open sell.',
  'event_ranking': 'Retain same event and four-group score, remove absolute .75 threshold; top3 only among events. Exposure likely increases.',
  'conditional_rebound': 'Rebuild historical rebound and path with the same current event; separate deep(z<-1) and mild events, pooled-shrinkage20,756-session,6-session outcome maturity. Replace rebound-group history and path-group; model trained on these new features.',
  'vol_scaled_ridge': 'Train same Ridge on five-day net return / signal-date20d volatility, convert predictions back using current volatility; Ridge remains half of rebound group.',
  'medium_momentum': 'Replace only momentum group with average percentile rank of 63/126-day return per63-day volatility; refit same Ridge with changed features.'},
 'selection_policy': 'Report every result; 2019-2023 is reused development, 2024+ and last252 also previously inspected; none pristine OOS.',
 'data_certified': False,
}


def planner(kind):
 base = prior.daily_plan(.15)
 if kind in ['baseline', 'event_ranking', 'conditional_rebound', 'vol_scaled_ridge', 'medium_momentum']:
  return base
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev]
  pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)]
  remaining=set(state['positions']);sells=[]
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];why=None
   if age >= (5 if kind=='five_day_clock' else 10): why='holding_limit'
   elif price < p['entry_price']-2*p['atr']: why='close_stop_atr'
   elif price > p['entry_price']+3*p['atr']: why='close_take_profit_atr'
   elif kind=='recovery_exit' and price>ma5.at[prev,s] and price>p['entry_price']: why='close_recovered_sma5'
   elif pd.notna(sc[s]) and sc[s]<.35: why='rank_deterioration'
   elif kind!='no_rank_replacement' and age>=2 and len(pool) and pool.iloc[0]-sc[s]>.15: why='better_candidate'
   if why: sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={x['symbol'] for x in sells};selected=[]
  for s,value in pool.items():
   if len(remaining)+len(selected)>=3: break
   held=remaining|{x[0] for x in selected}
   if s in held|sold or issuers[s] in {issuers[t] for t in held}: continue
   selected.append((s,value))
  n=len(remaining)+len(selected)
  if n<2:
   sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);selected=[];n=0
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,
   buys=[dict(symbol=s,budget=state['equity']/n,score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],
   candidate_count=len(pool),equity_basis=state['equity'])
 return plan


def condition_features(d,df):
 f=dict(df);f['groups']=list(df['groups']);f['X']=list(df['X']);a=d['adj'];c=d['close'];vol=a.pct_change(fill_method=None).rolling(20).std()
 z=a.pct_change(3,fill_method=None)/(vol*np.sqrt(3));deep=z.lt(-1);ao=d['opening']*a/c
 ah=d['high']*a/c;al=d['low']*a/c
 hi=pd.concat([ah.shift(-i) for i in range(1,6)],keys=range(5)).groupby(level=1).max().reindex(a.index)
 lo=pd.concat([al.shift(-i) for i in range(1,6)],keys=range(5)).groupby(level=1).min().reindex(a.index)
 path=hi/ao.shift(-1)-1+lo/ao.shift(-1)-1
 rank=lambda x:x.where(df['valid']).rank(axis=1,pct=True)
 def conditioned(y):
  yes=r.matured_average(y,df['event']&deep)
  no=r.matured_average(y,df['event']&~deep)
  return yes.where(deep,no)
 f['groups'][1]=rank(conditioned(df['y']))
 f['groups'][2]=.7*rank(conditioned(path))+.3*rank(vol)
 f['X'][1]=f['groups'][1];f['X'][2]=f['groups'][2]
 return f


def diagnostics(out):
 t=pd.DataFrame(out['trades']);reasons={}
 for reason,g in t.groupby('reason'):
  reasons[reason]=dict(count=len(g),win_rate=float((g.net_pnl>0).mean()),mean_net_return=float(g.net_return.mean()),mean_sessions=float(g.sessions.mean()),net_pnl=float(g.net_pnl.sum()))
 return dict(closed_trades=len(t),win_rate=float((t.net_pnl>0).mean()),mean_sessions=float(t.sessions.mean()),
  average_winner=float(t.loc[t.net_pnl>0,'net_return'].mean()),average_loser=float(t.loc[t.net_pnl<=0,'net_return'].mean()),
  reasons=reasons,mean_candidates=float(np.mean([v['candidate_count'] for v in out['decisions']])))


def main():
 P.mkdir(exist_ok=True);(P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 source_paths=[P.parent/'strategy_improvements/research.py',P.parent/'dual_strategy_research/research.py',r.U/'alias_resolved_inputs/MANIFEST.json']
 source_hash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
 m,d=r.prepare();d=r.apply(m,d,P/'pool');df,_=r.features(d);vol=d['adj'].pct_change(fill_method=None).rolling(20).std().clip(lower=.001)
 modeled={};audits={}
 for mode in ['baseline','conditional_rebound','vol_scaled_ridge','medium_momentum']:
  f=condition_features(d,df) if mode=='conditional_rebound' else dict(df)
  if mode=='vol_scaled_ridge': f['y']=df['y']/vol
  if mode=='medium_momentum':
   f['groups']=list(df['groups']);f['X']=list(df['X']);sig=d['adj'].pct_change(fill_method=None).rolling(63).std();rank=lambda x:x.where(df['valid']).rank(axis=1,pct=True)
   f['groups'][0]=(rank(d['adj'].pct_change(63,fill_method=None)/sig)+rank(d['adj'].pct_change(126,fill_method=None)/sig))/2
   f['X'][0]=f['groups'][0]
  dp,_,audit=r.model(f,6);audits[mode]=audit
  if mode=='vol_scaled_ridge':dp=dp*vol
  g=f['groups'];sc=(g[0]+.5*g[1]+.5*dp.where(df['valid']).rank(axis=1,pct=True)+g[2]+g[3])/4
  modeled[mode]=sc
  print('model_ready',mode,flush=True)
 (P/'TRAINING_AUDIT.json').write_text(json.dumps(audits,indent=2)+'\n');summary={};diags={}
 for cost in CONFIG['cost_bps']:
  for kind in CONFIG['variants']:
   sc=modeled.get(kind,modeled['baseline']);mask=df['event'] if kind=='event_ranking' else df['event']&sc.ge(.75)
   ff=dict(score=sc,mask=mask,atr=df['atr'],ma5=df['ma5']);h.plan=planner(kind);h.execute=functools.partial(prior.relaxed_daily,cost_bps=cost)
   ot=h.terminal_baskets.apply;os=h.spin_distributions.apply
   try:
    h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
    out=h.run_account('baseline',d,CONFIG['start'],CONFIG['end'],{'baseline':ff})
   finally:h.terminal_baskets.apply=ot;h.spin_distributions.apply=os
   key=f'{kind}_cost{cost}';(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
   assert out['status']=='complete',out.get('error');assert all(v['signal_date']<v['session'] for v in out['decisions'])
   assert all(v['cash']>=-1e-6 for v in out['daily'])
   if kind=='baseline':
    old=json.loads((P.parent/f'strategy_improvements/daily_baseline_cost{cost}.json').read_text())
    assert out['orders']==old['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in old['daily']],atol=1e-7,rtol=1e-12)
   summary[key]=prior.metrics(out);diags[key]=diagnostics(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(P/'DIAGNOSTICS.json').write_text(json.dumps(diags,indent=2)+'\n')
   print(key,json.dumps(summary[key]),flush=True)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in source_hash.items())
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,baseline_reproduced=True,frozen_source_hashes=source_hash,
  labels_mature=all(v['latest_label_session']<v['first_prediction'] for a in audits.values() for v in a),data_certified=False),indent=2)+'\n')


if __name__=='__main__':main()
