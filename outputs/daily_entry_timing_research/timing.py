"""Sixth finite daily study: entry timing and downside-shock controls. Research only."""
from pathlib import Path
import sys, json, hashlib, functools, copy
import numpy as np
import pandas as pd
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent/'daily_adaptation_research'))
import adaptation as a
from stable_ties import schedule
r, h = a.r, a.h
KINDS = ['fixed10', 'stable_control', 'armed_up3', 'armed_break3', 'hybrid_recovery',
         'intraday_gate', 'intraday_shape', 'shock_pause2', 'stop_pause3', 'fast_highvol']
CONFIG = dict(start='2019-01-02', end='2026-10-06', cost_bps=[10,30], variants=KINDS,
 declared_before_results=True, all_history_previously_inspected=True,
 data_certified=False, performance_publishable=False, initial_capital=100000,
 universe='Unchanged available-data historical NDX100 union SP100 past vendor-cap Top32 plus17 unlevered ETFs',
 base='Prior numerically stable monthly gap policy; frozen cached causal score, four groups25% each, max3,min2,long-only,no leverage,stop2ATR,profit3ATR,cap10; exits at next open after prior-close signal',
 hypotheses={
 'fixed10':'Exact prior fixed .10 control',
 'stable_control':'Exact prior stable126/252 monthly gap control',
 'armed_up3':'A qualifying setup (event and score>=.75) arms the next3sessions. Use most recent setup score, not a fresh oversold score. Require current close>previous close and close in upperhalf; current validity, positive R12-1, aboveSMA100, and close<=setupclose+setupATR. Held ranking uses current original score.',
 'armed_break3':'Same armed setup but confirmation is close>previous high instead of upday/upperhalf.',
 'hybrid_recovery':'Original immediate qualified entries OR armed_up3; larger of eligible immediate/armed scores. Other rules unchanged.',
 'intraday_gate':'Original entries additionally require negative sum of past3 log(close/open), distinguishing daytime pressure from overnight declines.',
 'intraday_shape':'Replace half of shape group with rank of negative3day log(close/open) / past20day close-return standard deviation. Remaining3groups and frozen model unchanged. Recompute final score threshold; this proxy is not a news identifier.',
 'shock_pause2':'Block new entry for shock day and following2sessions when open/previousclose-1 < -2*prior20day return stdev AND volume>1.5*prior20day average. Other entries unchanged. A price-volume shock proxy, no inferred earnings date.',
 'stop_pause3':'After an actually executed ATR-stop exit, prohibit same security reentry on next3sessions. Other exit types have no cooldown.',
 'fast_highvol':'If entry-signal63day volatility rank in contemporaneously valid pool>=2/3, holding cap5 instead of10; stored entry-date condition, never future volatility.'},
 selection='Select noncontrol by reused2019-2023 CAGR10, only if development DD10>=-35% and development CAGR30>=stable_control development CAGR30. Recent/full results not used in selection; no automatic observer promotion.',
 robustness='For selected candidate (if any), cost50,2024 cash restart, and predeclared nearest parameter neighbors. No best-neighbor reselection.',
 references=['https://www.nber.org/papers/w30917','https://www.nber.org/papers/w17653','https://doi.org/10.1142/S2010139219500022'],
 reference_scope='Primary research motivates reversal speed and daytime/overnight distinctions; this study is not a replication and sources do not validate these parameters or returns.')


def protected_hashes():
 result=a.old.protected_hashes()
 files=[Path(a.__file__),Path(sys.modules['stable_ties'].__file__),Path(a.old.__file__),
        a.P/'SCORES.npz',a.P/'AXES.json',a.P/'stable_ties/2019_cost10.json',
        r.U/'historical_backtest.py',r.U/'union_execution.py',
        P.parent/'four_factor_rebound/vendor_cap/COLLECTION.json']
 result.update({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
 return result


def armed_signals(d, f, score, window=3, confirmation='up'):
 """Expire/replace setups causally; a setup never confirms on its own day."""
 c=d['close']; valid=f['valid']; base=f['event'] & score.ge(.75)
 annual=d['adj'].shift(21)/d['adj'].shift(252)-1
 continued=valid & annual.gt(0) & d['adj'].gt(d['adj'].rolling(100).mean())
 values=np.full(score.shape,np.nan); origin=np.full(score.shape,-1,dtype=np.int64)
 # Descending lags: the most recent setup overrides earlier ones, even if it later fails the no-chase test.
 for lag in range(window,0,-1):
  qualified=base.shift(lag,fill_value=False).to_numpy()
  values=np.where(qualified,score.shift(lag).to_numpy(),values)
  rows=np.broadcast_to(np.arange(len(c))[:,None]-lag,score.shape)
  origin=np.where(qualified,rows,origin)
 sc=pd.DataFrame(values,index=c.index,columns=c.columns)
 setup_c=np.full(c.shape,np.nan);setup_atr=setup_c.copy()
 for lag in range(1,window+1):
  use=origin==np.arange(len(c))[:,None]-lag
  setup_c=np.where(use,c.shift(lag).to_numpy(),setup_c)
  setup_atr=np.where(use,f['atr'].shift(lag).to_numpy(),setup_atr)
 if confirmation=='up':
  location=(c-d['low'])/(d['high']-d['low']).replace(0,np.nan)
  trigger=c.gt(c.shift()) & location.ge(.5)
 elif confirmation=='break':trigger=c.gt(d['high'].shift())
 else:raise ValueError('Unknown confirmation')
 mask=continued & trigger & c.le(setup_c+setup_atr) & sc.ge(.75)
 return sc.where(mask),mask,origin


def entry_inputs(d,f,score,kind,window=3,shock_days=2):
 c=d['close'];ret=d['adj'].pct_change(fill_method=None);base=f['event'] & score.ge(.75)
 candidate=score;current=score;mask=base
 intraday=np.log(c/d['opening']).rolling(3).sum()
 if kind in ['armed_up3','armed_break3','hybrid_recovery']:
  armed,armed_mask,_=armed_signals(d,f,score,window,'break' if kind=='armed_break3' else 'up')
  if kind=='hybrid_recovery':
   candidate=pd.DataFrame(np.fmax(score.where(base).to_numpy(),armed.to_numpy()),index=c.index,columns=c.columns)
   mask=base|armed_mask
  else:candidate,mask=armed,armed_mask
 elif kind=='intraday_gate':mask=base & intraday.lt(0)
 elif kind=='intraday_shape':
  extra=(-intraday/ret.rolling(20).std().replace(0,np.nan)).where(f['valid']).rank(axis=1,pct=True)
  current=score+.125*(extra-f['groups'][3]);candidate=current;mask=f['event']&current.ge(.75)
 elif kind=='shock_pause2':
  shock=(d['opening']/c.shift()-1).lt(-2*ret.rolling(20).std().shift()) & d['volume'].gt(1.5*d['volume'].rolling(20).mean().shift())
  mask=base & ~shock.rolling(shock_days+1,min_periods=1).max().astype(bool)
 return current,candidate,mask


def planner(kind,gaps,candidate,volrank,cooldowns,pause=3,fast_cap=5):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev];gap=float(gaps.loc[prev]);allowed=mask.loc[prev].copy()
  if kind=='stop_pause3':
   for symbol,exit_i in cooldowns.items():
    if i<=exit_i+pause:allowed[symbol]=False
  pool=candidate.loc[prev].where(allowed).dropna().sort_values(ascending=False,kind='stable')
  pool=pool[pool.index.isin(reps)];remaining=set(state['positions']);sells=[];pairs=[]
  for s,p in state['positions'].items():
   entry_i=dates.get_loc(pd.Timestamp(p['entry_date']));age=i-entry_i;price=close.at[prev,s];why=None;cap=10
   if kind=='fast_highvol' and volrank.at[dates[entry_i-1],s]>=2/3:cap=fast_cap
   if age>=cap:why='holding_limit'
   elif price<p['entry_price']-2*p['atr']:why='close_stop_atr'
   elif price>p['entry_price']+3*p['atr']:why='close_take_profit_atr'
   elif pd.notna(sc[s]) and sc[s]<.35:why='rank_deterioration'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={v['symbol'] for v in sells};selected=[]
  def feasible(s,held):return s not in held|sold and issuers[s] not in {issuers[t] for t in held}
  for s,value in pool.items():
   if len(remaining)+len(selected)>=3:break
   if feasible(s,remaining|{v[0] for v in selected}):selected.append((s,value))
  for held in sorted(remaining,key=lambda s:(float(sc[s]) if pd.notna(sc[s]) else 2.,s)):
   age=i-dates.get_loc(pd.Timestamp(state['positions'][held]['entry_date']))
   if age<2 or pd.isna(sc[held]):continue
   others=(remaining-{held})|{v[0] for v in selected}
   challenger=next(((s,v) for s,v in pool.items() if s!=held and s not in state['positions'] and feasible(s,others) and v-sc[held]>gap),None)
   if challenger is None:continue
   s,v=challenger;sells.append(dict(symbol=held,reason='matched_better_candidate'));sold.add(held);remaining.remove(held);selected.append((s,v))
   pairs.append(dict(sell=held,buy=s,gap=float(v-sc[held]),required_gap=gap))
  n=len(remaining)+len(selected)
  if n<2:sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);selected=[];n=0
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,
   buys=[dict(symbol=s,budget=state['equity']/n,score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],
   candidate_count=len(pool),equity_basis=state['equity'],replacement_pairs=pairs,gap=gap)
 return plan


def prepare():
 m,d=r.prepare();d=r.apply(m,d,P/'pool');f,_=r.features(d)
 axes=json.loads((a.P/'AXES.json').read_text())
 assert axes['symbols']==list(d['close'].columns) and axes['dates']==[str(v.date()) for v in d['close'].index]
 with np.load(a.P/'SCORES.npz',allow_pickle=False) as z:score=pd.DataFrame(z['base'],index=d['close'].index,columns=d['close'].columns)
 refs={}
 for gap,file in [(.1,a.P/'fixed10_cost10.json'),(.15,a.P/'fixed15_cost10.json'),(.2,a.old.P/'robustness/gap20_cost10.json')]:
  rows=json.loads(file.read_text())['daily'];refs[gap]=pd.Series({pd.Timestamp(v['date']):v['return'] for v in rows})
 gaps,audit=schedule(score.index,refs)
 return d,f,score,gaps,audit


def run(kind,cost,d,f,score,gaps,start=None,window=3,shock_days=2,pause=3,fast_cap=5):
 current,candidate,mask=entry_inputs(d,f,score,kind,window,shock_days)
 volrank=d['adj'].pct_change(fill_method=None).rolling(63).std().where(f['valid']).rank(axis=1,pct=True)
 chosen_gaps=pd.Series(.1,index=gaps.index) if kind=='fixed10' else gaps
 cooldowns={}
 def execute(name,state,decision,bars,dates,sector=None):
  result=a.old.prior.relaxed_daily(name,state,decision,bars,dates,cost_bps=cost,sector=None)
  for trade in result[2]:
   if trade['reason']=='close_stop_atr':cooldowns[trade['symbol']]=dates.get_loc(pd.Timestamp(trade['exit_date']))
  return result
 oldplan,oldexec=h.plan,h.execute;ot,os=h.terminal_baskets.apply,h.spin_distributions.apply
 try:
  h.plan=planner(kind,chosen_gaps,candidate,volrank,cooldowns,pause,fast_cap);h.execute=execute
  h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
  out=h.run_account('baseline',d,start or CONFIG['start'],CONFIG['end'],{'baseline':dict(score=current,mask=mask,atr=f['atr'],ma5=f['ma5'])})
 finally:h.plan,h.execute=oldplan,oldexec;h.terminal_baskets.apply,h.spin_distributions.apply=ot,os
 if out['status']!='complete':raise ValueError(str({k:out.get(k) for k in ['status','error','blocked_session']}))
 assert all(x['signal_date']<x['session'] for x in out['decisions'])
 assert all(x['cash']>=-1e-6 and x['positions']<=3 and x['exposure']<=1+1e-8 for x in out['daily'])
 return out


def save_result(folder,key,out):
 (folder/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
 result=a.old.prior.metrics(out);result['diagnostics']=a.old.diagnostics(out)
 print(key,'CAGR',result['full']['cagr'],'DEV',result['development'].get('cagr'),'RECENT',result['last252']['return_'],flush=True)
 return result


def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Completed or partial results already exist; do not overwrite')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 protected=protected_hashes();(P/'SOURCE_HASHES.json').write_text(json.dumps(protected,indent=2)+'\n')
 d,f,score,gaps,audit=prepare();(P/'GAP_CHOICES.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('FEATURES_READY; cached predictions reused, no retraining',flush=True);summary={}
 for kind in KINDS:
  for cost in CONFIG['cost_bps']:
   out=run(kind,cost,d,f,score,gaps)
   if kind in ['fixed10','stable_control']:
    file=a.P/(f'fixed10_cost{cost}.json' if kind=='fixed10' else f'stable_ties/2019_cost{cost}.json')
    prior=json.loads(file.read_text());assert prior['orders']==out['orders']
    np.testing.assert_allclose([v['equity'] for v in prior['daily']],[v['equity'] for v in out['daily']],rtol=1e-12,atol=1e-7)
   key=f'{kind}_cost{cost}';summary[key]=save_result(P,key,out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 eligible=[k for k in KINDS[2:] if summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=summary['stable_control_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 (P/'SELECTION.json').write_text(json.dumps(dict(selected=selected,eligible=eligible,criterion=CONFIG['selection']),indent=2)+'\n')
 assert protected==protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),controls_reproduced=True,protected_unchanged=True,cached_predictions=True,no_leverage=True,signals_before_fills=True,data_certified=False),indent=2)+'\n')
 print('COMPLETE',selected,flush=True)

if __name__=='__main__':main()
