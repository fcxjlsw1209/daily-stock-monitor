"""Daily growth study: causal risk/context and holding management; no live writes."""
from pathlib import Path
import sys,json,hashlib,functools
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'daily_frontier_research'))
import frontier as z
BASE={**z.BASE,'tp':None,'cap':30,'sl':2.5}
VARIANTS={
 'control2':dict(sl=2.),'control25':{},
 'spy_entry':dict(entry_gate='spy200'),
 'qqq_entry':dict(entry_gate='qqq200'),
 'either_entry':dict(entry_gate='either200'),
 'breadth40_entry':dict(entry_gate='breadth40'),
 'breadth_trend_entry':dict(entry_gate='breadth_trend'),
 'crash_entry':dict(entry_gate='no_crash'),
 'spy_cash':dict(entry_gate='spy200',liquidate_gate='spy_bear5'),
 'dual_cash':dict(entry_gate='either200',liquidate_gate='dual_bear5'),
 'new_budget_vol30':dict(budget_vol=.30),
 'new_budget_vol40':dict(budget_vol=.40),
 'cap45':dict(cap=45),'cap60':dict(cap=60),
 'strong_cap45':dict(strong_cap=45),'strong_cap60':dict(strong_cap=60),
 'weak_cap15':dict(weak_cap=15),
 'late_trail3':dict(trail_atr=3.,trail_activate=3.),
 'late_trail4':dict(trail_atr=4.,trail_activate=3.),
 'breakeven_after3':dict(breakeven_atr=3.),
 'gap15':dict(gap=.15),'gap20':dict(gap=.20),'gap25':dict(gap=.25),
 'protect_trend':dict(protect_trend=True),
 'no_replace':dict(gap=2.),
 'retain_lone':dict(keep_lone=True),
 'sector2':dict(sector_cap=2)}
CONFIG=dict(start='2019-01-02',end='2026-10-06',variants=VARIANTS,base=BASE,cost_bps=[10,30],
 declared_before_this_batch_results=True,all_history_previously_inspected=True,
 objective='Improve full-period compounded annual return while checking pre-recent history, development, drawdown and higher fees. No guarantee of improvement.',
 unchanged='Available-data historical NDX100 union SP100 Top32 plus17 unlevered ETFs; four groups25%; same cached causal score and models; pullback score>=.75 plusshock2; next-open fills; max3,minimum2 except explicit retain_lone. No leverage, no cash yield, no tax.',
 context='Signal-close SPY/QQQ>SMA200, either one or pool breadth aboveSMA100>=40%; breadth among currently valid pool. no_crash excludes SPY63return<0 AND20dayvol>1.5*126dayvol.',
 cash_gate='Require5 completed consecutive below-MA sessions (both benchmarks for dual). Entry gate also applies. Portfolio sold next open, never current close.',
 holding='Strong cap45/60 if priorclose>=entry+1entryATR, aboveSMA20,momentumgroup>=.60. Otherwise30. Weak cap15 if belowSMA20 or momentum<.40. Initial2.5ATR stop and rank exit always remain.',
 profit_management='Trailing highest CLOSE activates after peak>=entry+3entryATR, retreats3/4entryATR; breakeven activates after peak>=entry+3ATR then close<entry. Next-open exit, gaps not limited.',
 protect_trend='Skip optional replacement only while priorclose>=entry+1ATR and aboveSMA20; hard exits unchanged.',
 volatility_budget='New entry budgets only multiplied by min(1,target_vol/projected_equalweight63dayportfolio_vol). Existing shares not resized. This is not an exact ongoing portfolio volatility target.',
 development_selection='Among noncontrols with developmentDD30>=-35%,developmentCAGR10>=control25 and developmentCAGR30>=control25, highest reused2019-2023 CAGR30. Full and recent metrics not used in this selection.',
 retrospective_growth_selection='Separately, among noncontrols with fullDD10>=-35%,fullCAGR30>control25 and ex_recentCAGR30>=control25, highest fullCAGR10; explicitly retrospective.',
 data_certified=False,performance_publishable=False,no_observation_changes=True,
 references=['https://www.nber.org/papers/w22208','https://www.aqr.com/Insights/Research/Journal-Article/Momentum-Crashes'],
 reference_scope='Only motivations for volatility/regime research, not validation or replication. Long-short momentum crash evidence is not directly identical to this long-only strategy.')

def protected_hashes():
 x=z.protected_hashes()
 for file in [Path(z.__file__),z.P/'FINAL_VALIDATION.json',z.P/'REPORT.txt',z.P/'TRIAL_REGISTRY.json',z.P/'robustness/stop25_cost10.json']:
  x[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
 return x

def context(d,f):
 a=d['adj'];ret=a.pct_change(fill_method=None);spy=a['SPY'];qqq=a['QQQ'];s=spy>spy.rolling(200).mean();q=qqq>qqq.rolling(200).mean()
 breadth=(a>a.rolling(100).mean()).where(f['valid']).mean(axis=1)
 crash=(spy/spy.shift(63)-1).lt(0)&ret['SPY'].rolling(20).std().gt(1.5*ret['SPY'].rolling(126).std())
 return dict(spy200=s,qqq200=q,either200=s|q,breadth40=breadth.ge(.4),breadth_trend=breadth.ge(.4)&(s|q),no_crash=~crash,
  spy_bear5=(~s).rolling(5,min_periods=5).sum().eq(5),dual_bear5=(~(s|q)).rolling(5,min_periods=5).sum().eq(5),
  ma20=d['close'].rolling(20).mean(),ret=ret)

def cap_for(cfg,p,price,ma,mom):
 cap=cfg['cap']
 if cfg.get('strong_cap') and price>=p['entry_price']+p['atr'] and price>ma and mom>=.6:cap=cfg['strong_cap']
 if cfg.get('weak_cap') and (price<ma or mom<.4):cap=cfg['weak_cap']
 return cap

def exit_reason(cfg,p,age,price,peak,ma,mom,score):
 if age>=cap_for(cfg,p,price,ma,mom):return 'holding_limit'
 if price<p['entry_price']-cfg['sl']*p['atr']:return 'close_stop_atr'
 if cfg.get('trail_atr') is not None and peak>=p['entry_price']+cfg.get('trail_activate',1.)*p['atr'] and price<peak-cfg['trail_atr']*p['atr']:return 'close_trailing_atr'
 if cfg.get('breakeven_atr') and peak>=p['entry_price']+cfg['breakeven_atr']*p['atr'] and price<p['entry_price']:return 'close_breakeven_after_gain'
 if pd.notna(score) and score<cfg['rank_floor']:return 'rank_deterioration'
 return None

def planner(cfg,gaps,momentum,ctx):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev];gap=cfg['gap'] if cfg['gap'] is not None else float(gaps.loc[prev])
  if cfg.get('liquidate_gate') and bool(ctx[cfg['liquidate_gate']].loc[prev]):
   return dict(session=str(day.date()),signal_date=str(prev.date()),sells=[dict(symbol=s,reason='market_bear_cash') for s in state['positions']],buys=[],candidate_count=0,equity_basis=state['equity'],replacement_pairs=[],gap=gap)
  pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)]
  remaining=set(state['positions']);sells=[];pairs=[]
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s]
   peak=max(p['entry_price'],float(close.loc[pd.Timestamp(p['entry_date']):prev,s].max())) if cfg.get('trail_atr') or cfg.get('breakeven_atr') else price
   why=exit_reason(cfg,p,age,price,peak,ctx['ma20'].at[prev,s],momentum.at[prev,s],sc[s])
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={v['symbol'] for v in sells};selected=[]
  def feasible(s,held):
   return s not in held|sold and issuers[s] not in {issuers[t] for t in held} and sum(sectors[t]==sectors[s] for t in held)<cfg.get('sector_cap',3)
  for s,v in pool.items():
   if len(remaining)+len(selected)>=3:break
   if feasible(s,remaining|{v[0] for v in selected}):selected.append((s,v))
  for held in sorted(remaining,key=lambda s:(float(sc[s]) if pd.notna(sc[s]) else 2.,s)):
   p=state['positions'][held];age=i-dates.get_loc(pd.Timestamp(p['entry_date']))
   if age<cfg['min_age'] or pd.isna(sc[held]):continue
   if cfg.get('protect_trend') and close.at[prev,held]>=p['entry_price']+p['atr'] and close.at[prev,held]>ctx['ma20'].at[prev,held]:continue
   others=(remaining-{held})|{v[0] for v in selected}
   challenger=next(((s,v) for s,v in pool.items() if s!=held and s not in state['positions'] and feasible(s,others) and v-sc[held]>gap),None)
   if challenger is None:continue
   s,v=challenger;sells.append(dict(symbol=held,reason='matched_better_candidate'));sold.add(held);remaining.remove(held);selected.append((s,v));pairs.append(dict(sell=held,buy=s,gap=float(v-sc[held]),required_gap=gap))
  n=len(remaining)+len(selected)
  if n<2 and not (cfg.get('keep_lone',False) and remaining):sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);selected=[];n=0
  scale=1.
  if cfg.get('budget_vol') and selected:
   targets=sorted(remaining|{v[0] for v in selected});sample=ctx['ret'].loc[:prev,targets].tail(63)
   if sample.notna().all().all() and len(sample)==63:
    estimate=float(sample.mean(axis=1).std()*np.sqrt(252));scale=min(1.,cfg['budget_vol']/max(estimate,1e-8))
   else:scale=0.
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,
   buys=[dict(symbol=s,budget=state['equity']/n*scale,score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],
   candidate_count=len(pool),equity_basis=state['equity'],replacement_pairs=pairs,gap=gap)
 return plan

def run(cfg,cost,d,f,base,gaps,x,ctx,start='2019-01-02',end=None):
 score,mask=z.signals(f,base,x,cfg)
 if cfg.get('entry_gate'):mask=mask.mul(ctx[cfg['entry_gate']],axis=0)
 h=z.h;oldplan,oldexec=h.plan,h.execute;ot,os=h.terminal_baskets.apply,h.spin_distributions.apply
 try:
  h.plan=planner(cfg,gaps,f['groups'][0],ctx);h.execute=functools.partial(z.t.a.old.prior.relaxed_daily,cost_bps=cost)
  h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
  out=h.run_account('baseline',d,start,end or CONFIG['end'],{'baseline':dict(score=score,mask=mask,atr=f['atr'],ma5=f['ma5'])})
 finally:h.plan,h.execute=oldplan,oldexec;h.terminal_baskets.apply,h.spin_distributions.apply=ot,os
 if out['status']!='complete':raise ValueError(str({k:out.get(k) for k in ['status','error','blocked_session']}))
 assert all(v['signal_date']<v['session'] for v in out['decisions'])
 assert all(v['cash']>=-1e-6 and v['positions']<=3 and v['exposure']<=1+1e-8 for v in out['daily'])
 return out

def prepare(folder=P):
 oldp=z.t.P;z.t.P=folder
 try:d,f,base,gaps,audit=z.t.prepare()
 finally:z.t.P=oldp
 return d,f,base,gaps,z.extras(d,f),context(d,f),audit

def select(summary):
 b10=summary['control25_cost10'];b30=summary['control25_cost30']
 dev=[k for k in VARIANTS if not k.startswith('control') and summary[k+'_cost30']['development']['max_drawdown']>=-.35 and summary[k+'_cost10']['development']['cagr']>=b10['development']['cagr'] and summary[k+'_cost30']['development']['cagr']>=b30['development']['cagr']]
 growth=[k for k in VARIANTS if not k.startswith('control') and summary[k+'_cost10']['full']['max_drawdown']>=-.35 and summary[k+'_cost30']['full']['cagr']>b30['full']['cagr'] and summary[k+'_cost30']['ex_recent']['cagr']>=b30['ex_recent']['cagr']]
 return dict(development_eligible=dev,development_selected=max(dev,key=lambda k:summary[k+'_cost30']['development']['cagr']) if dev else None,
  retrospective_eligible=growth,retrospective_growth_selected=max(growth,key=lambda k:summary[k+'_cost10']['full']['cagr']) if growth else None)

def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite this research')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');before=protected_hashes();(P/'SOURCE_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
 d,f,base,gaps,x,ctx,audit=prepare();(P/'GAP_CHOICES.json').write_text(json.dumps(audit,indent=2)+'\n');summary={}
 for name,overlay in VARIANTS.items():
  for cost in CONFIG['cost_bps']:
   out=run({**BASE,**overlay},cost,d,f,base,gaps,x,ctx)
   if name.startswith('control'):
    file=(z.P/'exits'/f'no_profit_cap30_cost{cost}.json') if name=='control2' else (z.P/'robustness'/f'stop25_cost{cost}.json')
    reference=json.loads(file.read_text());assert out['orders']==reference['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in reference['daily']],rtol=1e-12,atol=1e-7)
   z.save(P,f'{name}_cost{cost}',out,summary)
 selection=select(summary);(P/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
 assert before==protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,controls_reproduced=True,no_leverage=True),indent=2)+'\n')
 print('GROWTH_COMPLETE',len(summary),selection,flush=True)
if __name__=='__main__':main()
