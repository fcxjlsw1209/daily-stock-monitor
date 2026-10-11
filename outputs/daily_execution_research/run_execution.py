"""Finite daily execution and pullback ablations. Research only; frozen accounts untouched."""
from pathlib import Path
import sys, json, functools, hashlib, copy
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'daily_joint_research'))
import experiment as previous
import adaptive
prior=previous.prior
r=previous.r
h=previous.h
KINDS=['baseline','validated_base','matched_replacement','net_targets','matched_net',
       'no_take_profit','trailing_profit','no_progress','sector_cap2',
       'residual_pullback','dry_pullback','validated_cost_margin']
CONFIG=dict(start='2019-01-02',end='2026-10-06',cost_bps=[10,30],variants=KINDS,
 declared_before_results=True,all_history_previously_inspected=True,data_certified=False,
 no_live_changes=True,initial_capital=100000,max_positions=3,min_positions=2,leverage=1,
 core='Four groups25% each; same historical available-vendorcap Top32+17 ETFs, monthly mature-label Ridge and126/252 skill gating, score>=.75, next open execution',
 hypotheses={
 'baseline':'Exact original daily control',
 'validated_base':'Exact last-round validated-base control; all other variants start here',
 'matched_replacement':'Mandatory exits first, fill spare slots, then weakest eligible held position can be replaced only by one distinct not-held admissible issuer, score gap>.15 and age>=2; reserve each challenger once',
 'net_targets':'Same old exit logic, but whenever holdings change solve cost-aware equal target weights, trade only differences; preserve shares between membership changes',
 'matched_net':'Predeclared combination of matched replacement and net target weights',
 'no_take_profit':'Remove only fixed3ATR profit exit; retain cap10,stop2ATR,rank and replacement exits',
 'trailing_profit':'Replace3ATR profit exit with prior-close trailing2ATR after highest close since entry exceeds entry+3ATR; max10 unchanged',
 'no_progress':'Additional exit after4sessions if prior close<=entry, next open fill',
 'sector_cap2':'Enforce at most2 holdings per contemporaneous sector when adding a candidate; other variants retain the actual relaxed daily control without sector cap',
 'residual_pullback':'Additional entry gate: trailing3session residual return below zero using SPY beta126 estimated through prior day; no shorting',
 'dry_pullback':'Replace half of pullback-shape group with inverse rank of mean3day dollar volume / mean20day dollar volume; a volume proxy, not share-turnover or news identification',
 'validated_cost_margin':'Combine prior skill gating with raw Ridge predicted5day netreturn>.003 only in months when Ridge is enabled; off months keep original rules'},
 selection='Compare reused2019-2023 development only: maximize CAGR10 among noncontrols with DD10>=-35% and CAGR30>=validated_base30 development CAGR. Recent period not a selection criterion; report all variants.',
 references=['https://www.nber.org/papers/w30917','https://www.nber.org/papers/w35840'],
 source_note='Papers motivate liquidity/residual reversal hypotheses only. These simple proxies do not replicate their published factors; no future earnings or news data used.',
 net_accounting='Net trims create partial realizations; report completed entry-date/symbol groups separately from realization fragments. Added shares retain original holding clock and weighted entry basis.',
 execution_caveat='Next-open prices and proportional fractional-share sizing are proxies; no intraday fills or claimed stop protection against gaps.')


def causal_features(d,valid):
 a=d['adj'];ret=a.pct_change(fill_method=None);market=ret['SPY']
 beta=ret.rolling(126).cov(market).div(market.rolling(126).var(),axis=0).shift(1)
 residual=ret-beta.mul(market,axis=0)
 dollar=d['close']*d['volume'];dry=dollar.rolling(3).mean()/dollar.rolling(20).mean()
 return dict(residual3=residual.rolling(3).sum(),dry_rank=(-dry).where(valid).rank(axis=1,pct=True))


def planner(kind, gap=.15, sector_cap=None):
 if kind in ['baseline','validated_base','residual_pullback','dry_pullback','validated_cost_margin']:
  return prior.daily_plan(gap)
 matched=kind in ['matched_replacement','matched_net'];net=kind in ['net_targets','matched_net'];sectorcap=sector_cap if sector_cap is not None else (2 if kind=='sector_cap2' else 3)
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev]
  pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)]
  remaining=set(state['positions']);sells=[];pairs=[]
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];why=None
   if age>=10:why='holding_limit'
   elif price<p['entry_price']-2*p['atr']:why='close_stop_atr'
   elif kind not in ['no_take_profit','trailing_profit'] and price>p['entry_price']+3*p['atr']:why='close_take_profit_atr'
   elif kind=='trailing_profit':
    peak=close.loc[p['entry_date']:prev,s].max()
    if peak>p['entry_price']+3*p['atr'] and price<peak-2*p['atr']:why='close_trailing_profit'
   if why is None and kind=='no_progress' and age>=4 and price<=p['entry_price']:why='no_progress_after4'
   if why is None and pd.notna(sc[s]) and sc[s]<.35:why='rank_deterioration'
   if why is None and not matched and age>=2 and len(pool) and pool.iloc[0]-sc[s]>gap:why='better_candidate'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={v['symbol'] for v in sells};selected=[]
  def feasible(s,held):
   return s not in held|sold and issuers[s] not in {issuers[t] for t in held} and sum(sectors[t]==sectors[s] for t in held)<sectorcap
  # Mandatory vacancies first; these purchases cannot be reused to justify a replacement.
  for s,value in pool.items():
   if len(remaining)+len(selected)>=3:break
   if feasible(s,remaining|{v[0] for v in selected}):selected.append((s,value))
  if matched:
   for held in sorted(remaining,key=lambda s:(float(sc[s]) if pd.notna(sc[s]) else 2.,s)):
    p=state['positions'][held];age=i-dates.get_loc(pd.Timestamp(p['entry_date']))
    if age<2 or pd.isna(sc[held]):continue
    others=(remaining-{held})|{v[0] for v in selected}
    challenger=next(((s,v) for s,v in pool.items() if s!=held and s not in state['positions'] and feasible(s,others) and v-sc[held]>gap),None)
    if challenger is None:continue
    s,v=challenger;sells.append(dict(symbol=held,reason='matched_better_candidate'));sold.add(held);remaining.remove(held);selected.append((s,v));pairs.append(dict(sell=held,buy=s,gap=float(v-sc[held])))
  n=len(remaining)+len(selected)
  if n<2:
   sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);remaining=set();selected=[];n=0
  result=dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,
   buys=[dict(symbol=s,budget=state['equity']/n,score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],
   candidate_count=len(pool),equity_basis=state['equity'])
  if matched:result['replacement_pairs']=pairs
  if net and (sells or selected):
   targets=sorted(remaining|{v[0] for v in selected});result['target_weights']={s:1/len(targets) for s in targets}
   result['buys']=[dict(symbol=s,budget=state['equity']/len(targets),score=float(sc[s]) if pd.notna(sc[s]) else None,atr=float(atr.at[prev,s]) if pd.notna(atr.at[prev,s]) else state['positions'][s]['atr'],reference_close=float(close.at[prev,s])) for s in targets]
  return result
 return plan


def diagnostics(out):
 t=pd.DataFrame(out['trades']);reasons={};episodes=[]
 if len(t):
  for reason,g in t.groupby('reason'):
   reasons[reason]=dict(count=len(g),net_pnl=float(g.net_pnl.sum()),mean_net_return=float(g.net_return.mean()),win_rate=float((g.net_pnl>0).mean()))
  for (symbol,entry),g in t.groupby(['symbol','entry_date']):
   # Any full exit closes the episode. Partial slices do not each count as a win.
   if not all(bool(v.get('partial',False)) for v in g.to_dict('records')):
    episodes.append(dict(symbol=symbol,entry=entry,pnl=float(g.net_pnl.sum())))
 winners=t[t.net_pnl>0] if len(t) else t
 losers=t[t.net_pnl<=0] if len(t) else t
 return dict(realization_fragments=len(t),completed_position_episodes=len(episodes),episode_win_rate=sum(v['pnl']>0 for v in episodes)/len(episodes) if episodes else None,
  mean_sessions=float(t.sessions.mean()) if len(t) else None,
  mean_winner=float(winners.net_return.mean()) if len(winners) else None,mean_loser=float(losers.net_return.mean()) if len(losers) else None,
  reasons=reasons,paired_replacements=sum(len(v.get('replacement_pairs',[])) for v in out['decisions']))


def run_variant(kind,cost,d,f,score,mask,start,end,gap=.15,sector_cap=None):
 h.plan=planner(kind,gap,sector_cap);h.execute=functools.partial(prior.net_execute if kind in ['net_targets','matched_net'] else prior.relaxed_daily,cost_bps=cost,sector=None)
 # run_account supplies sector by keyword; net executor accepts it, but daily control has no sector cap.
 if kind in ['net_targets','matched_net']:
  def execution(name,state,decision,bars,dates,cost_bps=cost,sector=None):
   return prior.net_execute(name,state,decision,bars,dates,cost_bps=cost,sector=None)
  h.execute=execution
 ot=h.terminal_baskets.apply;os=h.spin_distributions.apply
 try:
  h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
  out=h.run_account('baseline',d,start,end,{'baseline':dict(score=score,mask=mask,atr=f['atr'],ma5=f['ma5'])})
 finally:h.terminal_baskets.apply=ot;h.spin_distributions.apply=os
 if out['status']!='complete':raise RuntimeError(str({k:out.get(k) for k in ['status','error','blocked_session']}))
 assert all(v['signal_date']<v['session'] for v in out['decisions'])
 assert all(v['cash']>=-1e-6 and v['positions']<=3 and v['exposure']<=1+1e-8 for v in out['daily'])
 for dec in out['decisions']:
  pairs=dec.get('replacement_pairs',[])
  assert len({v['buy'] for v in pairs})==len(pairs)
 return out


def protected_hashes():
 paths=[P.parent/'strategy_improvements/research.py',P.parent/'dual_strategy_research/research.py',Path(previous.__file__),Path(adaptive.__file__),r.U/'alias_resolved_inputs/MANIFEST.json']
 for folder in ['daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor','monthly_buffer5_monitor']:
  paths += [p for p in (P.parent/folder).rglob('*') if p.is_file() and (p.suffix=='.sqlite' or p.name in ['FREEZE.json','config.json'] or 'plans' in p.parts)]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare_features():
 m,d=r.prepare();d=r.apply(m,d,P/'pool');f,_=r.features(d);base,_,audit=r.model(f,6)
 choice,choices=adaptive.choices(base.index,{'baseline':adaptive.matured_skill(base,f['y'],f['event'])},['baseline'])
 original=adaptive.blended_score(f,base,.5);validated=sum(f['groups'])/4
 validated.loc[choice.eq('baseline')]=original.loc[choice.eq('baseline')]
 return m,d,f,base,choice,original,validated,audit,choices


def main():
 resume='--resume' in sys.argv
 if (P/'SUMMARY.json').exists() and not resume:raise RuntimeError('Existing results; use --resume only with unchanged configuration and protected inputs')
 before=protected_hashes()
 if resume:
  assert json.loads((P/'CONFIG.json').read_text())==CONFIG
  assert json.loads((P/'SOURCE_HASHES.json').read_text())==before
 else:
  (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');(P/'SOURCE_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
 m,d,f,base,choice,original,validated,audit,choices=prepare_features();extra=causal_features(d,f['valid'])
 (P/'TRAINING_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');(P/'CHOICES.json').write_text(json.dumps(choices,indent=2)+'\n')
 print('FEATURES_READY',flush=True);summary={}
 for cost in CONFIG['cost_bps']:
  for kind in KINDS:
   sc=original if kind=='baseline' else validated
   if kind=='dry_pullback':sc=sc+.125*(extra['dry_rank']-f['groups'][3])
   mask=f['event']&sc.ge(.75)
   if kind=='residual_pullback':mask &= extra['residual3'].lt(0)
   if kind=='validated_cost_margin':mask &= base.gt(.003).where(choice.eq('baseline'),True,axis=0)
   key=f'{kind}_cost{cost}';result_path=P/(key+'.json')
   out=json.loads(result_path.read_text()) if resume and result_path.exists() else run_variant(kind,cost,d,f,sc,mask,CONFIG['start'],CONFIG['end'])
   if kind in ['baseline','validated_base']:
    old=json.loads((previous.P/'adaptive'/f'{kind}_cost{cost}.json').read_text());assert old['orders']==out['orders']
    np.testing.assert_allclose([v['equity'] for v in old['daily']],[v['equity'] for v in out['daily']],rtol=1e-12,atol=1e-7)
   if not result_path.exists():result_path.write_text(json.dumps(out,allow_nan=False)+'\n')
   summary[key]=prior.metrics(out);summary[key]['diagnostics']=diagnostics(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
   print(key,'CAGR',summary[key]['full']['cagr'],'DEV',summary[key]['development']['cagr'],'RECENT',summary[key]['last252']['return_'],flush=True)
 controls=['baseline','validated_base'];eligible=[k for k in KINDS if k not in controls and summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=summary['validated_base_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 (P/'SELECTION.json').write_text(json.dumps(dict(selected=selected,eligible=eligible,criterion=CONFIG['selection']),indent=2)+'\n')
 assert before==protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),controls_reproduced=True,signals_before_fills=True,
  protected_files_unchanged=True,labels_mature=all(v['latest_label_session']<v['first_prediction'] for v in audit),performance_publishable=False),indent=2)+'\n')
 print('COMPLETE',selected,flush=True)

if __name__=='__main__':main()
