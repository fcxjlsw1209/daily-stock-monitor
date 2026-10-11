"""Isolated, predeclared ablations of the fixed monthly and daily recipes."""
from pathlib import Path
import sys,json,copy,functools,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent;DUAL=P.parent/'dual_strategy_research'
sys.path.insert(0,str(DUAL))
# Avoid a module-name collision when this script is imported by tests.
import importlib.util
spec=importlib.util.spec_from_file_location('prior_dual',DUAL/'research.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
spec2=importlib.util.spec_from_file_location('monthly_extension',DUAL/'extend_monthly_2010.py')
# Metrics are kept local because extend_monthly_2010 imports research by name.
h=r.h
ENGINE=h.engine('baseline')

MONTHLY=['net','buffer5','industry20','residual20','path20','correlation10','vol20','staged5']
DAILY=['baseline','confirmation','sector_context','cost_margin','replacement25']

def net_execute(name,state,decision,bars,dates,cost_bps=10,sector=None):
 """Cost-aware target weights; proportional average-cost accounting for partial sells."""
 previous=state['equity'];half=cost_bps/20000
 empty=dict(session=decision['session'],sells=[],buys=[])
 # Existing frozen engine applies splits/dividends once, including on non-trade dates.
 state,row,_,_,events=ENGINE.execute(state,empty,bars,dates,cost_bps=cost_bps,sector=None)
 if decision.get('target_weights') is None:return state,row,[],[],events,[]
 weight=dict(decision['target_weights']);allowed={b['symbol'] for b in decision['buys']};cancelled=[]
 for s in list(weight):
  if s not in allowed or not np.isfinite(bars[s]['open']) or bars[s]['open']<=0:
   if s in state['positions']:raise ValueError('Missing execution price for held target '+s)
   cancelled.append(dict(session=decision['session'],symbol=s,reason='target_entry_unavailable'))
   del weight[s] # do not redistribute cancelled-entry budget
 if any(not np.isfinite(w) or w<0 for w in weight.values()) or sum(weight.values())>1+1e-12:raise ValueError('Invalid long-only target weights')
 positions=state['positions'];values={s:p['qty']*bars[s]['open'] for s,p in positions.items()};liquid=state['cash']+sum(values.values())
 allsymbols=sorted(set(values)|set(weight))
 def fees(equity):return half*sum(abs(weight.get(s,0)*equity-values.get(s,0)) for s in allsymbols)
 lo,hi=0.,liquid
 for _ in range(60):
  mid=(lo+hi)/2
  if mid+fees(mid)>liquid:hi=mid
  else:lo=mid
 net=(lo+hi)/2;deltas={s:weight.get(s,0)*net-values.get(s,0) for s in allsymbols};orders=[];trades=[]
 day=decision['session'];tolerance=max(1e-8,liquid*1e-12)
 for s in allsymbols:
  delta=deltas[s]
  if delta>=-tolerance:continue
  p=positions[s];price=float(bars[s]['open']);gross=-delta;qty=min(p['qty'],gross/price);fraction=min(1,qty/p['qty']);fee=qty*price*half;basis=p['cost']*fraction;div=p['dividends']*fraction;entryfee=p['entry_fee']*fraction;pnl=qty*price-fee+div-basis
  trades.append(dict(symbol=s,entry_date=p['entry_date'],exit_date=day,entry_price=p['entry_price'],exit_price=price,quantity=qty,entry_outlay=basis,dividends=div,entry_fee=entryfee,exit_fee=fee,net_pnl=pnl,net_return=pnl/basis,sessions=dates.get_loc(pd.Timestamp(day))-dates.get_loc(pd.Timestamp(p['entry_date'])),reason='net_target_rebalance',partial=bool(fraction<1-1e-10)))
  state['cash']+=qty*price-fee;state['fees']+=fee;orders.append(dict(date=day,symbol=s,side='sell',price=price,quantity=qty,fee=fee))
  if fraction>=1-1e-10:del positions[s]
  else:
   p['qty']-=qty;p['cost']-=basis;p['dividends']-=div;p['entry_fee']-=entryfee
 attributes={b['symbol']:b for b in decision['buys']}
 for s in allsymbols:
  delta=deltas[s]
  if delta<=tolerance:continue
  price=float(bars[s]['open']);qty=delta/price;fee=delta*half;spend=delta+fee
  if spend>state['cash']+max(1e-6,liquid*1e-10):raise ValueError('Target overspends cash')
  atr=attributes[s]['atr']/bars[s]['split']
  if s in positions:
   p=positions[s];total=p['qty']+qty;p['entry_price']=(p['entry_price']*p['qty']+price*qty)/total;p['atr']=(p['atr']*p['qty']+atr*qty)/total;p['qty']=total;p['cost']+=spend;p['entry_fee']+=fee
  else:positions[s]=dict(entry_date=day,entry_price=price,qty=qty,cost=spend,entry_fee=fee,dividends=0.,atr=atr)
  state['cash']-=spend;state['fees']+=fee;orders.append(dict(date=day,symbol=s,side='buy',price=price,quantity=qty,fee=fee))
 if abs(state['cash'])<1e-7:state['cash']=0.
 invested=sum(p['qty']*bars[s]['close'] for s,p in positions.items());state['equity']=state['cash']+invested;state['last_date']=day
 assert state['cash']>=-1e-6 and len(positions)<=3
 if sector:
  assert all(sum(sector[t]==sector[s] for t in positions)<=2 for s in positions)
 row=dict(date=day,equity=state['equity'],cash=state['cash'],invested=invested,exposure=invested/state['equity'],positions=len(positions),fees_cumulative=state['fees'],dividends_cumulative=state['dividends'],**{'return':state['equity']/previous-1})
 return state,row,trades,orders,events,cancelled

def added_features(d,union_eligible,mf):
 a=d['adj'];ret=a.pct_change(fill_method=None);valid=mf['valid'];rank=lambda x:x.where(valid).rank(axis=1,pct=True)
 stock=[s for s in a if not d['issuers'][s].startswith('FUND_')];past63=a.shift(63);rise=a.pct_change(63,fill_method=None).gt(0)&a.gt(a.rolling(50).mean());breadth=pd.DataFrame(np.nan,index=a.index,columns=a.columns)
 for day in a.index:
  e=union_eligible.loc[day,stock]&a.loc[day,stock].notna()&past63.loc[day,stock].notna()
  members=[s for s in stock if e[s] and pd.notna(d['sectors'].at[day,s])]
  if not members:continue
  by=pd.DataFrame({'sector':d['sectors'].loc[day,members],'up':rise.loc[day,members].astype(float)}).groupby('sector').up.mean()
  # ETF/underspecified groups use contemporaneous overall union breadth explicitly.
  breadth.loc[day]=d['sectors'].loc[day].map(by).fillna(float(rise.loc[day,members].mean()))
 market=ret['SPY'];beta=ret.rolling(126).cov(market).div(market.rolling(126).var(),axis=0).shift(1)
 resid=ret-beta.mul(market,axis=0);residual_momentum=resid.shift(42).rolling(210).mean()/resid.shift(42).rolling(210).std().clip(lower=1e-6)
 lr=np.log(a).diff();path=lr.shift(42).rolling(210).sum()/lr.abs().shift(42).rolling(210).sum()
 return dict(ret=ret,breadth=breadth,industry_score=.8*mf['base']+.1*rank(d['sector60'])+.1*rank(breadth),residual_score=.8*mf['base']+.2*rank(residual_momentum),path_score=.8*mf['base']+.2*rank(path))

def monthly_plan(kind,ends,ret,event,buffer_limit=5):
 pending=None
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  nonlocal pending
  i=dates.get_loc(day);prev=dates[i-1];decision=dict(session=str(day.date()),signal_date=str(prev.date()),sells=[],buys=[],candidate_count=0,equity_basis=state['equity'])
  chosen=[];exposure=1.;reason='month_end'
  if day not in ends:
   if kind!='staged5' or pending is None:return decision
   elapsed=i-pending['index'];trigger=sum(bool(event.at[prev,s]) for s in pending['symbols'])>=2
   if elapsed<1 or (elapsed<5 and not trigger):return decision
   chosen=[s for s in pending['symbols'] if s in reps and bool(mask.at[prev,s])];reason='pullback' if trigger and elapsed<5 else 'five_session_deadline';pending=None
   if len(chosen)<2:return decision
  else:
   pool=score.loc[prev].where(mask.loc[prev]).dropna();pool=pool[pool.index.isin(reps)].sort_values(ascending=False,kind='stable');decision['candidate_count']=len(pool)
   history=ret.loc[:prev].tail(63);corr=history.corr(min_periods=50) if kind=='correlation10' else None
   candidates=list(pool.index)
   if kind=='buffer5':candidates=[s for s in pool.index[:buffer_limit] if s in state['positions']]+[s for s in pool.index[:3] if s not in state['positions']]
   while candidates and len(chosen)<3:
    feasible=[s for s in candidates if s not in chosen and issuers[s] not in {issuers[t] for t in chosen} and sum(sectors[t]==sectors[s] for t in chosen)<2]
    if not feasible:break
    if kind=='correlation10' and chosen:
     def adjusted(s):
      correlations=[corr.at[s,t] for t in chosen];penalty=max([max(0,float(x)) if pd.notna(x) else 1 for x in correlations]);return float(pool[s])-.1*penalty
     s=max(feasible,key=adjusted)
    else:s=feasible[0]
    chosen.append(s);candidates.remove(s)
   if len(chosen)<2:chosen=[]
   if kind=='vol20' and chosen:
    sigma=float(history[chosen].mean(axis=1).std()*np.sqrt(252));exposure=min(1.,.20/sigma) if np.isfinite(sigma) and sigma>0 else 0.
   if kind=='staged5' and chosen:exposure=.5;pending=dict(index=i,symbols=list(chosen))
  if len(chosen)<2:chosen=[]
  decision['target_weights']={s:exposure/len(chosen) for s in chosen};decision['rebalance_reason']=reason
  decision['sells']=[dict(symbol=s,reason='net_target_rebalance') for s in state['positions'] if s not in chosen]
  decision['buys']=[dict(symbol=s,budget=state['equity']*exposure/len(chosen),score=float(score.at[prev,s]),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s in chosen]
  return decision
 return plan

def daily_plan(gap):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev];pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)];sells=[];remaining=set(state['positions'])
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];why=None
   if age>=10:why='holding_limit'
   elif price<p['entry_price']-2*p['atr']:why='close_stop_atr'
   elif price>p['entry_price']+3*p['atr']:why='close_take_profit_atr'
   elif pd.notna(sc[s]) and sc[s]<.35:why='rank_deterioration'
   elif age>=2 and len(pool) and pool.iloc[0]-sc[s]>gap:why='better_candidate'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={r['symbol'] for r in sells};selected=[]
  for s,value in pool.items():
   if len(remaining)+len(selected)>=3:break
   held=remaining|{v[0] for v in selected}
   if s in held|sold or issuers[s] in {issuers[t] for t in held}:continue
   selected.append((s,value))
  n=len(remaining)+len(selected)
  if n<2:
   sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);selected=[];n=0
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,buys=[dict(symbol=s,budget=state['equity']/n,score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],equity_basis=state['equity'],candidate_count=len(pool))
 return plan

def relaxed_daily(name,state,decision,bars,dates,cost_bps=10,sector=None):
 return r.ORIGINAL_EXEC(name,state,decision,bars,dates,cost_bps=cost_bps,sector=None)

def metrics(result):
 if result['status']!='complete':return {k:result.get(k) for k in ['status','error','blocked_session']}
 rows=result['daily'];end=rows[-1]['date'];start=rows[0]['date'];last252=rows[-252]['date'];cut=rows[-253]['date']
 ranges=dict(full=(start,end),early=(start,'2018-12-31'),development=('2019-01-01','2023-12-31'),later=('2024-01-01',end),ex_recent=(start,cut),last252=(last252,end))
 x={k:r.period_metrics(rows,a,b) for k,(a,b) in ranges.items()}
 x.update(status='complete',start=start,end=end,ending_equity=result['metrics']['equity'],fees=result['state']['fees'],orders=len(result['orders']),mean_exposure=float(np.mean([v['exposure'] for v in rows])),first_fill=result['orders'][0]['date'] if result['orders'] else None,turnover_notional=sum(o['price']*o['quantity'] for o in result['orders']))
 return x

def main():
 config=dict(monthly=MONTHLY,daily=DAILY,monthly_start='2010-01-04',daily_start='2019-01-02',end='2026-10-06',cost_bps=[10,30],no_selection_or_retuning=True,strict_history_certified=False,original_accounts_modified=False,parameters=dict(buffer='Hold eligible positive-momentum positions while in top5; new names only top3; sectorcap2',industry='80%base+10%industry63returnrank+10%breadthrank; breadth among contemporaneous fullindexunion with63daygain and aboveSMA50; unmapped fund sectors use overall unionbreadth',residual='80%base+20%market-only residual momentum rank; dailySPY beta126lag1, mean/std of lag42trailing210 residuals; no industry-neutral claim',path='80%base+20%lag42trailing210 logreturn sum / sum absolute logreturns rank',correlation='Greedy score minus0.10 max positive trailing63d correlation to already selected',risk='Monthend63day equalweight portfolio voltarget20%annualized; cap100%; cash earns0',hybrid='Frozen monthly3names:50%atmonthendclose, remainder first subsequent open after at least2targets have prior-close pullback signals, or fifth trading-session open; if fewer2remaineligible do not add; all net rebalance then hold',daily='same Ridge forecasts as previouslyselected; confirmation=prior1or2sessions event plus latest upday/closeupperhalf; sector_context=positive sector60,breadth>.5; cost_margin=predictednet5day>.003; replacement25=scoregap.25 instead.15'),execution='Monthly previous-session signal,next monthendclose; daily previousclose,nextopen; hybrid deferred tranche nextopen; netting solves afterfee target equity; proportional cost basis on trims makes fragment winrates not comparable')
 (P/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 frozen=[DUAL/'research.py',DUAL/'monthly_2010/cost10.json',DUAL/'x_d_ridge_75_gateFalse.json',r.U/'alias_resolved_inputs/MANIFEST.json']
 hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen}
 (P/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 m,d=r.prepare();union=d['eligible'].copy();d=r.apply(m,d,P/'pool');df,mf=r.features(d);added=added_features(d,union,mf);print('Features prepared',flush=True)
 dates=d['close'].index;ends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);md=dict(d);md['opening']=d['close'];hybrid=dict(d);hybrid['opening']=d['opening'].copy();hybrid['opening'].loc[sorted(ends)]=d['close'].loc[sorted(ends)]
 summary={};checks={};oldbaseline=json.loads((DUAL/'monthly_2010/cost10.json').read_text());summary['monthly_original_cost10']=metrics(oldbaseline)
 def perform(key,data,ff,plan,execute,start,cost):
  h.plan=plan;h.execute=functools.partial(execute,cost_bps=cost);ot=h.terminal_baskets.apply;os=h.spin_distributions.apply
  try:
   h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
   output=h.run_account('baseline',data,start,m['end'],{'baseline':ff})
  finally:h.terminal_baskets.apply=ot;h.spin_distributions.apply=os
  (P/(key+'.json')).write_text(json.dumps(output,allow_nan=False)+'\n');summary[key]=metrics(output);(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
  if output['status']=='complete':
   assert all(x['signal_date']<x['session'] for x in output['decisions']);assert all(x['cash']>=-1e-6 for x in output['daily'])
   assert all(len({o['side'] for o in output['orders'] if o['date']==day and o['symbol']==s})<=1 for day,s in {(o['date'],o['symbol']) for o in output['orders']}) if key.startswith('monthly_') else True
  print(key,json.dumps(summary[key]),flush=True);return output
 for cost in config['cost_bps']:
  if cost==30:summary['monthly_original_cost30']=metrics(json.loads((DUAL/'monthly_2010/cost30.json').read_text()))
  for kind in MONTHLY:
   sc=added.get({'industry20':'industry_score','residual20':'residual_score','path20':'path_score'}.get(kind,''),mf['base']);ff=dict(score=sc,mask=mf['valid']&mf['positive'],atr=mf['atr'],ma5=mf['ma5']);out=perform(f'monthly_{kind}_cost{cost}',hybrid if kind=='staged5' else md,ff,monthly_plan(kind,ends,added['ret'],df['event']),net_execute,'2010-01-04',cost)
   if kind=='net' and cost==10 and out['status']=='complete':
    a=[[b['symbol'] for b in v['buys']] for v in out['decisions']];b=[[b['symbol'] for b in v['buys']] for v in oldbaseline['decisions']];assert a==b;checks['net_same_monthly_targets']=True
 dp,_,audit=r.model(df,6);(P/'TRAINING_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');g=df['groups'];sc=(g[0]+.5*g[1]+.5*dp.where(df['valid']).rank(axis=1,pct=True)+g[2]+g[3])/4
 loc=((d['close']-d['low'])/(d['high']-d['low'])).fillna(.5);a=d['adj'];trend=(a.shift(21)/a.shift(252)>1)&(a>a.rolling(100).mean())
 for cost in config['cost_bps']:
  for kind in DAILY:
   gate=df['event']
   if kind=='confirmation':gate=(df['event'].shift(1,fill_value=False)|df['event'].shift(2,fill_value=False))&df['valid']&trend&a.gt(a.shift())&loc.gt(.5)
   if kind=='sector_context':gate=gate&d['sector60'].gt(0)&added['breadth'].gt(.5)
   if kind=='cost_margin':gate=gate&dp.gt(.003)
   ff=dict(score=sc,mask=gate&sc.ge(.75),atr=df['atr'],ma5=df['ma5']);out=perform(f'daily_{kind}_cost{cost}',d,ff,daily_plan(.25 if kind=='replacement25' else .15),relaxed_daily,'2019-01-02',cost)
   if kind=='baseline' and cost==10:
    prior=json.loads((DUAL/'x_d_ridge_75_gateFalse.json').read_text());assert out['status']=='complete';np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in prior['daily']],rtol=1e-12,atol=1e-7);assert out['orders']==prior['orders'];checks['daily_baseline_reproduced']=True
 for path,before in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==before
 (P/'CHECKS.json').write_text(json.dumps(dict(checks=checks,frozen_sources_unchanged=True,training_batches=len(audit),all_labels_mature=all(v['latest_label_session']<v['first_prediction'] for v in audit),all_runs_complete=all(v['status']=='complete' for v in summary.values())),indent=2)+'\n');print('DONE',flush=True)
if __name__=='__main__':main()
