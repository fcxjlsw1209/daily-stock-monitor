"""Seventh research round: disclosed broad search, never a live strategy promotion."""
from pathlib import Path
import sys,json,hashlib,functools
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'daily_entry_timing_research'))
import timing as t
r,h=t.r,t.h
BASE=dict(shock=True,threshold=.75,cap=10,tp=3.,sl=2.,min_age=2,gap=None,rank_floor=.35,score_mode='base',extend=False,protect_winner=False)
VARIANTS={
 'stable_control':dict(shock=False),'shock_control':{},
 'profit2':dict(tp=2.),'profit4':dict(tp=4.),'profit5':dict(tp=5.),
 'cap8':dict(cap=8),'cap12':dict(cap=12),'cap15':dict(cap=15),
 'stop15':dict(sl=1.5),'stop25':dict(sl=2.5),
 'threshold72':dict(threshold=.72),'threshold78':dict(threshold=.78),
 'gap05':dict(gap=.05),'gap075':dict(gap=.075),'gap125':dict(gap=.125),
 'min_age3':dict(min_age=3),'min_age4':dict(min_age=4),
 'rank_floor25':dict(rank_floor=.25),'rank_floor45':dict(rank_floor=.45),
 'extend_winner':dict(extend=True),'protect_winner':dict(protect_winner=True),
 'momentum_rawhalf':dict(score_mode='momentum_rawhalf'),
 'momentum_raw':dict(score_mode='momentum_raw'),
 'momentum_accel':dict(score_mode='momentum_accel'),
 'route_vol50':dict(score_mode='route_vol50'),
 'route_upside':dict(score_mode='route_upside'),
 'shape_high52':dict(score_mode='shape_high52'),
 'geometric_groups':dict(score_mode='geometric_groups')}
CONFIG=dict(start='2019-01-02',end='2026-10-06',cost_bps=[10,30],base=BASE,variants=VARIANTS,
 declared_before_results=True,all_history_previously_inspected=True,broad_search=True,goal_last252_return=1.,
 no_leverage=True,max_positions=3,min_positions=2,initial_capital=100000,group_weights=[.25]*4,
 source_pool='Unchanged available-data historical NDX100 union SP100 past vendor cap Top32 plus17 unlevered ETFs',
 data_certified=False,performance_publishable=False,no_observation_changes=True,
 target_policy='Report ALL target hits as in-sample research observations. Continue whole declared batch after a hit. No recent winner promoted as an out-of-sample discovery.',
 development_selection='Among noncontrols with development DD10>=-35% and development CAGR30>=shock_control30, highest reused2019-2023 CAGR10; no use of recent return in this selection.',
 hypotheses={
 'profit':'Profit targets2,4,5 ATR test whether default3 cuts rebounds too early; exit uses prior close,next open.',
 'cap':'Holding caps8,12,15 sessions test rebound horizon; no monthly-sleeve substitution.',
 'stop':'Stop1.5/2.5ATR relative to default2; not guaranteed intraday stops.',
 'threshold':'Entry scores.72/.78 vs.75 retain trend and pullback eligibility.',
 'gap':'Fixed replacement gaps .05/.075/.125 vs causal monthly gap; replacement still one-to-one.',
 'min_age':'Only paired replacement minimum age changes to3/4; hard exits unchanged.',
 'rank_floor':'Held-score exits.25/.45 vs.35.',
 'extend_winner':'At signal close profit>=entryATR, momentum group>=.8 and current score>=.6 allow cap15 and profit5ATR; otherwise cap10/profit3. Other exits unchanged.',
 'protect_winner':'No optional replacement while prior-close profit>=1entryATR and score>=.6; other exits unchanged.',
 'momentum_rawhalf':'Momentum group half prior risk-adjusted score, half rankR12-1; group remains25%.',
 'momentum_raw':'Momentum group entirely rankR12-1; group remains25%.',
 'momentum_accel':'Momentum group70% prior plus30% rank(log63return - previous63logreturn); group remains25%.',
 'route_vol50':'Path group50% mature path history+50%20day realizedvol rank vs70/30; group remains25%.',
 'route_upside':'Replace only30% realizedvol component within path group with20day upside semideviation rank.',
 'shape_high52':'Shape group75%prior+25%rank(distance to252day high).',
 'geometric_groups':'Equal-weight geometric mean of four groups, including original causally enabled learned rebound group. Threshold unchanged. Penalizes weakness in any group.'},
 model_policy='Frozen original causal predictions; modifications are explicit score overlays, no refit using full sample. Prediction component and activation clock unchanged.',
 references=['https://www.aqr.com/insights/research/working-paper/trading-costs-of-asset-pricing-anomalies','https://www.nber.org/papers/w20660','https://www.nber.org/papers/w8734'],
 source_scope='Research motivations only, not replication or validation of these rules.')

def protected_hashes():
 x=t.protected_hashes()
 for file in [Path(t.__file__),t.P/'FINAL_VALIDATION.json',t.P/'shock_pause2_cost10.json']:
  x[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
 return x

def extras(d,f):
 a=d['adj'];ret=a.pct_change(fill_method=None);rank=lambda x:x.where(f['valid']).rank(axis=1,pct=True)
 volrank=rank(ret.rolling(20).std());path=(f['groups'][2]-.3*volrank)/.7
 shock=(d['opening']/d['close'].shift()-1).lt(-2*ret.rolling(20).std().shift()) & d['volume'].gt(1.5*d['volume'].rolling(20).mean().shift())
 return dict(raw=rank(a.shift(21)/a.shift(252)-1),accel=rank(np.log(a/a.shift(63))-np.log(a.shift(63)/a.shift(126))),
  vol=volrank,path=path,upside=rank(np.sqrt(ret.clip(lower=0).pow(2).rolling(20).mean())),
  high52=rank(a/a.rolling(252).max()),shock_allowed=~shock.rolling(3,min_periods=1).max().astype(bool))

def signals(f,base,x,cfg):
 g=f['groups'];mode=cfg['score_mode'];score=base.copy()
 if mode=='momentum_rawhalf':score+=.125*(x['raw']-g[0])
 elif mode=='momentum_raw':score+=.25*(x['raw']-g[0])
 elif mode=='momentum_accel':score+=.075*(x['accel']-g[0])
 elif mode=='route_vol50':score+=.05*(x['vol']-x['path'])
 elif mode=='route_upside':score+=.075*(x['upside']-x['vol'])
 elif mode=='shape_high52':score+=.0625*(x['high52']-g[3])
 elif mode=='geometric_groups':
  rebound=4*base-g[0]-g[2]-g[3]
  score=np.exp((np.log(g[0].clip(lower=1e-12))+np.log(rebound.clip(lower=1e-12))+np.log(g[2].clip(lower=1e-12))+np.log(g[3].clip(lower=1e-12)))/4)
 elif mode!='base':raise ValueError(mode)
 mask=f['event'] & score.ge(cfg['threshold'])
 if cfg['shock']:mask &= x['shock_allowed']
 return score,mask

def planner(cfg,gaps,momentum):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev]
  gap=cfg['gap'] if cfg['gap'] is not None else float(gaps.loc[prev])
  pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)]
  remaining=set(state['positions']);sells=[];pairs=[]
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];cap,tp=cfg['cap'],cfg['tp'];why=None
   if cfg['extend'] and price>=p['entry_price']+p['atr'] and momentum.at[prev,s]>=.8 and sc[s]>=.6:cap,tp=15,5.
   if age>=cap:why='holding_limit'
   elif price<p['entry_price']-cfg['sl']*p['atr']:why='close_stop_atr'
   elif tp is not None and price>p['entry_price']+tp*p['atr']:why='close_take_profit_atr'
   elif cfg.get('trail_atr') is not None:
    peak=max(p['entry_price'],float(close.loc[pd.Timestamp(p['entry_date']):prev,s].max()))
    if peak>=p['entry_price']+cfg.get('trail_activate',1.)*p['atr'] and price<peak-cfg['trail_atr']*p['atr']:why='close_trailing_atr'
   if why is None and cfg.get('ma_exit') is not None and age>=cfg.get('ma_min_age',2) and price<close.loc[:prev,s].tail(cfg['ma_exit']).mean():why='close_below_ma'
   if why is None and pd.notna(sc[s]) and sc[s]<cfg['rank_floor']:why='rank_deterioration'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={v['symbol'] for v in sells};selected=[]
  def feasible(s,held):return s not in held|sold and issuers[s] not in {issuers[t] for t in held}
  for s,v in pool.items():
   if len(remaining)+len(selected)>=cfg.get('positions',3):break
   if feasible(s,remaining|{v[0] for v in selected}):selected.append((s,v))
  for held in sorted(remaining,key=lambda s:(float(sc[s]) if pd.notna(sc[s]) else 2.,s)):
   p=state['positions'][held];age=i-dates.get_loc(pd.Timestamp(p['entry_date']))
   if age<cfg['min_age'] or pd.isna(sc[held]):continue
   if cfg['protect_winner'] and close.at[prev,held]>=p['entry_price']+p['atr'] and sc[held]>=.6:continue
   others=(remaining-{held})|{v[0] for v in selected}
   challenger=next(((s,v) for s,v in pool.items() if s!=held and s not in state['positions'] and feasible(s,others) and v-sc[held]>gap),None)
   if challenger is None:continue
   s,v=challenger;sells.append(dict(symbol=held,reason='matched_better_candidate'));sold.add(held);remaining.remove(held);selected.append((s,v))
   pairs.append(dict(sell=held,buy=s,gap=float(v-sc[held]),required_gap=gap))
  n=len(remaining)+len(selected)
  if n<cfg.get('min_positions',2) and not (cfg.get('keep_lone',False) and remaining):sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);selected=[];n=0
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,
   buys=[dict(symbol=s,budget=state['equity']/max(n,cfg.get('budget_floor',0)),score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],
   candidate_count=len(pool),equity_basis=state['equity'],replacement_pairs=pairs,gap=gap)
 return plan

def run(cfg,cost,d,f,base,gaps,x,start='2019-01-02'):
 score,mask=signals(f,base,x,cfg);oldplan,oldexec=h.plan,h.execute;ot,os=h.terminal_baskets.apply,h.spin_distributions.apply
 try:
  h.plan=planner(cfg,gaps,f['groups'][0]);h.execute=functools.partial(t.a.old.prior.relaxed_daily,cost_bps=cost)
  h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
  out=h.run_account('baseline',d,start,CONFIG['end'],{'baseline':dict(score=score,mask=mask,atr=f['atr'],ma5=f['ma5'])})
 finally:h.plan,h.execute=oldplan,oldexec;h.terminal_baskets.apply,h.spin_distributions.apply=ot,os
 if out['status']!='complete':raise ValueError(str({k:out.get(k) for k in ['status','error','blocked_session']}))
 assert all(v['signal_date']<v['session'] for v in out['decisions'])
 assert all(v['cash']>=-1e-6 and v['positions']<=cfg.get('positions',3) and v['exposure']<=1+1e-8 for v in out['daily'])
 return out

def summarize(out):
 if len(out['daily'])>=253:return t.a.old.prior.metrics(out)
 rows=out['daily'];start=rows[0]['date'];end=rows[-1]['date']
 ranges=dict(full=(start,end),early=(start,'2018-12-31'),development=('2019-01-01','2023-12-31'),later=('2024-01-01',end),last252=(start,end))
 v={key:r.period_metrics(rows,a,b) for key,(a,b) in ranges.items()};v['ex_recent']=None
 v.update(status='complete',start=start,end=end,ending_equity=out['metrics']['equity'],fees=out['state']['fees'],orders=len(out['orders']),mean_exposure=float(np.mean([row['exposure'] for row in rows])),first_fill=out['orders'][0]['date'] if out['orders'] else None,turnover_notional=sum(o['price']*o['quantity'] for o in out['orders']))
 return v

def save(folder,key,out,summary):
 (folder/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
 v=summarize(out);v['diagnostics']=t.a.old.diagnostics(out);summary[key]=v
 (folder/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(key,'CAGR',None if v['full']['cagr'] is None else round(v['full']['cagr']*100,3),'DD',round(v['full']['max_drawdown']*100,3),'DEV',None if v['development'] is None else round(v['development']['cagr']*100,3),'RECENT',round(v['last252']['return_']*100,3),flush=True)

def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite research')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');before=protected_hashes();(P/'SOURCE_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
 # Direct derived-output copies go to this new study, not the preceding study.
 oldp=t.P;t.P=P
 try:d,f,base,gaps,audit=t.prepare()
 finally:t.P=oldp
 x=extras(d,f);(P/'GAP_CHOICES.json').write_text(json.dumps(audit,indent=2)+'\n');summary={}
 print('PREPARED',len(VARIANTS),'variants',flush=True)
 for name,overlay in VARIANTS.items():
  cfg={**BASE,**overlay}
  for cost in CONFIG['cost_bps']:
   out=run(cfg,cost,d,f,base,gaps,x)
   if name in ['stable_control','shock_control']:
    ref=json.loads((t.P/f'{"stable_control" if name=="stable_control" else "shock_pause2"}_cost{cost}.json').read_text())
    assert out['orders']==ref['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in ref['daily']],rtol=1e-12,atol=1e-7)
   save(P,f'{name}_cost{cost}',out,summary)
 eligible=[k for k in VARIANTS if not k.endswith('control') and summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=summary['shock_control_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 hits=[k for k,v in summary.items() if v['last252']['return_']>=1]
 (P/'SELECTION.json').write_text(json.dumps(dict(development_selected=selected,eligible=eligible,in_sample_target_hits=hits,criterion=CONFIG['development_selection'],target_policy=CONFIG['target_policy']),indent=2)+'\n')
 assert before==protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),controls_reproduced=True,protected_unchanged=True,cached_predictions=True,no_leverage=True,data_certified=False),indent=2)+'\n')
 print('COMPLETE','selected',selected,'hits',hits,flush=True)

if __name__=='__main__':main()
