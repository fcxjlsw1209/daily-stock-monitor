"""Independent reconstruction from four concepts; no production writes.

Four equal prediction experts, two distinct entry/continuation training populations.
Cached public data are inherited research inputs, not certified point-in-time data.
"""
from pathlib import Path
import sys,json,hashlib,copy,functools
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'dual_strategy_research'))
import research as r
import vendor_pool as vp
from prepare import ETFS
from factors import build
START='2016-08-17';END='2026-09-24'
CASES={
 'rules10':dict(mode='rules'),
 'experts10':dict(mode='experts'),
 'experts_tree10':dict(mode='tree'),
 'experts_risk10':dict(mode='risk'),
 'experts5':dict(mode='experts',cap=5),
 'experts20':dict(mode='experts',cap=20),
 'experts_no_profit':dict(mode='experts',tp=None),
 'experts_no_replace':dict(mode='experts',replace=False),
 'experts_continue_exit':dict(mode='experts',negative_exit=True),
 'experts_wide_gate':dict(mode='experts',strong_rank=0.),
}
BASE=dict(mode='experts',cap=10,tp=3.,sl=2.,replace=True,negative_exit=False,strong_rank=.5)
CONFIG=dict(start=START,end=END,created_before_results=True,cases=CASES,base=BASE,
 costs_bps=[0,10,30],timing=['next_open','same_close_diagnostic'],
 data_scope='Historical NDX100 union SP100 available-data issuers, top32 past cap snapshots price-scaled forward;17 predefined unlevered ETFs. Not author exact universe.',
 cap_proxy='Latest observation available one calendar day later, max45days. Scale cap by split-adjusted current close / latest close ON OR BEFORE snapshot date; no future interpolation. Assumes share-count unchanged between snapshots; company/class and historical vintage limitations remain.',
 factors='Four groups each25%: risk-adjusted annual momentum; mature similar-event rebound; mature rebound path and volatility space; present pullback depth/location/volume. Four features per group, separate Ridge expert each, equal prediction average.',
 model='Monthly retrain trailing1008sessions. Five-session next-open return target; six-session maturation, no target overlap into first prediction. Each training date equal total weight; half-life504sessions; standardization fit on training only. Ridge alpha100. Separate event-entry and strong-trend continuation populations. Residual stump15% optional; normalized-return target optional.',
 history='Similar events partition by z3<-1 or milder and close location<.2 or other. Own-symbol trailing756 mature outcomes shrink to corresponding historical pool mean with20pseudoobservations. Missing history stays missing.',
 gate='Positive R12-1, close>SMA100, annual riskadjusted momentum rank>=.5 (explicitwidecase0); z3<-.35 OR location<.2; z3>-3.5; entry experts predicted5day grossreturn>.001. Rules score>=.6.',
 management='Max3,min2 else cash. Fixed existing shares between exits. Cap10sessions,close stop2entryATR,profit3ATR; optional replacements after2days only if a distinct new event expectedreturn exceeds held continuation prediction by .002. No obsolete pullback requirement for holders. Sectorcap3; duplicate issuers forbidden. New budget priorNAV/targetN cash-capped.',
 chronology='next_open uses previous completed session; same_close_diagnostic uses finalized same-day bars and then that close, deliberately optimistic non-executable diagnostic without intraday signal/MOC evidence.',
 selection='Report everycase. Development shortlist only2016-2023 CAGR at30bps,DD>=-40%; recent and full best explicitly retrospective, not independent OOS.',
 data_certified=False,independent_OOS=False,no_live_changes=True,no_leverage=True,
 source_references=['https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'])

def dump(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def protected():
 files=[]
 for directory in ['daily_stock_monitor','pressure_shadow_monitor','resilience_shadow_monitor','accel_shadow_monitor','high52_shadow_monitor','accel40_shadow_monitor','monthly_shadow_monitor']:
  d=P.parent/directory
  if d.exists():files.extend(p for p in d.rglob('*') if p.is_file() and p.suffix in ['.py','.sqlite','.json','.md'])
 files.extend([Path(r.__file__),Path(vp.__file__),P.parent/'daily_growth_research/REPORT.txt',P.parent/'daily_frontier_research/REPORT.txt'])
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def scale_cap(series,observed,close):
 """As-of denominator can only be a price at/before the source observation."""
 prices=pd.DataFrame({'price_date':close.index,'price':close.to_numpy()}).dropna()
 q=pd.DataFrame({'row':np.arange(len(series)),'observation':pd.to_datetime(observed.to_numpy())}).dropna().sort_values('observation')
 joined=pd.merge_asof(q,prices,left_on='observation',right_on='price_date',direction='backward')
 den=pd.Series(np.nan,index=series.index)
 good=(joined.observation-joined.price_date).dt.days.le(7)&joined.price.gt(0)
 den.iloc[joined.loc[good,'row'].to_numpy()]=joined.loc[good,'price'].to_numpy()
 return series*close/den

def prepare():
 # Divert prepare's derived ETF manifest into this study; raw caches never change.
 import prepare as pp
 old=pp.P;pp.P=P
 try:m,d=pp.prepare()
 finally:pp.P=old
 # ETF fallback paths are normally in original study: see seed_sources() before call.
 original=d['eligible'].copy();d=vp.apply(m,d,P/'snapshot_pool')
 caps=pd.read_csv(P/'snapshot_pool/cap_snapshots_aligned.csv.gz',index_col=0,parse_dates=True)
 obs=pd.read_csv(P/'snapshot_pool/observation_dates.csv.gz',index_col=0,parse_dates=True,low_memory=False)
 scaled=caps.copy();stocks=[s for s in caps if s not in ETFS]
 for s in stocks:scaled[s]=scale_cap(caps[s],obs[s],d['close'][s])
 selected=pd.DataFrame(False,index=caps.index,columns=caps.columns)
 coverage=[]
 for day in caps.index:
  ranking=scaled.loc[day,stocks].where(original.loc[day,stocks]).dropna().sort_values(ascending=False,kind='stable');seen=set();chosen=[]
  for s in ranking.index:
   if d['issuers'][s] in seen:continue
   seen.add(d['issuers'][s]);chosen.append(s)
   if len(chosen)==32:break
  if len(chosen)==32:selected.loc[day,chosen]=True
  coverage.append(dict(date=str(day.date()),selected=len(chosen),covered=len(ranking),eligible=int(original.loc[day,stocks].sum())))
 selected[ETFS]=original[ETFS]
 old_members=d['eligible'].copy();d['eligible']=selected
 selected.to_csv(P/'price_scaled_membership.csv.gz');scaled.to_csv(P/'price_scaled_caps.csv.gz');pd.DataFrame(coverage).to_csv(P/'COVERAGE.csv',index=False)
 changed=(old_members!=selected).loc[START:END,stocks].sum(axis=1)/2
 dump(P/'POOL_DIFFERENCES.json',dict(days_changed=int(changed.gt(0).sum()),mean_stock_swaps_per_day=float(changed.mean()),max_swaps=float(changed.max()),exact_PIT_certified=False))
 if (P/'PFE_REPAIR.json').exists():
  from repair_actions import patch
  d=patch(d)
 return m,d,old_members

def seed_sources():
 # Independent immutable copies allow existing prepare() to find all cached ETFs;
 # no network fetch, no re-download and no mutations in the source study.
 import shutil
 (P/'sources').mkdir(exist_ok=True)
 for s in ETFS:
  f=P.parent/'four_factor_rebound/sources'/f'{s}.csv'
  if f.exists():
   target=P/'sources'/f'{s}.csv'
   if not target.exists():shutil.copyfile(f,target)
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(f.read_bytes()).digest()

def conditional_history(values,event,bucket,delay=6,window=756,prior=20):
 """Outcome data become features only after the last required bar closes."""
 result=pd.DataFrame(np.nan,index=values.index,columns=values.columns)
 for b in range(4):
  mask=event&bucket.eq(b)&values.notna();n=mask.astype(float).shift(delay).fillna(0);v=values.where(mask,0).shift(delay).fillna(0)
  counts=n.rolling(window,min_periods=1).sum();sums=v.rolling(window,min_periods=1).sum()
  pooled=sums.sum(axis=1)/counts.sum(axis=1).replace(0,np.nan)
  smoothed=sums.add(prior*pooled,axis=0)/(counts+prior)
  result=result.where(~bucket.eq(b),smoothed)
 return result

def features(d,outcome_clock="next_open"):
 a,c=d['adj'],d['close'];ret=a.pct_change(fill_method=None);vol=ret.rolling(20).std();v63=ret.rolling(63).std();low,high=d['low'],d['high'];volume=d['volume']
 base=build(d);valid=base['valid'];annual=a.shift(21)/a.shift(252)-1;six=a.shift(21)/a.shift(126)-1
 loc=((c-low)/(high-low).replace(0,np.nan)).fillna(.5);z=a.pct_change(3,fill_method=None)/(vol*np.sqrt(3));depth=(a/a.rolling(20).max()-1)/vol
 strong=valid&annual.gt(0)&a.gt(a.rolling(100).mean());event=strong&((z<-.35)|(loc<.2))&z.gt(-3.5)
 bucket=(z.lt(-1).astype(int)*2+loc.lt(.2).astype(int))
 ao=d['opening']*a/c;ah=high*a/c;al=low*a/c
 y=ao.shift(-6)/ao.shift(-1)-1
 y3=ao.shift(-4)/ao.shift(-1)-1
 # Full five future bars must all exist; no skip-NaN path summaries.
 hs=np.stack([ah.shift(-i).to_numpy() for i in range(1,6)]);ls=np.stack([al.shift(-i).to_numpy() for i in range(1,6)])
 mfe=pd.DataFrame(np.max(hs,axis=0),index=a.index,columns=a.columns)/ao.shift(-1)-1
 mae=pd.DataFrame(np.min(ls,axis=0),index=a.index,columns=a.columns)/ao.shift(-1)-1
 if outcome_clock=='same_close_diagnostic':
  y=a.shift(-5)/a-1;y3=a.shift(-3)/a-1
  mfe=pd.DataFrame(np.max(hs,axis=0),index=a.index,columns=a.columns)/a-1
  mae=pd.DataFrame(np.min(ls,axis=0),index=a.index,columns=a.columns)/a-1
 elif outcome_clock!='next_open':raise ValueError(outcome_clock)
 hist=lambda x:conditional_history(x,event,bucket)
 histmean=hist(y);hit=hist(y.gt(0).astype(float).where(y.notna()));hist3=hist(y3);excess=hist(y.sub(y['SPY'],axis=0))
 histpath=hist(mfe+mae);histup=hist(mfe)
 cmf=(((2*c-high-low)/(high-low).replace(0,np.nan)).fillna(0)*volume).rolling(21).sum()/volume.rolling(21).sum()
 rank=lambda x:x.where(valid).rank(axis=1,pct=True)
 groups=[
  [rank(annual/v63),rank(six/v63),rank(annual),rank(a.pct_change(63,fill_method=None).sub(a['SPY'].pct_change(63,fill_method=None),axis=0)/v63)],
  [rank(histmean),rank(hit),rank(hist3),rank(excess)],
  [rank(histpath),rank(histup),rank(vol),rank(base['atr']/c)],
  [rank(-abs(z+1.25)),rank(-loc),rank(-abs(depth+2)),rank(cmf)],
 ]
 groupmean=[sum(g)/4 for g in groups];rawscore=sum(groupmean)/4
 spy=a['SPY'];regime=(spy/spy.rolling(200).mean()-1).clip(-.4,.4)
 # Shared context interactions confined inside each expert; no new fifth budget.
 X=[]
 for g in groups:
  block=g+[v.mul(regime,axis=0) for v in g]
  X.append(np.stack([v.to_numpy() for v in block],axis=2))
 return dict(valid=valid,strong=strong,event=event,y=y,groups=groups,groupmean=groupmean,rule=rawscore,X=X,vol=vol,atr=base['atr'],ma5=base['ma5'],annual=annual,histmean=histmean,momentum=rank(annual/v63))

def ridge_fit_predict(X,y,train,idx,normalize=None,first='2016-08-01'):
 """Each date has equal total weight; all features standardized on past train only."""
 pred=np.full(y.shape,np.nan);tree=pred.copy();audit=[]
 for month in idx[idx>=first].to_period('M').unique():
  target=np.flatnonzero(idx.to_period('M')==month);t=target[0];end=t-6;start=max(0,end-1008)
  ok=train[start:end]&np.isfinite(y[start:end])
  for x in X:ok &= np.isfinite(x[start:end]).all(2)
  if ok.sum()<300:continue
  rows,cols=np.where(ok);time=rows+start;yy=y[time,cols].copy()
  if normalize is not None:yy/=normalize[time,cols]*np.sqrt(5)
  yy=np.clip(yy,-5,5) if normalize is not None else np.clip(yy,-.5,.5)
  counts=np.bincount(rows,minlength=end-start);w=np.exp2(-(end-1-time)/504)/counts[rows];w*=len(w)/w.sum()
  ym=np.average(yy,weights=w);fit=[];pp=[];trainpred=[];exposure=[]
  for x in X:
   xx=x[time,cols];mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1;z=(xx-mu)/sd
   coef=np.linalg.solve(np.einsum("ni,nj,n->ij",z,z,w)+100*np.eye(z.shape[1]),np.einsum("ni,n,n->i",z,w,yy-ym))
   block=x[target];finite=np.isfinite(block).all(2);out=np.full(finite.shape,np.nan);out[finite]=np.einsum("ni,i->n",(block[finite]-mu)/sd,coef)+ym
   assert np.isfinite(coef).all() and np.isfinite(out[finite]).all()
   pp.append(out);trainpred.append(np.einsum("ni,i->n",z,coef)+ym);exposure.append(float(np.linalg.norm(coef)))
  p=sum(pp)/4;residual=yy-sum(trainpred)/4;best=None
  # Single split fitted solely on training residuals; small correction only.
  stumpX=np.concatenate([x[time,cols,:4] for x in X],axis=1)
  for j in range(stumpX.shape[1]):
   for cut in [.25,.5,.75]:
    left=stumpX[:,j]<=cut
    if min(left.sum(),(~left).sum())<50:continue
    l=np.average(residual[left],weights=w[left]);rr=np.average(residual[~left],weights=w[~left]);err=np.average((residual-np.where(left,l,rr))**2,weights=w)
    if best is None or err<best[0]:best=(err,j,cut,l,rr)
  pt=p.copy()
  if best:
   _,j,cut,l,rr=best;tx=np.concatenate([x[target,:,:4] for x in X],axis=2);pt+=.15*np.where(tx[:,:,j]<=cut,l,rr)
  if normalize is not None:p*=normalize[target]*np.sqrt(5);pt*=normalize[target]*np.sqrt(5)
  pred[target]=p;tree[target]=pt
  audit.append(dict(month=str(month),first_prediction=str(idx[t].date()),last_train_signal=str(idx[time.max()].date()),latest_label_session=str(idx[time.max()+6].date()),samples=int(len(yy)),expert_weights=[.25]*4,coefficient_norm=exposure))
 return pred,tree,audit

def train(f):
 idx=f['valid'].index;columns=f['valid'].columns;models={};audit={}
 for population,mask in [('entry',f['event']),('hold',f['strong'])]:
  for mode in ['experts','risk']:
   p,t,a=ridge_fit_predict(f['X'],f['y'].to_numpy(),mask.to_numpy(),idx,f['vol'].to_numpy() if mode=='risk' else None)
   models[population+'_'+mode]=pd.DataFrame(p,index=idx,columns=columns)
   if mode=='experts':models[population+'_tree']=pd.DataFrame(t,index=idx,columns=columns)
   audit[population+'_'+mode]=a
   print('trained',population,mode,len(a),flush=True)
 dump(P/'TRAINING_AUDIT.json',audit)
 np.savez_compressed(P/'PREDICTIONS.npz',**{k:v.to_numpy() for k,v in models.items()})
 return models

def planner(cfg,f,models,clock):
 mode=cfg['mode'];entry=f['rule'] if mode=='rules' else models['entry_'+mode];hold=f['momentum'] if mode=='rules' else models['hold_'+mode]
 event=f['event']&f['momentum'].ge(cfg['strong_rank'])
 event&=entry.ge(.6) if mode=='rules' else entry.gt(.001)
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);when=day if clock=='same_close_diagnostic' else dates[i-1]
  values=entry.loc[when];pool=values.where(event.loc[when]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)]
  remaining=set(state['positions']);sells=[];pairs=[]
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[when,s];why=None
   if age>=cfg['cap']:why='holding_limit'
   elif price<p['entry_price']-cfg['sl']*p['atr']:why='close_stop_atr'
   elif cfg['tp'] is not None and price>p['entry_price']+cfg['tp']*p['atr']:why='close_take_profit_atr'
   elif f['annual'].at[when,s]<=0:why='annual_momentum_lost'
   elif cfg['negative_exit'] and mode!='rules' and age>=2 and hold.at[when,s]<=0:why='negative_continuation_prediction'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={x['symbol'] for x in sells};selected=[]
  def feasible(s,held):return s not in held|sold and issuers[s] not in {issuers[t] for t in held}
  for s,v in pool.items():
   if len(remaining)+len(selected)>=3:break
   if feasible(s,remaining|{x[0] for x in selected}):selected.append((s,v))
  if cfg['replace'] and mode!='rules':
   for old in sorted(remaining):
    if i-dates.get_loc(pd.Timestamp(state['positions'][old]['entry_date']))<2:continue
    expected=hold.at[when,old]
    if pd.isna(expected):continue
    held=(remaining-{old})|{x[0] for x in selected}
    candidate=next(((s,v) for s,v in pool.items() if s not in state['positions'] and feasible(s,held) and v-expected>.002),None)
    if candidate:
     s,v=candidate;sells.append(dict(symbol=old,reason='new_event_beats_continuation'));remaining.remove(old);sold.add(old);selected.append((s,v));pairs.append(dict(sell=old,buy=s,expected_gap=float(v-expected)))
  n=len(remaining)+len(selected)
  if n<2:sells.extend(dict(symbol=s,reason='fewer_than_two_targets') for s in remaining);selected=[];n=0
  return dict(session=str(day.date()),signal_date=str(when.date()),sells=sells,buys=[dict(symbol=s,budget=state['equity']/n,score=float(v),atr=float(atr.at[when,s]),reference_close=float(close.at[when,s])) for s,v in selected],candidate_count=len(pool),equity_basis=state['equity'],replacement_pairs=pairs,clock=clock)
 return plan,dict(score=entry,mask=event,atr=f['atr'],ma5=f['ma5'])

def metrics(out):
 rows=out['daily'];eq=np.array([100000.]+[x['equity'] for x in rows]);ret=eq[1:]/eq[:-1]-1
 x=pd.DataFrame(rows)
 return dict(full=r.period_metrics(rows,rows[0]['date'],rows[-1]['date']),development=r.period_metrics(rows,START,'2023-12-31'),later=r.period_metrics(rows,'2024-01-01',END),last252_return=float(eq[-1]/eq[-253]-1),sharpe=float(np.mean(ret)/np.std(ret,ddof=1)*np.sqrt(252)),final_multiple=float(eq[-1]/100000),closed_trades=len(out['trades']),win_rate=float(np.mean([t['net_pnl']>0 for t in out['trades']])) if out['trades'] else None,mean_exposure=float(x.exposure.mean()),mean_hold=float(np.mean([t['sessions'] for t in out['trades']])) if out['trades'] else None,fees=float(out['state']['fees']))

def save(name,out,summary):
 dump(P/'runs'/f'{name}.json',out)
 if out['status']!='complete':raise ValueError({k:out.get(k) for k in ['status','error','blocked_session']})
 summary[name]=metrics(out);dump(P/'SUMMARY.json',summary)
 v=summary[name];print(name,'CAGR',round(100*v['full']['cagr'],2),'DD',round(100*v['full']['max_drawdown'],2),'recent',round(100*v['last252_return'],2),flush=True)

def main():
 from clock_execution import run_account
 if (P/'SUMMARY.json').exists():raise RuntimeError('Research already has results, no overwrite')
 dump(P/'CONFIG.json',CONFIG);before=protected();dump(P/'SOURCE_HASHES.json',before);seed_sources();(P/'runs').mkdir(exist_ok=True)
 m,d,old_members=prepare();print('data ready',flush=True);f=features(d);print('fresh features ready',flush=True);models=train(f);summary={}
 for name,overlay in CASES.items():
  cfg={**BASE,**overlay}
  for clock,cost in [('next_open',0),('next_open',10),('next_open',30),('same_close_diagnostic',0)]:
   pl,ff=planner(cfg,f,models,clock);out=run_account(d,ff,pl,START,END,cost,clock);save(f'{name}_{clock}_cost{cost}',out,summary)
 dump(P/'CHECKS.json',dict(complete=True,runs=len(summary),protected_unchanged=before==protected(),data_certified=False))
 assert before==protected()
 print('COMPLETE',len(summary),flush=True)
if __name__=='__main__':main()
