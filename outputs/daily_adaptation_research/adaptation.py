"""Fifth finite daily study: causal adaptation and entry opportunities, not a live monitor."""
from pathlib import Path
import sys,json,hashlib,functools,copy
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'daily_execution_research'))
import run_execution as old
r=old.r;h=old.h
KINDS=['validated_control','fixed15','fixed10','dispersion_gap','age_decay','walk_gap',
       'momentum_exception','entry_limit','model_decay252','model_window504','model_ensemble','history_decay252']
CONFIG=dict(start='2019-01-02',end='2026-10-06',variants=KINDS,cost_bps=[10,30],declared_before_results=True,
 four_groups_equal=True,base='Prior validated Ridge score and matched replacement gap.15 except explicitly named changes',
 universe='Same available-data historical NDX100 union SP100 vendorcap Top32+17 unlevered ETFs',
 hypotheses={
 'validated_control':'Exact older validated-model control without paired replacement',
 'fixed15':'Exact paired gap.15 control',
 'fixed10':'Exact previous adjacent gap.10 result; now a disclosed prior-research comparator, not a new discovery',
 'dispersion_gap':'Prior-signal cross-sectional score IQR relative to its126session median; gap=.15*ratio clamped.075..225; fallback.15',
 'age_decay':'Replacement gap .15 at2sessions, decreases linearly to.075 at6sessions, then stays.075; hard exits unchanged',
 'walk_gap':'Monthly choose fixed-gap.10,.15,.20 reference with largest minimum of past126/252 mean daily log growth, using only data before first signal of month; need252sessions, default.15. Change parameter only; carry own holdings, never splice NAVs',
 'momentum_exception':'Additional entries allowed when score>=.70 AND momentum-group>=.90 AND historical-rebound-group>=.60; original>=.75 entries retained. Original pullback event required for both',
 'entry_limit':'Prior-close buy auction limit=reference_close+.5ATR; cancel new entry if next open exceeds split-adjusted limit. Sells remain as planned, no substitute. Uses auction limit proxy, not a claim of guaranteed execution',
 'model_decay252':'Same1008session pooled Ridge, exponential observation weights half-life252sessions; weighted standardization, weights normalized mean1, alpha10; validate this model causally before use',
 'model_window504':'Same Ridge using last504sessions, same validation gate',
 'model_ensemble':'Average original, decayed and504session raw-return forecasts; validate ensemble before use',
 'history_decay252':'Rebound-history group uses event-conditioned exponentially weighted return average half-life252session, prior20 event shrinkage and20 local matured observations. Other groups unchanged; refit same1008session Ridge and validate'},
 selector='Highest reused2019-2023 CAGR atcost10 among noncontrols with devDD>=-35% and devCAGR30>=fixed15devCAGR30; full/recent returns never select. All history previously inspected.',
 no_leverage=True,max_positions=3,min_target_positions=2,initial_capital=100000,cap_sessions=10,stop_atr=2,take_profit_atr=3,
 model_weight='Ridge remains half rebound group,12.5% total when126/252 matured rank-correlation bothpositive; otherwise pure4groups',
 data_certified=False,performance_publishable=False,no_observation_changes=True,
 references=['https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies?aqrPDF=1','https://www.nber.org/papers/w35158'],
 reference_scope='Motivate costs and state-dependent predictability only; these are independent simple hypotheses, not replications of those papers.')


def weighted_ridge(x,y,weights,alpha=10):
 weights=np.asarray(weights,dtype=float);weights=weights/weights.mean()
 mu=np.average(x,axis=0,weights=weights);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=weights));sd[sd<1e-8]=1
 z=(x-mu)/sd;ym=np.average(y,weights=weights)
 coef=np.linalg.solve(np.einsum('ni,nj,n->ij',z,z,weights)+alpha*np.eye(z.shape[1]),np.einsum('ni,n,n->i',z,y-ym,weights))
 if not np.isfinite(coef).all():raise ValueError('Nonfinite Ridge fit')
 return mu,sd,ym,coef


def walk_model(f,window=1008,halflife=None,first='2019-01-01'):
 idx=f['valid'].index;cols=f['valid'].columns
 X=np.stack([v.to_numpy() for v in f['X']],axis=2);Y=f['y'].to_numpy();event=f['train'].to_numpy();pred=np.full(Y.shape,np.nan);audit=[]
 for month in idx[idx>=first].to_period('M').unique():
  target=np.flatnonzero(idx.to_period('M')==month);t=target[0];end=t-6;start=max(0,end-window)
  x=X[start:end].reshape(-1,X.shape[2]);y=Y[start:end].ravel()
  good=np.isfinite(x).all(1)&np.isfinite(y)&event[start:end].ravel()
  if good.sum()<300:continue
  weights=np.repeat(np.exp2(-np.arange(end-start-1,-1,-1)/halflife) if halflife else np.ones(end-start),len(cols))[good]
  mu,sd,ym,coef=weighted_ridge(x[good],y[good],weights)
  block=X[target];finite=np.isfinite(block).all(2);v=np.full(finite.shape,np.nan);v[finite]=np.einsum('ni,i->n',(block[finite]-mu)/sd,coef)+ym;pred[target]=v
  audit.append(dict(month=str(month),first_prediction=str(idx[t].date()),latest_label=str(idx[end-1+6].date()),samples=int(good.sum()),window=window,halflife=halflife))
 return pd.DataFrame(pred,index=idx,columns=cols),audit


def recent_history(value,event,halflife=252):
 # Zero contributions on non-events keep decay in calendar sessions, not event count.
 observed=event&value.notna();v=value.where(observed,0.).shift(6);n=observed.astype(float).shift(6)
 alpha=1-np.exp2(-1/halflife)
 total=v.ewm(alpha=alpha,adjust=False,min_periods=20).mean()/alpha
 count=n.ewm(alpha=alpha,adjust=False,min_periods=20).mean()/alpha
 pool=total.sum(1,min_count=1)/count.sum(1,min_count=1).replace(0,np.nan)
 local=observed.astype(float).shift(6).expanding(min_periods=20).sum()
 return total.add(pool*20,axis=0).div(count+20).where(local>=20)


def validated_score(f,pred):
 chosen,audit=old.adaptive.choices(pred.index,{'model':old.adaptive.matured_skill(pred,f['y'],f['event'])},['model'])
 pure=sum(f['groups'])/4;with_model=old.adaptive.blended_score(f,pred,.5)
 pure.loc[chosen.eq('model')]=with_model.loc[chosen.eq('model')]
 return pure,audit


def monthly_gap(index,reference_returns,windows=(126,252)):
 out=pd.Series(.15,index=index);audit=[];reference=pd.DataFrame(reference_returns).reindex(index)
 log=np.log1p(reference)
 for month in index[index>='2019-01-01'].to_period('M').unique():
  ids=np.flatnonzero(index.to_period('M')==month);prior=index[ids[0]-1];past=log.loc[:prior].tail(max(windows))
  criterion={str(k):min(float(past[k].tail(w).mean()) for w in windows) for k in past if past[k].notna().sum()==max(windows)}
  chosen=float(max(criterion,key=criterion.get)) if criterion else .15
  out.iloc[ids]=chosen;audit.append(dict(month=str(month),selected_on=str(prior.date()),first_signal=str(index[ids[0]].date()),gap=chosen,criterion=criterion))
 return out,audit


def gap_schedule(index,score,kind,references=None):
 if kind=='walk_gap':return monthly_gap(index,references)
 if kind=='dispersion_gap':
  spread=score.quantile(.75,axis=1)-score.quantile(.25,axis=1)
  series=(.15*spread/spread.rolling(126,min_periods=63).median().replace(0,np.nan)).clip(.075,.225).fillna(.15)
  return series,[]
 return pd.Series(.10 if kind=='fixed10' else .15,index=index),[]


def planner(kind,gaps):
 if kind=='validated_control':return old.prior.daily_plan(.15)
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev];gap=float(gaps.loc[prev])
  pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)]
  remaining=set(state['positions']);sells=[];pairs=[]
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];why=None
   if age>=10:why='holding_limit'
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
   p=state['positions'][held];age=i-dates.get_loc(pd.Timestamp(p['entry_date']))
   if age<2 or pd.isna(sc[held]):continue
   threshold=gap*max(.5,1-(age-2)/8) if kind=='age_decay' else gap
   others=(remaining-{held})|{v[0] for v in selected}
   challenger=next(((s,v) for s,v in pool.items() if s!=held and s not in state['positions'] and feasible(s,others) and v-sc[held]>threshold),None)
   if challenger is None:continue
   s,v=challenger;sells.append(dict(symbol=held,reason='matched_better_candidate'));sold.add(held);remaining.remove(held);selected.append((s,v));pairs.append(dict(sell=held,buy=s,gap=float(v-sc[held]),required_gap=threshold))
  n=len(remaining)+len(selected)
  if n<2:sells.extend(dict(symbol=s,reason='fewer_than_two_holdings') for s in remaining);selected=[];n=0
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,
   buys=[dict(symbol=s,budget=state['equity']/n,score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected],
   candidate_count=len(pool),equity_basis=state['equity'],replacement_pairs=pairs,gap=gap)
 return plan


def execute_limit(name,state,decision,bars,dates,cost_bps=10,sector=None,limit_atr=.5):
 decision=copy.deepcopy(decision);keep=[];cancel=[]
 for buy in decision['buys']:
  s=buy['symbol'];bar=bars.get(s,{});split=bar.get('split',np.nan);price=bar.get('open',np.nan)
  if np.isfinite(price) and price>0 and np.isfinite(split) and split>0:
   limit=(buy['reference_close']+limit_atr*buy['atr'])/split
   if price>limit:
    cancel.append(dict(session=decision['session'],symbol=s,reason='opening_above_predeclared_limit',opening=float(price),limit=float(limit)));continue
  keep.append(buy)
 decision['buys']=keep
 out=old.prior.relaxed_daily(name,state,decision,bars,dates,cost_bps=cost_bps,sector=None)
 return (*out[:5],out[5]+cancel)


def prepare():
 m,d=r.prepare();d=r.apply(m,d,P/'pool');f,_=r.features(d)
 base,_,audit=r.model(f,6);decayed,da=walk_model(f,halflife=252);short,sa=walk_model(f,window=504)
 ff=dict(f);ff['groups']=list(f['groups']);ff['X']=list(f['X'])
 hist=recent_history(f['y'],f['event']);ff['groups'][1]=hist.where(f['valid']).rank(axis=1,pct=True);ff['X'][1]=ff['groups'][1]
 hp,ha=walk_model(ff)
 models={'base':base,'model_decay252':decayed,'model_window504':short,'model_ensemble':(base+decayed+short)/3,'history_decay252':hp}
 scores={};choices={}
 for name,pred in models.items():scores[name],choices[name]=validated_score(ff if name=='history_decay252' else f,pred)
 references={}
 for gap,filename in [(.1,old.P/'robustness/gap10_cost10.json'),(.15,old.P/'matched_replacement_cost10.json'),(.2,old.P/'robustness/gap20_cost10.json')]:
  rows=json.loads(filename.read_text())['daily'];references[gap]=pd.Series({pd.Timestamp(z['date']):z['return'] for z in rows})
 return d,f,scores,{'base':audit,'decay':da,'short':sa,'history':ha},choices,references


def run(kind,cost,d,f,score,mask,gaps,start='2019-01-02',limit_atr=None):
 h.plan=planner(kind,gaps);h.execute=functools.partial(execute_limit if kind=='entry_limit' else old.prior.relaxed_daily,cost_bps=cost)
 if limit_atr is not None:h.execute=functools.partial(execute_limit,cost_bps=cost,limit_atr=limit_atr)
 ot=h.terminal_baskets.apply;os=h.spin_distributions.apply
 try:
  h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
  out=h.run_account('baseline',d,start,'2026-10-06',{'baseline':dict(score=score,mask=mask,atr=f['atr'],ma5=f['ma5'])})
 finally:h.terminal_baskets.apply=ot;h.spin_distributions.apply=os
 if out['status']!='complete':raise ValueError(str({k:out.get(k) for k in ['status','error','blocked_session']}))
 assert all(v['signal_date']<v['session'] for v in out['decisions'])
 assert all(v['cash']>=-1e-6 and v['exposure']<=1+1e-8 for v in out['daily'])
 return out


def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite completed or partial research')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 protected=old.protected_hashes();protected[str(Path(old.__file__))]=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest();(P/'SOURCE_HASHES.json').write_text(json.dumps(protected,indent=2)+'\n')
 d,f,scores,audits,choices,refs=prepare();(P/'TRAINING_AUDIT.json').write_text(json.dumps(audits,indent=2)+'\n');(P/'MODEL_CHOICES.json').write_text(json.dumps(choices,indent=2)+'\n');summary={};gap_audits={}
 # Store score arrays so follow-up checks use identical predictions, not retraining.
 np.savez_compressed(P/'SCORES.npz',**{k:v.to_numpy() for k,v in scores.items()});(P/'AXES.json').write_text(json.dumps(dict(dates=[str(v.date()) for v in d['close'].index],symbols=list(d['close'].columns)))+'\n')
 for kind in KINDS:
  sc=scores.get(kind,scores['base']);mask=f['event']&sc.ge(.75)
  if kind=='momentum_exception':mask=f['event']&(sc.ge(.75)|(sc.ge(.70)&f['groups'][0].ge(.90)&f['groups'][1].ge(.60)))
  gaps,gap_audits[kind]=gap_schedule(sc.index,sc,kind,refs)
  for cost in CONFIG['cost_bps']:
   out=run(kind,cost,d,f,sc,mask,gaps)
   if kind in ['validated_control','fixed15','fixed10']:
    path=old.P/(f'validated_base_cost{cost}.json' if kind=='validated_control' else f'matched_replacement_cost{cost}.json' if kind=='fixed15' else f'robustness/gap10_cost{cost}.json')
    ref=json.loads(path.read_text());assert ref['orders']==out['orders'];np.testing.assert_allclose([v['equity'] for v in ref['daily']],[v['equity'] for v in out['daily']],rtol=1e-12,atol=1e-7)
   key=f'{kind}_cost{cost}';(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
   summary[key]=old.prior.metrics(out);summary[key]['diagnostics']=old.diagnostics(out);summary[key]['cancellations']=len(out['cancellations'])
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(key,'CAGR',summary[key]['full']['cagr'],'DEV',summary[key]['development']['cagr'],'LAST',summary[key]['last252']['return_'],flush=True)
 (P/'GAP_CHOICES.json').write_text(json.dumps(gap_audits,indent=2)+'\n')
 eligible=[k for k in KINDS[3:] if summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=summary['fixed15_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 (P/'SELECTION.json').write_text(json.dumps(dict(selected=selected,eligible=eligible,criterion=CONFIG['selector']),indent=2)+'\n')
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in protected.items())
 assert all(v.get('latest_label',v.get('latest_label_session'))<v['first_prediction'] for a in audits.values() for v in a)
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),controls_reproduced=True,protected_unchanged=True,labels_mature=True,data_certified=False),indent=2)+'\n')
 print('COMPLETE',selected,flush=True)

if __name__=='__main__':main()
