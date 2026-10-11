"""Predeclared long-only daily/monthly sleeve research; no live ledger imports."""
from pathlib import Path
import sys,json,copy,functools,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent;F=P.parent/'four_factor_rebound';sys.path.insert(0,str(F))
from prepare import prepare,U
from vendor_pool import apply
from factors import build,matured_average
from run import period_metrics
from validate_selected import summarise
import historical_backtest as h
ORIGINAL_EXEC=h.execute

def model(f,horizon,first='2019-01-01'):
 idx=f['valid'].index;X=np.stack([v.to_numpy() for v in f['X']],axis=2);y=f['y'].to_numpy();valid=f['train'].to_numpy();pred=np.full(y.shape,np.nan);tree=pred.copy();audit=[]
 for month in idx[idx>=first].to_period('M').unique():
  targets=np.flatnonzero(idx.to_period('M')==month);t=targets[0];end=t-horizon;start=max(0,end-1008)
  xx=X[start:end].reshape(-1,X.shape[2]);yy=y[start:end].ravel();ok=np.isfinite(xx).all(axis=1)&np.isfinite(yy)&valid[start:end].ravel();xx=xx[ok];yy=yy[ok]
  if len(yy)<300:continue
  mu=xx.mean(0);sd=xx.std(0);sd[sd<1e-8]=1;z=(xx-mu)/sd;ym=yy.mean();coef=np.linalg.solve(np.einsum('ni,nj->ij',z,z)+10*np.eye(z.shape[1]),np.einsum('ni,n->i',z,yy-ym))
  residual=yy-(np.einsum('ni,i->n',z,coef)+ym);best=None
  for j in range(xx.shape[1]):
   for threshold in np.quantile(xx[:,j],[.25,.5,.75]):
    left=xx[:,j]<=threshold
    if min(left.sum(),(~left).sum())<50:continue
    a=residual[left].mean();b=residual[~left].mean();err=np.square(residual-np.where(left,a,b)).mean()
    if best is None or err<best[0]:best=(err,j,threshold,a,b)
  block=X[targets];finite=np.isfinite(block).all(2);v=np.full(finite.shape,np.nan);v[finite]=np.einsum('ni,i->n',(block[finite]-mu)/sd,coef)+ym;pred[targets]=v
  tv=v.copy()
  if best is not None:
   _,j,threshold,a,b=best;tv[finite]=v[finite]+.15*np.where(block[finite][:,j]<=threshold,a,b)
  tree[targets]=tv
  if not np.isfinite(v[finite]).all():raise ValueError('Nonfinite model')
  audit.append(dict(month=str(month),train_last_signal=str(idx[end-1].date()),latest_label_session=str(idx[end-1+horizon].date()),first_prediction=str(idx[t].date()),samples=len(yy),horizon=horizon))
 return pd.DataFrame(pred,index=idx,columns=f['valid'].columns),pd.DataFrame(tree,index=idx,columns=f['valid'].columns),audit

def features(d):
 f=build(d);a=d['adj'];c=d['close'];ret=a.pct_change(fill_method=None);vol=ret.rolling(63).std();valid=f['valid'];rank=lambda x:x.where(valid).rank(axis=1,pct=True)
 annual=a.shift(21)/a.shift(252)-1;mom=(rank(annual/vol)+rank(a.pct_change(252,fill_method=None)/vol))/2
 z=a.pct_change(3,fill_method=None)/(ret.rolling(20).std()*np.sqrt(3));loc=((c-d['low'])/(d['high']-d['low'])).fillna(.5)
 gate=valid&(annual>0)&(a>a.rolling(100).mean())&((z<-.35)|(loc<.2))&(z>-3.5)
 # Event-conditioned trailing outcomes with maturity delay; original rebound/path formulas retained.
 ao=d['opening']*a/c;y=ao.shift(-6)/ao.shift(-1)-1-.001
 hist=matured_average(y,gate);rebound=rank(hist);route=f['groups'][2];shape=f['groups'][3]
 groups=[mom,rebound,route,shape];X=groups+[rank(z),rank(vol),rank(a/a.rolling(20).max()-1),f['X'][6]]
 daily=dict(valid=valid,train=gate,X=X,y=y,groups=groups,event=gate,atr=f['atr'],ma5=f['ma5'],market=f['market'])
 skip2=a.shift(42)/a.shift(252)-1;skip6=a.shift(42)/a.shift(126)-1
 relative=rank((1+a.pct_change(63,fill_method=None))/(1+d['sector60'])-1)
 cmf=f['X'][6];risk=rank(skip2/vol);raw=rank(skip2);short=rank(skip6);lowvol=rank(-vol);upside=rank(vol);near=rank(a/a.rolling(252).max())
 monthly=dict(valid=valid,train=valid,X=[raw,risk,relative,cmf,lowvol,near,short,upside],y=a.shift(-22)/a.shift(-1)-1-.001,atr=f['atr'],ma5=f['ma5'],positive=skip2>0,market=f['market'])
 monthly['base']=.50*raw+.20*risk+.15*relative+.10*cmf+.05*near
 monthly['risk']=.60*risk+.20*relative+.10*cmf+.10*lowvol
 monthly['fast']=.50*raw+.20*short+.15*relative+.10*cmf+.05*upside
 return daily,monthly

def daily_plan(cfg):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  i=dates.get_loc(day);prev=dates[i-1];sc=score.loc[prev];pool=sc.where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');pool=pool[pool.index.isin(reps)];sells=[];remaining=set(state['positions'])
  for s,p in state['positions'].items():
   age=i-dates.get_loc(pd.Timestamp(p['entry_date']));price=close.at[prev,s];why=None
   if age>=cfg['cap']:why='holding_limit'
   elif price<p['entry_price']-cfg['sl']*p['atr']:why='close_stop_atr'
   elif price>p['entry_price']+cfg['tp']*p['atr']:why='close_take_profit_atr'
   elif pd.notna(sc[s]) and sc[s]<.35:why='rank_deterioration'
   elif age>=2 and len(pool) and pool.iloc[0]-sc[s]>.15:why='better_candidate'
   if cfg['cash_exit'] and len(pool)<2:why='too_few_candidates_cash'
   if why:sells.append(dict(symbol=s,reason=why));remaining.remove(s)
  sold={r['symbol'] for r in sells};selected=[]
  for s,value in pool.items():
   if len(remaining)+len(selected)>=3:break
   held=remaining|{r[0] for r in selected}
   if s in held|sold or issuers[s] in {issuers[t] for t in held}:continue
   if sum(sectors[t]==sectors[s] for t in held)>=cfg['sector_cap']:continue
   selected.append((s,value))
  n=len(remaining)+len(selected)
  if n<2:
   for s in remaining:sells.append(dict(symbol=s,reason='fewer_than_two_holdings'))
   selected=[];n=0
  buys=[dict(symbol=s,budget=float(state['equity']/n),score=float(v),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s,v in selected]
  return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,buys=buys,equity_basis=state['equity'],candidate_count=len(pool))
 return plan

def monthly_plan(monthends,sector_cap=2):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  prev=dates[dates.get_loc(day)-1];decision=dict(session=str(day.date()),signal_date=str(prev.date()),sells=[],buys=[],equity_basis=state['equity'],candidate_count=0)
  if day not in monthends:return decision
  pool=score.loc[prev].where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable');selected=[]
  for s,v in pool.items():
   if s not in reps or issuers[s] in {issuers[t] for t in selected}:continue
   if sum(sectors[t]==sectors[s] for t in selected)>=sector_cap:continue
   selected.append(s)
   if len(selected)==3:break
  if len(selected)<2:selected=[]
  decision['sells']=[dict(symbol=s,reason='month_end_rebalance') for s in state['positions']]
  decision['buys']=[dict(symbol=s,budget=state['equity']/len(selected),score=float(pool[s]),atr=float(atr.at[prev,s]),reference_close=float(close.at[prev,s])) for s in selected]
  decision['candidate_count']=len(pool);return decision
 return plan

def execute_equal_close(name,state,decision,bars,dates,cost_bps=10,sector=None):
 # Rank is already frozen using prior close. Size fractions from actual close liquidation
 # proceeds, including costs/actions. Liquidate/rebuy even unchanged targets (conservative cost).
 decision=copy.deepcopy(decision);half=cost_bps/20000
 if decision['buys']:
  if set(state['positions'])!={r['symbol'] for r in decision['sells']}:raise ValueError('Monthly replacement requires full liquidation')
  proceeds=state['cash']
  for s,p in state['positions'].items():
   bar=bars[s];q=p['qty']*bar['split'];proceeds+=q*(bar['dividends']+bar['open']*(1-half))
  if not np.isfinite(proceeds) or proceeds<0:raise ValueError('Invalid monthly proceeds')
  for r in decision['buys']:r['budget']=proceeds/len(decision['buys'])
 result=h.engine(name).execute(state,decision,bars,dates,cost_bps=cost_bps,sector=sector)
 return (*result,[])

def combo_daily(a,b,w):
 x=pd.DataFrame(a).set_index('date');y=pd.DataFrame(b).set_index('date')
 if not x.index.equals(y.index):raise ValueError('Sleeve date mismatch')
 return [dict(date=t,equity=float(w*x.at[t,'equity']+(1-w)*y.at[t,'equity'])) for t in x.index]

def main():
 dailycfg=[]
 for mode in ['rules','ridge','ridge_tree']:
  for threshold in [.65,.75]:
   for variant in ['tight','patient']:
    dailycfg.append(dict(name=f'd_{mode}_{int(threshold*100)}_{variant}',mode=mode,threshold=threshold,cap=5 if variant=='tight' else 10,tp=2 if variant=='tight' else 3,sl=2,cash_exit=variant=='tight',sector_cap=2,market_gate=True))
 monthlycfg=[dict(name=f'm_{mode}_{gate}',mode=mode,gate=gate) for mode in ['base','risk','fast','ridge','ridge_tree'] for gate in ['positive','market']]
 config=dict(daily=dailycfg,monthly=monthlycfg,sleeve_weights=[0,.25,.5,.75,1],initial_total=100000,leverage=1,start='2019-01-02',end='2026-10-06',selection='Max development2019-2023 CAGR subject to developmentDD>=-40%; do not choose on recent returns',target='last252 net return >=100%; do not force outcome',monthly_execution='Last full trading session of calendar month CLOSE using previous session signals; liquidate/rebuy equal net budgets, fixed shares between month ends; corporate actions handled',combination='Fixed initial capital split, no transfers or duplicate capital; NAV=w*daily_NAV+(1-w)*monthly_NAV',pool='Vendor past snapshots Top32 +17 unleveredETFs, same limitations as vendor32_research',cost_bps=10)
 (P/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 m,d=prepare();d=apply(m,d,P/'pool');df,mf=features(d);print('features ready',flush=True)
 dp,dt,da=model(df,6);mp,mt,ma=model(mf,22);(P/'TRAINING_AUDIT.json').write_text(json.dumps(dict(daily=da,monthly=ma),indent=2)+'\n');print('models ready',flush=True)
 result={};summary={};cache={}
 def save(name,r):
  result[name]=r;summary[name]=summarise(r);(P/(name+'.json')).write_text(json.dumps(r,allow_nan=False)+'\n');(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(name,json.dumps(summary[name]),flush=True)
 for cfg in dailycfg:
  g=df['groups'];sc=sum(g)/4
  if cfg['mode']!='rules':
   pred=dt if cfg['mode']=='ridge_tree' else dp
   sc=(g[0]+.5*g[1]+.5*pred.where(df['valid']).rank(axis=1,pct=True)+g[2]+g[3])/4
  ff=dict(score=sc,mask=(df['event']&(sc>=cfg['threshold'])).mul(df['market'],axis=0),atr=df['atr'],ma5=df['ma5']);h.plan=daily_plan(cfg)
  # Planner permits up to two per sector; unchanged execution guards remain active.
  h.execute=ORIGINAL_EXEC;cache[cfg['name']]=(d,ff,h.plan,False);save(cfg['name'],h.run_account('baseline',d,config['start'],m['end'],{'baseline':ff}))
 dates=d['close'].index;monthends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);md=dict(d);md['opening']=d['close']
 for cfg in monthlycfg:
  sc=mf[cfg['mode']] if cfg['mode'] in ['base','risk','fast'] else .9*mf['base']+.1*(mt if cfg['mode']=='ridge_tree' else mp).where(mf['valid']).rank(axis=1,pct=True)
  mask=mf['valid']&mf['positive']
  if cfg['gate']=='market':mask=mask.mul(mf['market'],axis=0)
  ff=dict(score=sc,mask=mask,atr=mf['atr'],ma5=mf['ma5']);h.plan=monthly_plan(monthends);h.execute=execute_equal_close;cache[cfg['name']]=(md,ff,h.plan,True);save(cfg['name'],h.run_account('baseline',md,config['start'],m['end'],{'baseline':ff}))
 # Every pair is disclosed; development-selected pair is distinct from retrospective winner.
 combinations={}
 for dn in [c['name'] for c in dailycfg]:
  for mn in [c['name'] for c in monthlycfg]:
   if result[dn]['status']!='complete' or result[mn]['status']!='complete':continue
   for w in [.25,.5,.75]:
    rows=combo_daily(result[dn]['daily'],result[mn]['daily'],w);x=pd.DataFrame(rows);name=f'{dn}__{mn}__daily{w}'
    combinations[name]=dict(daily=dn,monthly=mn,daily_weight=w,development=period_metrics(rows,config['start'],'2023-12-31'),evaluation=period_metrics(rows,'2024-01-01',m['end']),full=period_metrics(rows,config['start'],m['end']),last252_return=float(x.equity.iloc[-1]/x.equity.iloc[-253]-1),year2026=period_metrics(rows,'2026-01-01',m['end']))
 (P/'COMBINATIONS.json').write_text(json.dumps(combinations,indent=2)+'\n')
 eligible={k:v for k,v in combinations.items() if v['development']['max_drawdown']>=-.4};selected=max(eligible,key=lambda k:eligible[k]['development']['cagr']) if eligible else None
 pick=dict(selected=selected,combination=eligible.get(selected),retrospective_best_recent=max(combinations,key=lambda k:combinations[k]['last252_return']) if combinations else None,target_met_count=sum(v['last252_return']>=1 for v in combinations.values()),combinations_tested=len(combinations))
 (P/'SELECTION.json').write_text(json.dumps(pick,indent=2)+'\n')
 if selected:
  costs={}
  for name in [pick['combination']['daily'],pick['combination']['monthly']]:
   data,ff,pl,is_month=cache[name];h.plan=pl;baseexec=execute_equal_close if is_month else ORIGINAL_EXEC
   h.execute=functools.partial(baseexec,cost_bps=30)
   original_terminal=h.terminal_baskets.apply;original_spin=h.spin_distributions.apply
   h.terminal_baskets.apply=functools.partial(original_terminal,cost_bps=30);h.spin_distributions.apply=functools.partial(original_spin,cost_bps=30)
   r=h.run_account('baseline',data,config['start'],m['end'],{'baseline':ff});costs[name]=r;h.terminal_baskets.apply=original_terminal;h.spin_distributions.apply=original_spin
   (P/(name+'_cost30.json')).write_text(json.dumps(r,allow_nan=False)+'\n')
  if all(r['status']=='complete' for r in costs.values()):
   rows=combo_daily(costs[pick['combination']['daily']]['daily'],costs[pick['combination']['monthly']]['daily'],pick['combination']['daily_weight']);x=pd.DataFrame(rows);stress=dict(full=period_metrics(rows,config['start'],m['end']),last252_return=float(x.equity.iloc[-1]/x.equity.iloc[-253]-1));(P/'COST_STRESS.json').write_text(json.dumps(stress,indent=2)+'\n')
 print('DONE',json.dumps(pick),flush=True)
if __name__=='__main__':main()
