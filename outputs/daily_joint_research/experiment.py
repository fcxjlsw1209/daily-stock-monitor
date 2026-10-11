"""Declared return/path-risk and asset-class experiments on the existing daily pool."""
from pathlib import Path
import importlib.util
import sys,json,functools,hashlib
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'strategy_improvements'))
import research as prior
r=prior.r
h=r.h

VARIANTS=['baseline','cost_margin','context_return','asset_split','context_split',
 'joint_path','joint_tail_gate','joint_cost_gate','joint_split_gate','stocks_only','horizon10']
CONFIG=dict(start='2019-01-02',end='2026-10-06',cost_bps=[10,30],variants=VARIANTS,
 declared_before_results=True,optimization_grid=False,initial_capital=100000,
 source_pool='Same available-data historical index-union vendor Top32 plus17 nonlevered ETFs',
 keep_existing_rules='Long-only,max3,minimum2,score>=.75,stop2ATR,takeprofit3ATR,cap10,rank replacement.15',
 group_weights=[.25,.25,.25,.25],ridge_alpha=10,logistic_alpha=10,training_sessions=1008,
 refit='monthly, label fully mature before prediction month',min_train_samples=300,
 tail_event='Future1..H intraday low below next open minus2 signal-date ATR; risk feature, not an intraday stop fill model',
 hypotheses={
  'baseline':'Exact reproduction of prior daily Ridge score and execution',
  'cost_margin':'Existing control: baseline predicted5day netreturn>.003',
  'context_return':'Add lagged-beta residual weakness, market/sector context and fixed interactions to return Ridge; replace only predicted return half of rebound group',
  'asset_split':'Same original features, separate equity/ETF Ridge models, pooled fallback if fewer300 matured training observations',
  'context_split':'Same context features with separate equity/ETF models',
  'joint_path':'Context return score; path group=.5oldpath+.5rank(predictedMFE-predictedMAE)',
  'joint_tail_gate':'Joint path score plus logisticP(2ATRbadpath)<=.35',
  'joint_cost_gate':'Joint path score plus predictednetreturn>.003',
  'joint_split_gate':'Split context/path models plus both netreturn>.003 and badpathprobability<=.35',
  'stocks_only':'Baseline scores/models; prohibit new ETF entries. Tests entry pool rather than model quality.',
  'horizon10':'Context return/path models target next-open to open11,10-day excursions; actual10-session holding cap unchanged'},
 selection='Highest2019-2023 net CAGR with developmentDD>=-35% and cost30 development CAGR>=baseline cost30 development CAGR. Reused historical development, not pristine holdout. Report all failures.',
 avoid_recent_target_tuning=True,data_certified=False,live_accounts_modified=False,
 research_reference='https://www.nber.org/papers/w25398; motivates context interactions only, not evidence for this particular short-horizon strategy')


def context_features(d,f):
 a=d['adj'];ret=a.pct_change(fill_method=None);vol=ret.rolling(20).std().clip(lower=.001)
 market=ret['SPY'];mv=market.rolling(20).std().clip(lower=.001)
 beta=ret.rolling(126).cov(market).div(market.rolling(126).var(),axis=0).shift(1)
 residual=ret-beta.mul(market,axis=0);rv=residual.rolling(20).std().clip(lower=.001)
 mz=(a['SPY'].pct_change(3,fill_method=None)/(mv*np.sqrt(3))).clip(-8,8)
 rz=(residual.rolling(3).sum()/(rv*np.sqrt(3))).clip(-8,8)
 z=(a.pct_change(3,fill_method=None)/(vol*np.sqrt(3))).clip(-8,8)
 broad=lambda series:pd.DataFrame(np.repeat(series.to_numpy()[:,None],len(a.columns),axis=1),index=a.index,columns=a.columns)
 market_z=broad(mz);mt=broad(a['SPY']/a['SPY'].rolling(200).mean()-1)
 market_vol=broad(mv/market.rolling(252).std().clip(lower=.001))
 sector=(a.pct_change(63,fill_method=None)-d['sector60']).clip(-2,2)
 vr=(d['volume']/d['volume'].rolling(20).mean()).clip(0,8)
 gap=((d['opening']/d['close'].shift()-1)/vol).clip(-8,8)
 intraday=((d['close']/d['opening']-1)/vol).clip(-8,8)
 isfund=pd.DataFrame([float(d['issuers'][s].startswith('FUND_')) for s in a.columns],index=a.columns).T
 isfund=pd.DataFrame(np.repeat(isfund.to_numpy(),len(a),axis=0),index=a.index,columns=a.columns)
 # All interactions are specified before looking at results; no feature selection.
 return list(f['X'])+[z,rz,market_z,mt,market_vol,sector,vr,gap,intraday,isfund,
  rz*market_z,z*market_vol,f['groups'][0]*market_z,f['groups'][3]*vr]


def path_labels(d,f,horizon):
 a=d['adj'];ratio=a/d['close'];op=d['opening']*ratio;hi=d['high']*ratio;lo=d['low']*ratio
 entry=op.shift(-1);exit_=op.shift(-(horizon+1))
 hh=np.stack([hi.shift(-i).to_numpy() for i in range(1,horizon+1)],axis=2)
 ll=np.stack([lo.shift(-i).to_numpy() for i in range(1,horizon+1)],axis=2)
 valid=np.isfinite(hh).all(2)&np.isfinite(ll).all(2)&np.isfinite(entry.to_numpy())&np.isfinite(exit_.to_numpy())
 valid&=(entry.to_numpy()>0)&(exit_.to_numpy()>0)&(ll>0).all(2)
 up=pd.DataFrame(hh.max(2),index=a.index,columns=a.columns)/entry-1
 down=1-pd.DataFrame(ll.min(2),index=a.index,columns=a.columns)/entry
 up=up.clip(lower=0).where(valid);down=down.clip(lower=0).where(valid)
 bad=(down>2*f['atr']*ratio/entry).astype(float).where(valid&np.isfinite(f['atr'].to_numpy()))
 return {'mu':(exit_/entry-1-.001).where(valid),'up':up,'down':down,'bad':bad}


def fit_models(x,y,alpha=10):
 mean=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.;z=(x-mean)/sd
 ym=y[:,:3].mean(0)
 coef=np.linalg.solve(np.einsum('ni,nj->ij',z,z)+alpha*np.eye(z.shape[1]),np.einsum('ni,nj->ij',z,y[:,:3]-ym))
 # Regularized logistic regression for the binary adverse-path outcome.
 design=np.column_stack([np.ones(len(z)),z]);target=y[:,3];p=(target.sum()+.5)/(len(target)+1)
 theta=np.zeros(design.shape[1]);theta[0]=np.log(p/(1-p));penalty=np.r_[0.,np.repeat(alpha,z.shape[1])]
 converged=False
 for iteration in range(40):
  prob=1/(1+np.exp(-np.clip(np.einsum('ni,i->n',design,theta),-35,35)))
  weight=np.maximum(prob*(1-prob),1e-8)
  grad=np.einsum('ni,n->i',design,target-prob)-penalty*theta
  fisher=np.einsum('ni,nj->ij',design,weight[:,None]*design)+np.diag(penalty)+np.eye(len(theta))*1e-10
  delta=np.linalg.solve(fisher,grad)
  # Fixed Newton step backtracking stabilizes separable/small groups.
  def loss(v):
   eta=np.einsum('ni,i->n',design,v)
   return np.sum(np.logaddexp(0,eta)-target*eta)+.5*np.sum(penalty*v*v)
  step=1.;old=loss(theta)
  while step>1/1024 and loss(theta+step*delta)>old+1e-9:step*=.5
  theta+=step*delta
  if np.max(np.abs(step*delta))<1e-6:converged=True;break
 if not np.isfinite(coef).all() or not np.isfinite(theta).all():raise ValueError('Nonfinite model coefficients')
 return dict(mean=mean,sd=sd,ym=ym,coef=coef,theta=theta,converged=converged,iterations=iteration+1,base_rate=float(target.mean()))


def predict(model,x):
 z=(x-model['mean'])/model['sd'];continuous=np.einsum('ni,ij->nj',z,model['coef'])+model['ym']
 continuous[:,1:]=np.maximum(continuous[:,1:],0)
 bad=1/(1+np.exp(-np.clip(np.einsum('ni,i->n',np.column_stack([np.ones(len(z)),z]),model['theta']),-35,35)))
 return np.column_stack([continuous,bad])


def walk_fit(f,features,labels,issuers,horizon,split=False,first='2019-01-01'):
 idx=f['valid'].index;cols=f['valid'].columns
 x=np.stack([v.to_numpy() for v in features],axis=2)
 y=np.stack([labels[k].to_numpy() for k in ['mu','up','down','bad']],axis=2)
 train=f['train'].to_numpy();out=np.full(y.shape,np.nan);audit=[]
 groups=np.array([int(issuers[s].startswith('FUND_')) for s in cols])
 for month in idx[idx>=first].to_period('M').unique():
  target=np.flatnonzero(idx.to_period('M')==month);t=target[0];end=t-(horizon+1);start=max(0,end-1008)
  xx=x[start:end].reshape(-1,x.shape[2]);yy=y[start:end].reshape(-1,4)
  known=np.isfinite(xx).all(1)&np.isfinite(yy).all(1)&train[start:end].ravel()
  gx=np.tile(groups,end-start);pooled=fit_models(xx[known],yy[known]) if known.sum()>=300 else None
  if pooled is None:continue
  for group in ([0,1] if split else [None]):
   fitmask=known if group is None else known&(gx==group)
   fallback=fitmask.sum()<300
   model=pooled if group is None or fallback else fit_models(xx[fitmask],yy[fitmask])
   block=x[target];finite=np.isfinite(block).all(2)
   if group is not None:finite &= groups[None,:]==group
   values=out[target].copy();values[finite]=predict(model,block[finite]);out[target]=values
   audit.append(dict(month=str(month),asset_group=group,samples=int(known.sum() if fallback else fitmask.sum()),pooled_fallback=bool(fallback),
    train_last_signal=str(idx[end-1].date()),latest_label_session=str(idx[end-1+horizon+1].date()),
    first_prediction=str(idx[t].date()),horizon=horizon,logistic_converged=model['converged'],
    logistic_iterations=model['iterations'],training_bad_rate=model['base_rate']))
 return {key:pd.DataFrame(out[:,:,i],index=idx,columns=cols) for i,key in enumerate(['mu','up','down','bad'])},audit


def score(f,pred,joint=False):
 rank=lambda x:x.where(f['valid']).rank(axis=1,pct=True);g=f['groups']
 path=.5*g[2]+.5*rank(pred['up']-pred['down']) if joint else g[2]
 return (g[0]+.5*g[1]+.5*rank(pred['mu'])+path+g[3])/4


def diagnostic(out):
 t=pd.DataFrame(out['trades'])
 return dict(closed_trades=len(t),win_rate=float((t.net_pnl>0).mean()) if len(t) else None,
  mean_sessions=float(t.sessions.mean()) if len(t) else None,mean_candidates=float(np.mean([v['candidate_count'] for v in out['decisions']])))


def calibration(pred,labels,event):
 rows=[]
 for period,start,end in [('development','2019-01-02','2023-12-31'),('later','2024-01-01','2026-10-06')]:
  ev=event.loc[start:end].to_numpy()
  actual=labels['mu'].loc[start:end].to_numpy();p=pred['mu'].loc[start:end].to_numpy()
  good=ev&np.isfinite(actual)&np.isfinite(p)
  yy=actual[good];pp=p[good]
  if not len(yy):continue
  cuts=np.quantile(pp,[.2,.4,.6,.8]);bins=np.searchsorted(cuts,pp,side='right')
  for b in range(5):
   mask=bins==b;rows.append(dict(period=period,return_prediction_quintile=b+1,n=int(mask.sum()),mean_prediction=float(pp[mask].mean()),actual_mean_net5_or10day=float(yy[mask].mean())))
  if 'bad' not in pred:continue
  actual_bad=labels['bad'].loc[start:end].to_numpy();pbad=pred['bad'].loc[start:end].to_numpy()
  good=ev&np.isfinite(actual_bad)&np.isfinite(pbad);yy=actual_bad[good];pp=pbad[good]
  rows.append(dict(period=period,metric='adverse_path_brier',n=len(yy),value=float(np.mean((pp-yy)**2)),constant_period_rate_brier=float(np.mean((yy-yy.mean())**2)),
   note='Constant uses evaluation-period rate as a hindsight diagnostic, not a tradable model.'))
 return rows


def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Completed/partial results exist; use a new experiment directory instead of overwriting')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 protected=[P.parent/'strategy_improvements/research.py',P.parent/'dual_strategy_research/research.py',r.U/'alias_resolved_inputs/MANIFEST.json']
 protected += list((P.parent/'monthly_buffer5_monitor').glob('FREEZE.json'))
 hashes={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in protected}
 (P/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 m,d=r.prepare();d=r.apply(m,d,P/'pool');f,_=r.features(d)
 basic,_,baseaudit=r.model(f,6);basepred=dict(mu=basic)
 cx=context_features(d,f);labels5=path_labels(d,f,5);labels10=path_labels(d,f,10)
 models={};audits={'baseline':baseaudit};cal={'baseline':calibration(basepred,labels5,f['event'])}
 for name,features,labels,horizon,split in [
  ('context',cx,labels5,5,False),('split',f['X'],labels5,5,True),
  ('context_split',cx,labels5,5,True),('context10',cx,labels10,10,False)]:
  models[name],audits[name]=walk_fit(f,features,labels,d['issuers'],horizon,split)
  cal[name]=calibration(models[name],labels,f['event'])
  np.savez_compressed(P/(name+'_predictions.npz'),**{k:v.to_numpy() for k,v in models[name].items()})
  print('MODEL_READY',name,'fits',len(audits[name]),flush=True)
 (P/'PREDICTION_AXES.json').write_text(json.dumps(dict(dates=[str(v.date()) for v in d['close'].index],symbols=list(d['close'].columns)))+'\n')
 (P/'TRAINING_AUDIT.json').write_text(json.dumps(audits,indent=2)+'\n');(P/'CALIBRATION.json').write_text(json.dumps(cal,indent=2)+'\n')
 summary={};diagnostics={}
 for cost in CONFIG['cost_bps']:
  for kind in VARIANTS:
   model_name={'context_return':'context','asset_split':'split','context_split':'context_split',
    'joint_path':'context','joint_tail_gate':'context','joint_cost_gate':'context','joint_split_gate':'context_split','horizon10':'context10'}.get(kind)
   pred=models[model_name] if model_name else basepred
   sc=score(f,pred,kind.startswith('joint_') or kind=='horizon10')
   mask=f['event']&sc.ge(.75)
   if kind in ['cost_margin','joint_cost_gate','joint_split_gate']:mask &= pred['mu'].gt(.003)
   if kind in ['joint_tail_gate','joint_split_gate']:mask &= pred['bad'].le(.35)
   if kind=='stocks_only':
    for s in mask:
     if d['issuers'][s].startswith('FUND_'):mask[s]=False
   ff=dict(score=sc,mask=mask,atr=f['atr'],ma5=f['ma5']);h.plan=prior.daily_plan(.15);h.execute=functools.partial(prior.relaxed_daily,cost_bps=cost)
   ot=h.terminal_baskets.apply;os=h.spin_distributions.apply
   try:
    h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
    out=h.run_account('baseline',d,CONFIG['start'],CONFIG['end'],{'baseline':ff})
   finally:h.terminal_baskets.apply=ot;h.spin_distributions.apply=os
   assert out['status']=='complete',out.get('error')
   assert all(x['signal_date']<x['session'] for x in out['decisions']);assert all(x['cash']>=-1e-6 for x in out['daily'])
   key=f'{kind}_cost{cost}';(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
   if kind in ['baseline','cost_margin']:
    old=json.loads((P.parent/f'strategy_improvements/daily_{kind}_cost{cost}.json').read_text())
    assert out['orders']==old['orders'];np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in old['daily']],rtol=1e-12,atol=1e-7)
   summary[key]=prior.metrics(out);diagnostics[key]=diagnostic(out)
   (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(P/'DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
   print(key,json.dumps(summary[key]),flush=True)
 candidates=[k for k in VARIANTS if summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=summary['baseline_cost30']['development']['cagr']]
 selected=max(candidates,key=lambda k:summary[k+'_cost10']['development']['cagr']) if candidates else None
 (P/'SELECTION.json').write_text(json.dumps(dict(selected=selected,eligible=candidates,criterion=CONFIG['selection']),indent=2)+'\n')
 assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,baseline_and_cost_margin_reproduced=True,
  label_maturity=all(v['latest_label_session']<v['first_prediction'] for a in audits.values() for v in a),
  logistic_nonconvergence=sum(not v.get('logistic_converged',True) for a in audits.values() for v in a),
  protected_sources_unchanged=True,data_certified=False),indent=2)+'\n');print('DONE',selected,flush=True)


if __name__=='__main__':main()
