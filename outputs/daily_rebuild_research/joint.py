"""Second declared architecture: equal-budget group scores, jointly learned ranking.
Uses fixed25% group scores as four inputs, learns their interactions and relevance.
Unlike the equal expert ensemble, fitted coefficients are NOT fixed25% factors.
"""
import rebuild as b
import numpy as np,pandas as pd,json
from clock_execution import run_account
P=b.P/'joint';P.mkdir(exist_ok=True)
CASES={
 'joint_raw10':dict(target='raw'),
 'joint_rank10':dict(target='rank'),
 'joint_risk10':dict(target='risk'),
 'joint_rank5':dict(target='rank',cap=5),
 'joint_rank15':dict(target='rank',cap=15),
 'joint_rank_no_replace':dict(target='rank',replace=False),
}

def predictions(f,target,population,until=None):
 idx=f['valid'].index;gm=f['groupmean'];cols=f['valid'].columns
 # Equal group aggregates, all six pair interactions, plus within-group centered
 # features. Same feature count and scaling treatment for every group.
 features=gm+[gm[i]*gm[j] for i in range(4) for j in range(i+1,4)]
 features += [v-gm[i] for i,g in enumerate(f['groups']) for v in g]
 X=np.stack([v.to_numpy() for v in features],axis=2);mask=f['event'] if population=='entry' else f['strong'];yy=f['y']
 if target=='rank':yy=yy.where(mask).rank(axis=1,pct=True)-.5
 elif target=='risk':yy=yy/(f['vol']*np.sqrt(5))
 Y=yy.to_numpy();use=mask.to_numpy();out=np.full(Y.shape,np.nan);audit=[]
 months=idx[idx>='2016-08-01'].to_period('M').unique()
 for month in months:
  targetrows=np.flatnonzero(idx.to_period('M')==month);t=targetrows[0]
  if until is not None and idx[t]>pd.Timestamp(until):break
  end=t-6;start=max(0,end-1008);ok=use[start:end]&np.isfinite(Y[start:end])&np.isfinite(X[start:end]).all(2)
  if ok.sum()<300:continue
  ri,ci=np.where(ok);ti=ri+start;y=Y[ti,ci].copy();y=np.clip(y,-5,5) if target=='risk' else np.clip(y,-.5,.5);xx=X[ti,ci]
  counts=np.bincount(ri,minlength=end-start);w=np.exp2(-(end-1-ti)/504)/counts[ri];w*=len(w)/w.sum()
  mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1;z=(xx-mu)/sd;ym=np.average(y,weights=w)
  coef=np.linalg.solve(np.einsum('ni,nj,n->ij',z,z,w)+100*np.eye(z.shape[1]),np.einsum('ni,n,n->i',z,w,y-ym))
  block=X[targetrows];finite=np.isfinite(block).all(2);p=np.full(finite.shape,np.nan);p[finite]=np.einsum('ni,i->n',(block[finite]-mu)/sd,coef)+ym
  if target=='risk':p*=f['vol'].to_numpy()[targetrows]*np.sqrt(5)
  if target=='rank':p*=.05 # Ranking units, NOT forecast returns; five percentile-point turnover threshold via .0025.
  assert np.isfinite(p[finite]).all();out[targetrows]=p
  audit.append(dict(month=str(month),first_prediction=str(idx[t].date()),latest_label=str(idx[ti.max()+6].date()),samples=len(y),coefficients=coef.tolist()))
 return pd.DataFrame(out,index=idx,columns=cols),audit

def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('No overwrite')
 b.dump(P/'CONFIG.json',dict(cases=CASES,declared_before_own_results=True,prior_phase_results_seen=True,group_inputs='Equal four group aggregate scores and within-group centered features, jointly learned interactions; coefficients learned, not fixed25% predictions.',model='Monthly past1008sessions, alpha100, six-session maturity, equal total weight per training date, half-life504; no new libraries.',rank_target='Future5day cross-sectional percentile minus.5 among training population, entry and continuation populations differ: scores are relative opportunity ranks, not commensurable dollars. Scaled .05 solely to use same fixed thresholds: entry above52nd predicted percentile, replace difference>.04.',cost_bps=[0,10,30],clock='next_open;zero-costsameclosediagnostic',data_certified=False))
 # Prepared artifacts reused, models recomputed only for new architecture.
 b.seed_sources();m,d,_=b.prepare();f=b.features(d);modelset={};audits={}
 for target in ['raw','rank','risk']:
  models={}
  for population in ['entry','hold']:
   models[population+'_experts'],a=predictions(f,target,population);audits[target+'_'+population]=a
  modelset[target]=models;print('joint trained',target,flush=True)
 b.dump(P/'TRAINING_AUDIT.json',audits);summary={};(P/'runs').mkdir(exist_ok=True)
 for name,overlay in CASES.items():
  cfg={**b.BASE,**{k:v for k,v in overlay.items() if k!='target'}};models=modelset[overlay['target']]
  for clock,cost in [('next_open',0),('next_open',10),('next_open',30),('same_close_diagnostic',0)]:
   pl,ff=b.planner(cfg,f,models,clock);out=run_account(d,ff,pl,b.START,b.END,cost,clock);key=f'{name}_{clock}_cost{cost}'
   b.dump(P/'runs'/f'{key}.json',out)
   if out['status']!='complete':raise ValueError(out.get('error'))
   summary[key]=b.metrics(out);b.dump(P/'SUMMARY.json',summary);print(key,summary[key]['full'],flush=True)
 b.dump(P/'CHECKS.json',dict(complete=True,runs=len(summary)))
if __name__=='__main__':main()
