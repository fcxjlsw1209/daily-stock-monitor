"""Exact unordered top-k Plackett-Luce likelihood; inverse research only."""
from pathlib import Path
import itertools,sys,argparse,json
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import open_inverse as m
P=Path(__file__).resolve().parent/'listwise'
P.mkdir(exist_ok=True)

def dump(n,z):
 (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def event_design(d,mask):
 month_ids=[];allowed=[];stage=[];target=[];slices=[]
 for i in np.flatnonzero(mask):
  sel=[int(k) for k in d['order'][i] if k>=0]
  if not sel:continue
  start=len(month_ids)
  for seq in itertools.permutations(sel):
   am=np.zeros((3,d['eligible'].shape[1]),bool);st=np.zeros(3,bool)
   remain=d['eligible'][i].copy()
   for j,k in enumerate(seq):am[j]=remain;st[j]=True;remain=remain.copy();remain[k]=False
   month_ids.append(i);allowed.append(am);stage.append(st)
   target.append(np.bincount(sel,minlength=d['eligible'].shape[1]))
  slices.append((start,len(month_ids)))
 return dict(month=np.array(month_ids),allowed=np.stack(allowed),stage=np.stack(stage),target=np.stack(target),slices=slices)

def objective(w,X,ev,groups,reg,side_penalty,order_mass):
 local=np.einsum('ijk,k->ij',X,w,optimize=False);scores=local[ev['month']]
 z=np.where(ev['allowed'],scores[:,None,:],-1e30)
 lse=logsumexp(z,axis=-1);prob=np.exp(z-lse[:,:,None])*ev['allowed']
 eventlog=np.sum(ev['target']*scores,axis=1)-np.sum(lse*ev['stage'],axis=1)
 eventgrad=ev['target']-np.sum(prob*ev['stage'][:,:,None],axis=1)
 monthgrad=np.zeros_like(local);loss=0.
 for left,right in ev['slices']:
  lp=eventlog[left:right];logp=logsumexp(lp);post=np.exp(lp-logp)*(1-order_mass);post[0]+=order_mass
  loss-=(1-order_mass)*logp+order_mass*lp[0]
  monthgrad[ev['month'][left]]-=np.einsum('p,pn->n',post,eventgrad[left:right],optimize=False)
 n=len(ev['slices']);loss/=n;monthgrad/=n
 penalty=np.array([1 if g=='core' else side_penalty for g in groups])
 loss+=reg*np.sum(penalty*w*w)
 grad=np.einsum('ij,ijk->k',monthgrad,X,optimize=False)+2*reg*penalty*w
 return float(loss),grad

def fit(X,d,mask,groups,reg,order_mass,side_penalty):
 ev=event_design(d,mask);bounds=[(0,None) if g=='core' else (None,None) for g in groups]
 opt=minimize(objective,np.zeros(X.shape[-1]),args=(X,ev,groups,reg,side_penalty,order_mass),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=1000,ftol=1e-10,gtol=1e-6,maxls=40))
 assert opt.success,opt.message
 return opt.x

def train():
 assert not (P/'FROZEN.json').exists()
 meta,packs,states=m.load();rows=[]
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),change='Optimize exact top3 set likelihood by summing all6 order probabilities, compared with ordered likelihood. Same feature hypotheses, no inherited numeric budgets.',grid='8packs x3architectures x4regularizers x2order-mass x2sidepenalty x3retentions=1152 policies',training='first17/eval6,first23/eval12; refit35, evaluate34 only after freezing',retrospective='All69 refit is only explanatory matching, not validation',code_sha256=m.sha(__file__)))
 for packkey,arch in itertools.product(packs,['linear','regime','spline']):
  d=packs[packkey];X,names,groups=m.design(d,states,arch)
  for reg,order_mass,sidepenalty in itertools.product([.0001,.001,.01,.1],[0.,1.],[1.,16.]):
   summaries={b:[] for b in [3,5,7]};audits=[]
   for before,until in [(17,23),(23,35)]:
    mask=np.arange(69)<before;vm=(np.arange(69)>=before)&(np.arange(69)<until)
    w=fit(X,d,mask,groups,reg,order_mass,sidepenalty);scores=np.einsum('ijk,k->ij',X[:until],w,optimize=False)
    audits.append(m.contributions(X,w,groups,d,mask))
    for b in summaries:summaries[b].append(m.matching(m.choices(scores,d,b),meta['periods'],vm))
   for b,mm in summaries.items():
    rows.append(dict(id=len(rows),pack=packkey,arch=arch,reg=reg,order_mass=order_mass,side_penalty=sidepenalty,buffer=b,inner_validation={k:sum(z[k] for z in mm) for k in mm[0]},core_largest_inner=all(z['core_largest'] for z in audits)))
  dump('PROGRESS.json',dict(policies_completed=len(rows),last_pack=packkey,last_arch=arch))
  print('Completed',len(rows),packkey,arch,flush=True)
 dump('INNER_TRIALS.json',rows)
 categories={'all':(rows,'mixed'),'core_primary':([r for r in rows if r['core_largest_inner']],'mixed'),'ordered':([r for r in rows if r['order_mass']==1 and r['core_largest_inner']],'mixed'),'top32_core_primary':([r for r in rows if r['core_largest_inner'] and r['pack'].startswith('top32')],'mixed')}
 leaders={}
 for category,(rr,criterion) in categories.items():
  for row in sorted(rr,key=lambda z:m.key_for(z,criterion),reverse=True):
   d=packs[row['pack']];X,names,groups=m.design(d,states,row['arch']);mask=np.arange(69)<35
   w=fit(X,d,mask,groups,row['reg'],row['order_mass'],row['side_penalty']);audit=m.contributions(X,w,groups,d,mask)
   if category!='all' and not audit['core_largest']:continue
   leaders[category]=dict(**row,weights=w.tolist(),names=names,groups=groups,training_contribution_audit=audit);break
 assert len(leaders)==4
 dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders,code_sha256=m.sha(__file__),base_code_sha256=m.sha(m.__file__)))
 print('Frozen', {k:dict(pack=v['pack'],arch=v['arch'],inner=v['inner_validation']) for k,v in leaders.items()},flush=True)

def evaluate():
 meta,packs,states=m.load();frozen=m.read(P/'FROZEN.json');fh=m.sha(P/'FROZEN.json');results={}
 assert frozen['code_sha256']==m.sha(__file__) and frozen['base_code_sha256']==m.sha(m.__file__)
 assert all(m.sha(f)==h for f,h in m.read(m.P/'SOURCE_HASHES.json').items())
 _,_,px,opening,c0,_=m.s.load()
 for name,row in frozen['leaders'].items():
  d=packs[row['pack']];X,names,groups=m.design(d,states,row['arch']);w=np.array(row['weights']);local=np.einsum('ijk,k->ij',X,w,optimize=False)
  c=m.a.context_for(d,c0);scores=np.full(c['eligible'].shape,-np.inf)
  for i,ss in enumerate(d['symbols']):scores[i,[c['ids'][t] for t in ss]]=local[i,:len(ss)]
  cfg=dict(buffer=row['buffer'],gap=0.,exit=dict(name='none'));sim=m.policy.simulate(c,scores,np.ones_like(scores),cfg,detailed=True)
  chosen=m.choices(local,d,row['buffer']);assert chosen==[[c['symbols'][j] for j in ch] for ch in sim['choices']]
  met,slices,eq,trades=m.old.performance(px,opening,c,sim,scores,'close');os=m.policy.simulate(c,scores,np.ones_like(scores),cfg,execution='open');om,oss,_,_=m.old.performance(px,opening,c,os,scores,'open')
  mm={k:m.matching(chosen,meta['periods'],mask) for k,mask in m.old.masks(69).items() if k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=mm,rule=row,validation_contribution_audit=m.contributions(X,w,groups,d,np.arange(69)>=35))
  dump(name+'_TRADES.json',trades);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  details=[]
  for i,p in enumerate(meta['periods']):
   ss=d['symbols'][i];ranked=sorted(np.flatnonzero(d['eligible'][i]),key=lambda j:(-local[i,j],ss[j]));rank={ss[j]:k+1 for k,j in enumerate(ranked)};target=list(map(m.e.actual,p['holdings']));new=chosen[i];prior=chosen[i-1] if i else []
   records=[]
   for ticker in sorted(set(target)|set(new)):
    if ticker not in ss:records.append(dict(ticker=m.e.display(ticker),reason='outside_proxy_pool'));continue
    j=ss.index(ticker);components={g:float(sum(X[i,j,k]*w[k] for k,gg in enumerate(groups) if gg==g)) for g in set(groups)}
    records.append(dict(ticker=m.e.display(ticker),rank=rank.get(ticker),selected=ticker in new,author=ticker in target,components=components))
   details.append(dict(date=p['start'],end=p['end'],split='training' if i<35 else 'validation',author=p['holdings'],selected=list(map(m.e.display,new)),entered=list(map(m.e.display,sorted(set(new)-set(prior)))),exited=list(map(m.e.display,sorted(set(prior)-set(new)))),details=records))
  dump(name+'_MONTHS.json',details)
  w_all=fit(X,d,np.ones(69,bool),groups,row['reg'],row['order_mass'],row['side_penalty']);ch_all=m.choices(np.einsum('ijk,k->ij',X,w_all,optimize=False),d,row['buffer'])
  results[name]['retrospective_fit']=dict(matching=m.matching(ch_all,meta['periods'],np.ones(69,bool)),contributions=m.contributions(X,w_all,groups,d,np.ones(69,bool)),weights=w_all.tolist())
  print(name,dict(full_cagr=met['cagr'],full_sharpe=met['sharpe'],validation=slices['validation34'],matching=mm,retrospective=results[name]['retrospective_fit']['matching']),flush=True)
 prior=m.read(m.O/'monthly_style_inverse/RESULTS.json');results.update({k:prior[k] for k in ['author','qqq']});dump('RESULTS.json',results)
 assert m.sha(P/'FROZEN.json')==fh
 changes=[f for f,h in m.read(m.P/'PROTECTED_BEFORE.json').items() if m.sha(f)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=fh,independent_replay_verified=True,selection='first35_only',outer_reused_history=True,complete_reproduction=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','evaluate']);globals()[ap.parse_args().phase]()
