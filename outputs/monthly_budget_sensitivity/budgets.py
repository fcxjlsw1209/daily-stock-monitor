"""Audit research-imposed weight budgets; do not alter frozen paper strategies."""
from pathlib import Path
from datetime import datetime,timezone
import sys,argparse,itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_score_architecture'))
import architecture as s
a,old,e,policy=s.a,s.old,s.e,s.policy
read,sha=s.read,s.sha
MODES=['strict','separate_acceleration','core50','dominant40']
RETENTIONS=[(3,0.),(5,.03),(7,.03)]

def dump(n,v):
 (P/n).write_text(old.json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def limits(spec,mode):
 if mode=='strict':return e.limits(spec)
 groups=np.array([g for k,v,g in spec]);A=[];b=[]
 def cap(gs,value):A.append(np.array([g in gs for g in groups],float));b.append(value)
 cap({'shape'},.20);cap({'ml'},.10);cap({'accel'},.20)
 if mode=='separate_acceleration':cap({'vol'},.30);cap({'industry'},.20)
 A.append(-(groups=='core').astype(float));b.append(-(.4 if mode=='dominant40' else .5))
 if mode=='dominant40':
  A.append(-np.array([g in ['core','accel'] for g in groups],float));b.append(-.5)
  for group in ['vol','industry']:
   A.append((groups==group).astype(float)-(groups=='core').astype(float));b.append(0.)
 prior=e.limits(spec)[2]
 return np.array(A),np.array(b),prior

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('SEEDS.json',read(O/'monthly_regime_tree_inverse/SEEDS.json'))
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),hypothesis='Old shape+lagged-acceleration<=20% was an invented research constraint, not disclosed guidance. Test separately capped acceleration and less restrictive support budgets.',
 modes=dict(strict='prior budgets unchanged',separate_acceleration='core>=50%,vol<=30%,industry<=20%,shape<=20%,acceleration<=20%,ML<=10%; no shape+accel aggregate cap',core50='core>=50%,shape<=20%,acceleration<=20%,ML<=10%; remaining volatility/sector weights share remaining<=50%',dominant40='explicit assumption sensitivity: long core>=40%,long core+lagged acceleration>=50%,core weight>=volatility and own-sector groups; shape20%,accel20%,ML10%'),
 hard_rules='Monthend fixed3 equal shares, no intramonth trades, zero costs. Core/own-sector returns and acceleration exclude42sessions/two calendar months. No additional features or stock/date labels. No observer/account migration.',
 grid='6inner-only families x4budget modes x2losses x2states x3retentions=288policies. Fit17/eval6,fit23/eval12. Only first35 choose. Refit35/freeze then later34. Select matching, never returns.',
 caveat='No numeric original weight limits known. dominant40 is explicitly broader than earlier >=50% long-core convention, separately reported. All historical validation has previously been seen, not pristine OOS.'))
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in read(O/'monthly_score_architecture/PROTECTED_BEFORE.json')})
 sources=read(O/'monthly_close_exit_recheck/SOURCE_HASHES.json');sources[str(Path(__file__))]=sha(__file__);dump('SOURCE_HASHES.json',sources)

def fit_base(X,d,mask,spec,objective,mode):
 A,b,prior=limits(spec,mode)
 if objective=='ordered':
  E,allowed,selected,pw=e.events(X,d,mask)
  def loss(w):return e.likelihood(w,E,allowed,selected,pw,.05,.10,prior)
 else:
  D,pw=s.pair_design(X,d,mask)
  def loss(w):
   z=np.einsum('ij,j->i',D,w,optimize=False)/.05
   return float(np.sum(pw*np.logaddexp(0,-z))+.1*np.sum((w-prior)**2)),-np.einsum('i,ij->j',pw*expit(-z),D,optimize=False)/.05+.2*(w-prior)
 opt=minimize(loss,prior,jac=True,method='SLSQP',bounds=[(0,1)]*len(spec),constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(len(w))),dict(type='ineq',fun=lambda w:b-np.einsum('ij,j->i',A,w,optimize=False),jac=lambda w:-A)],options=dict(maxiter=500,ftol=1e-10))
 assert opt.success,opt.message;assert np.max(np.einsum('ij,j->i',A,opt.x,optimize=False)-b)<1e-7
 return opt.x

def fit_rule(X,d,mask,spec,objective,mode,groups,base=None):
 base=fit_base(X,d,mask,spec,objective,mode) if base is None else base;weights={}
 for g in sorted(set(groups)):
  gm=mask&(groups==g)
  if len(set(groups))==1 or gm.sum()<8:w=base.copy()
  else:
   target=fit_base(X,d,gm,spec,objective,mode);delta=target-base;w=base+min(1.,.3/max(1e-12,np.abs(delta).sum()))*delta
  weights[str(g)]=w.tolist()
 return dict(spec=spec,base=base.tolist(),weights=weights,budget=mode)

def setup(seed,packs,c0):
 d=s.apply_gate(packs[seed['key']],seed['menu'],seed['gate']);c=a.context_for(d,c0);X,spec=a.design(d,s.MENUS[seed['menu']],'legacy');return d,c,X,spec

def choose(r):
 m=r['inner_validation'];v=.6*m['shared_slots']/(3*m['months'])+.25*m['set_exact']/m['months']+.15*m['order_exact']/m['months']
 return v,-(r['mode']!='strict'),-(r['mode']=='dominant40'),-(r['partition']!='global'),-r['config']['buffer'],-r['id']

def train():
 _,packs,px,opening,c0,Z=s.load();rows=[]
 for sid,seed in enumerate(read(P/'SEEDS.json')):
  d,c,X,spec=setup(seed,packs,c0)
  for mode,objective in itertools.product(MODES,s.OBJECTIVES):
   agg={(part,b,g):[] for part in ['global','growth'] for b,g in RETENTIONS}
   for before,until in a.FOLDS:
    mask=np.arange(69)<before;val=(np.arange(69)>=before)&(np.arange(69)<until);base=fit_base(X,d,mask,spec,objective,mode)
    for part in ['global','growth']:
     groups=Z[part].to_numpy();rule=fit_rule(X,d,mask,spec,objective,mode,groups,base);score,strength,_,_=a.scores_for(rule,X,spec,d,c,groups)
     for b,g in RETENTIONS:
      cfg=dict(buffer=b,gap=g,exit=dict(name='none'));sim=policy.simulate(c,score,strength,cfg,count=until);agg[(part,b,g)].append(policy.matching(sim['choices'],score,c,val))
   for (part,b,g),mm in agg.items():rows.append(dict(id=len(rows),seed_id=sid,seed=seed,mode=mode,objective=objective,partition=part,config=dict(buffer=b,gap=g,exit=dict(name='none')),inner_validation={k:sum(v[k] for v in mm) for k in ['months','shared_slots','set_exact','order_exact']}))
  print('Budget policies completed',len(rows),flush=True)
 leaders={}
 for name,group in [('overall',rows),('core50',[r for r in rows if r['mode']!='dominant40']),('strict',[r for r in rows if r['mode']=='strict']),('dominant40',[r for r in rows if r['mode']=='dominant40'])]:
  row=max(group,key=choose);d,c,X,spec=setup(row['seed'],packs,c0);rule=fit_rule(X,d,np.arange(69)<35,spec,row['objective'],row['mode'],Z[row['partition']].to_numpy());leaders[name]=dict(**row,rule=rule)
 dump('INNER_TRIALS.json',rows);dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders));print('FROZEN',{n:{k:r[k] for k in ['seed_id','mode','objective','partition','config','inner_validation']} for n,r in leaders.items()},flush=True)

def evaluate():
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items());h=sha(P/'FROZEN.json');_,packs,px,opening,c0,Z=s.load();ms=old.masks(69);results={}
 for name,row in read(P/'FROZEN.json')['leaders'].items():
  d,c,X,spec=setup(row['seed'],packs,c0);score,strength,local,W=a.scores_for(row['rule'],X,spec,d,c,Z[row['partition']].to_numpy());sim=policy.simulate(c,score,strength,row['config'],detailed=True)
  met,slices,eq,trades=old.performance(px,opening,c,sim,score,'close');os=policy.simulate(c,score,strength,row['config'],execution='open');om,oss,_,_=old.performance(px,opening,c,os,score,'open');matching={k:policy.matching(sim['choices'],score,c,ms[k]) for k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=matching,rule=row);dump(name+'_TRADES.json',trades);dump(name+'_DECISIONS.json',sim['events']);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  print(name,dict(cagr=met['cagr'],sharpe=met['sharpe'],validation=slices['validation34'],matching=matching),flush=True)
 prior=read(O/'monthly_style_inverse/RESULTS.json')
 for k in ['author','qqq']:results[k]=prior[k]
 dump('RESULTS.json',results);assert sha(P/'FROZEN.json')==h;changes=[f for f,v in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=v];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=h,split='first35/later34',zero_cost=True,no_intramonth_trades=True,independent_replay_verified=True,not_pristine_oos=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','train','evaluate']);globals()[p.parse_args().phase]()
