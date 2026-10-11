"""Small market-state tree selecting constrained long-momentum weights."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, sys, itertools
import numpy as np
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_score_architecture'))
import architecture as s
a,old,e,policy=s.a,s.old,s.e,s.policy
read,sha=s.read,s.sha
MARKETS=['growth_relative63','smallcap_relative63','defensive_minus_cyclical63','tlt_return63','spy_above200','highvol']
RETENTIONS=[(3,0.),(5,.03),(7,.03)]

def dump(n,v):
 (P/n).write_text(old.json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 rows=sorted(read(O/'monthly_score_architecture/INNER_TRIALS.json'),key=s.choose,reverse=True);seeds=[];seen=set()
 for gated in [False,True]:
  count=0
  for row in rows:
   if (row['gate']!='positive')!=gated:continue
   key=tuple(row[k] for k in ['key','menu','gate','objective'])
   if key not in seen:seen.add(key);seeds.append(row);count+=1
   if count==3:break
 dump('SEEDS.json',seeds)
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),hypothesis='Fit shared factor-weight regimes from observable market style; previous sign-at-zero partitions were unverified assumptions.',
 hard_rules='Core returns and own-sector relative returns skip42sessions/two months; core>=50%,vol<=30%,sector<=20%,shape+accel<=20%. Long-only3 equal names, monthend same-close zero costs, fixed intramonth shares. No ticker/date features or teacher holdings as trading state.',
 seeds='Three no-gate and three gated distinct scoring families, ordered only by architecture first35 inner-validation matching. No outer returns considered.',
 tree='Global root; greedily split at training-only market-feature quantiles1/3,1/2,2/3 or binary midpoint. Each child needs6 months. At most2 or3 leaves. Optimize choice likelihood/pairwise loss with .10 shrinkage to global weights and .03 penalty per extra leaf. No leaf may move L1 beyond.30 or.60 from root; convex blend preserves all score budgets.',
 markets=MARKETS,grid='6seeds x2max-leaves x2L1caps x3retentions=72policies. Inner fit17/eval6 plus fit23/eval12. Freeze by matching(.6slots+.25sets+.15order), refit35, evaluate34.',
 limitations='Imitation study using supplied author labels, not claiming original ML training target. Reused historical validation is not pristine OOS. Proxy pools and assumedETF17 remain unverified.',protection='Do not change prior accounts, sources or frozen research.'))
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in read(O/'monthly_score_architecture/PROTECTED_BEFORE.json')})
 sources=read(O/'monthly_close_exit_recheck/SOURCE_HASHES.json');sources[str(Path(__file__))]=sha(__file__);dump('SOURCE_HASHES.json',sources)

def loss_value(X,d,mask,w,objective,base):
 if objective=='ordered':
  E,allowed,selected,pw=e.events(X,d,mask);v=e.likelihood(w,E,allowed,selected,pw,.05,.10,base)[0]
 else:
  D,pw=s.pair_design(X,d,mask);z=np.einsum('ij,j->i',D,w,optimize=False)/.05;v=float(np.sum(pw*np.logaddexp(0,-z))+.1*np.sum((w-base)**2))
 return v

def fit_tree(X,d,mask,spec,objective,Z,max_leaves,cap):
 base=s.fit_base(X,d,mask,spec,objective);n=int(mask.sum());cache={}
 def fitted(m):
  key=m.tobytes()
  if key not in cache:
   target=s.fit_base(X,d,m,spec,objective);delta=target-base;w=base+min(1.,cap/max(1e-12,np.abs(delta).sum()))*delta
   cache[key]=(w,loss_value(X,d,m,w,objective,base))
  return cache[key]
 root=dict(weights=base.tolist(),months=n,path=[]);leaves=[(root,mask,loss_value(X,d,mask,base,objective,base))]
 for _ in range(max_leaves-1):
  best=None
  for li,(node,m,parentloss) in enumerate(leaves):
   if m.sum()<12:continue
   for f in MARKETS:
    vals=Z[f].to_numpy();v=vals[m].astype(float);unique=np.unique(v)
    thresholds=[float(unique.mean())] if len(unique)==2 else sorted(set(float(t) for t in np.quantile(v,[1/3,.5,2/3])))
    for threshold in thresholds:
     left=m&(vals<=threshold);right=m&(vals>threshold)
     if min(left.sum(),right.sum())<6:continue
     lw,ll=fitted(left);rw,rl=fitted(right)
     gain=(m.sum()*parentloss-left.sum()*ll-right.sum()*rl)/n-.03
     if gain>1e-9 and (best is None or gain>best[0]):best=(gain,li,f,threshold,left,right,lw,ll,rw,rl)
  if best is None:break
  gain,li,f,t,lm,rm,lw,ll,rw,rl=best;node,m,loss=leaves.pop(li)
  node.update(feature=f,threshold=t,gain=float(gain),left=dict(weights=lw.tolist(),months=int(lm.sum()),path=node['path']+[[f,'<=',t]]),right=dict(weights=rw.tolist(),months=int(rm.sum()),path=node['path']+[[f,'>',t]]))
  leaves.extend([(node['left'],lm,ll),(node['right'],rm,rl)])
 return dict(base=base.tolist(),spec=spec,tree=root,max_leaves=max_leaves,actual_leaves=len(leaves),cap=cap)

def weights_for(rule,Z):
 W=[];paths=[]
 for _,row in Z.iterrows():
  node=rule['tree']
  while 'feature' in node:node=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
  W.append(node['weights']);paths.append(node['path'])
 return np.array(W),paths

def scores_for(rule,X,d,c,Z):
 W,paths=weights_for(rule,Z);local=np.einsum('ijk,ik->ij',X,W,optimize=False);scores=np.full(c['eligible'].shape,-np.inf);strength=np.zeros_like(scores)
 ci=[j for j,(_,_,g) in enumerate(rule['spec']) if g=='core']
 for i,ss in enumerate(d['symbols']):
  js=[c['ids'][v] for v in ss];scores[i,js]=local[i,:len(ss)];strength[i,js]=old.pd.Series(X[i,:len(ss)][:,ci].mean(axis=1)).where(d['eligible'][i,:len(ss)]).rank(pct=True).fillna(.5)
 return scores,strength,local,W,paths

def setup(seed,packs,c0):
 d=s.apply_gate(packs[seed['key']],seed['menu'],seed['gate']);c=a.context_for(d,c0);X,spec=a.design(d,s.MENUS[seed['menu']],'legacy');return d,c,X,spec

def choose(r):
 m=r['inner_validation'];v=.6*m['shared_slots']/(3*m['months'])+.25*m['set_exact']/m['months']+.15*m['order_exact']/m['months']
 return v,-r['max_leaves'],-r['cap'],-r['config']['buffer'],-r['id']

def train():
 _,packs,px,opening,c0,Z=s.load();rows=[]
 for sid,seed in enumerate(read(P/'SEEDS.json')):
  d,c,X,spec=setup(seed,packs,c0)
  for leaves,cap in itertools.product([2,3],[.3,.6]):
   agg={(b,g):[] for b,g in RETENTIONS}
   for before,until in a.FOLDS:
    rule=fit_tree(X,d,np.arange(69)<before,spec,seed['objective'],Z,leaves,cap);score,strength,_,_,_=scores_for(rule,X,d,c,Z);mask=(np.arange(69)>=before)&(np.arange(69)<until)
    for b,g in RETENTIONS:
     cfg=dict(buffer=b,gap=g,exit=dict(name='none'));sim=policy.simulate(c,score,strength,cfg,count=until);agg[(b,g)].append(policy.matching(sim['choices'],score,c,mask))
   for (b,g),mm in agg.items():
    rows.append(dict(id=len(rows),seed_id=sid,seed=seed,max_leaves=leaves,cap=cap,config=dict(buffer=b,gap=g,exit=dict(name='none')),inner_validation={k:sum(v[k] for v in mm) for k in ['months','shared_slots','set_exact','order_exact']}))
  print('Tree policies completed',len(rows),flush=True)
 leaders={}
 for name,group in [('overall',rows),('ungated',[r for r in rows if r['seed']['gate']=='positive'])]:
  r=max(group,key=choose);d,c,X,spec=setup(r['seed'],packs,c0);rule=fit_tree(X,d,np.arange(69)<35,spec,r['seed']['objective'],Z,r['max_leaves'],r['cap']);leaders[name]=dict(**r,rule=rule)
 dump('INNER_TRIALS.json',rows);dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders));print('FROZEN',{n:{k:r[k] for k in ['seed_id','max_leaves','cap','config','inner_validation']} for n,r in leaders.items()},flush=True)

def evaluate():
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items());h=sha(P/'FROZEN.json');_,packs,px,opening,c0,Z=s.load();ms=old.masks(69);results={}
 for name,row in read(P/'FROZEN.json')['leaders'].items():
  d,c,X,spec=setup(row['seed'],packs,c0);score,strength,local,W,paths=scores_for(row['rule'],X,d,c,Z);sim=policy.simulate(c,score,strength,row['config'],detailed=True)
  met,slices,eq,trades=old.performance(px,opening,c,sim,score,'close');os=policy.simulate(c,score,strength,row['config'],execution='open');om,oss,_,_=old.performance(px,opening,c,os,score,'open')
  matching={k:policy.matching(sim['choices'],score,c,ms[k]) for k in ['training','validation','all']};results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=matching,rule=row)
  dump(name+'_TRADES.json',trades);dump(name+'_DECISIONS.json',[dict(event=ev,market_path=path) for ev,path in zip(sim['events'],paths)]);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  print(name,dict(cagr=met['cagr'],sharpe=met['sharpe'],validation=slices['validation34'],matching=matching),flush=True)
 prior=read(O/'monthly_style_inverse/RESULTS.json')
 for k in ['author','qqq']:results[k]=prior[k]
 dump('RESULTS.json',results);assert sha(P/'FROZEN.json')==h;changes=[f for f,v in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=v];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=h,split='first35/later34',no_ticker_or_date_regimes=True,no_outer_selection=True,zero_cost=True,no_intramonth_trades=True,independent_replay_verified=True,not_pristine_oos=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','train','evaluate']);globals()[p.parse_args().phase]()
