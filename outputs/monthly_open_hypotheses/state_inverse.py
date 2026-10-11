"""Autonomous monthly policy imitation with observable position state.

Historical author states can seed training prefixes only. Evaluation starts in
cash and computes all costs, peaks and holding ages from the model's own trades.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,json
import numpy as np
import open_inverse as m
P=Path(__file__).resolve().parent/'stateful';P.mkdir(exist_ok=True)
STATE_NAMES=['held','profit_sigma','drawdown_sigma','holding_age','profit_weak_market','drawdown_highvol','entry_signal_change','loss_weak_signal']

def dump(n,z):
 (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def state_row(c,i,ss,state,core,entry_scores):
 ans=np.zeros((len(ss),len(STATE_NAMES)))
 for k,t in enumerate(ss):
  j=c['ids'][t]
  if j not in state:continue
  h=state[j];price=c['close'][i,j];peak=max(h['peak'],c['peak'][i,j]);sigma=max(float(c['sigma'][i,j]),.01)
  profit=price/h['cost']-1;drawdown=price/peak-1
  values=[1.,np.tanh(profit/(2*sigma)),np.tanh(drawdown/(2*sigma)),min((i-entry_scores[j][0])/12,1),np.tanh(profit/(2*sigma))*(1-c['trend'][i]),np.tanh(drawdown/(2*sigma))*c['highvol'][i],core[k]-entry_scores[j][1],np.tanh(min(profit,0)/(2*sigma))*(1-(core[k]+.5))]
  ans[k]=np.array(values)*.5
 return ans

def rollout(c,d,X,groups,w=None,teacher_until=None,count=None,execution='close'):
 n=len(c['dates']) if count is None else count;state={};entries={};cash=100000.;choices=[];events=[];S=np.zeros((len(X),X.shape[1],len(STATE_NAMES)));local=np.zeros(X.shape[:2]);global_scores=np.full(c['eligible'].shape,-np.inf)
 ci=[k for k,g in enumerate(groups) if g=='core']
 for i in range(n):
  ss=d['symbols'][i];core=X[i,:len(ss)][:,ci].mean(axis=1)
  for j,h in state.items():
   assert np.isfinite(c['close'][i,j]) and np.isfinite(c['peak'][i,j])
   h['peak']=max(h['peak'],float(c['peak'][i,j]))
  S[i,:len(ss)]=state_row(c,i,ss,state,core,entries)
  if w is not None:
   local[i]=np.einsum('jk,k->j',np.concatenate([X[i],S[i]],axis=-1),w,optimize=False)
   js=[c['ids'][t] for t in ss];global_scores[i,js]=local[i,:len(ss)]
  if teacher_until is not None:
   assert i<teacher_until,'No author state may be consumed after training prefix'
   selected=[c['ids'][m.e.actual(t)] for t in c['periods'][i]['holdings']]
  else:
   selected=sorted(np.flatnonzero(c['eligible'][i]),key=lambda j:(-global_scores[i,j],c['symbols'][j]))[:3]
   if len(selected)<2:selected=[]
  before=dict(state);fill=c['close'][i] if execution=='close' else c['opening'][i]
  assert all(np.isfinite(fill[j]) and fill[j]>0 for j in set(state)|set(selected))
  nav=cash+sum(h['units']*fill[j] for j,h in state.items());assert nav>0
  date=c['dates'][i] if execution=='close' else c['opening_dates'][i]
  state=m.policy.rebalance_state(state,selected,fill,nav,date,c['peak'][i]);cash=0. if state else nav
  new_entries={}
  for j in selected:
   t=c['symbols'][j];currentcore=core[ss.index(t)] if t in ss else 0.
   new_entries[j]=entries[j] if j in entries else (i,float(currentcore))
  entries=new_entries;choices.append(selected)
  events.append(dict(date=c['dates'][i],chosen=[m.e.display(c['symbols'][j]) for j in selected],before={m.e.display(c['symbols'][j]):h for j,h in before.items()}))
 return dict(choices=choices,events=events,state=state,cash=cash),S,local,global_scores

def fit_policy(X,d,c,groups,mask,reg,order_mass,side_penalty,iterations):
 before=int(mask.sum());_,S,_,_=rollout(c,d,X,groups,teacher_until=before,count=before)
 allgroups=groups+['management']*len(STATE_NAMES)
 for iteration in range(iterations):
  combined=np.concatenate([X,S],axis=-1)
  w=m.fit(combined,d,mask,allgroups,reg,order_mass,side_penalty)
  if iteration+1<iterations:_,S,_,_=rollout(c,d,X,groups,w=w,count=before)
 return w

def train():
 assert not (P/'FROZEN.json').exists()
 meta,packs,states=m.load();_,_,px,opening,c0,_=m.s.load();rows=[]
 keys=[k for k in packs if k.endswith('zclip')]
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),hypothesis='Monthend exit/retain/selection can depend on existing position P&L, trailing peak, age and signal change. No intramonth stop or trading.',training='Initial training state uses previous author holdings inside prefix17/23/35 only. Optional3 imitation iterations regenerate all positions with model then refit. Evaluation autonomously starts cash, never inserts author positions.',features=STATE_NAMES,grid='4pools x2architectures x3regularizers x2order masses x2side penalties x2iterations=192 policies',selection='First17/eval6 and first23/eval12 holdings; first35 refit, freeze then evaluate34. Reused historical validation. No return selection.',code_sha256=m.sha(__file__)))
 for key,arch in itertools.product(keys,['linear','regime']):
  d=packs[key];c=m.a.context_for(d,c0);X,names,groups=m.design(d,states,arch)
  for reg,order_mass,sidepenalty,iterations in itertools.product([.001,.01,.1],[0.,.5],[1.,16.],[1,3]):
   mm=[];audits=[]
   for before,until in [(17,23),(23,35)]:
    mask=np.arange(69)<before;vm=(np.arange(69)>=before)&(np.arange(69)<until)
    w=fit_policy(X,d,c,groups,mask,reg,order_mass,sidepenalty,iterations)
    sim,S,_,_=rollout(c,d,X,groups,w=w,count=until)
    chosen=[[c['symbols'][j] for j in ch] for ch in sim['choices']];mm.append(m.matching(chosen,meta['periods'],vm))
    audits.append(m.contributions(np.concatenate([X,S],axis=-1),w,groups+['management']*len(STATE_NAMES),d,mask))
   rows.append(dict(id=len(rows),pack=key,arch=arch,reg=reg,order_mass=order_mass,side_penalty=sidepenalty,iterations=iterations,buffer=3,inner_validation={k:sum(z[k] for z in mm) for k in mm[0]},core_largest_inner=all(z['core_largest'] for z in audits)))
  print('Completed',len(rows),key,arch,flush=True)
 dump('INNER_TRIALS.json',rows);leaders={}
 for category,rr in [('all',rows),('core_primary',[r for r in rows if r['core_largest_inner']]),('top32_core_primary',[r for r in rows if r['core_largest_inner'] and r['pack'].startswith('top32')])]:
  for row in sorted(rr,key=m.key_for,reverse=True):
   d=packs[row['pack']];c=m.a.context_for(d,c0);X,names,groups=m.design(d,states,row['arch']);mask=np.arange(69)<35
   w=fit_policy(X,d,c,groups,mask,row['reg'],row['order_mass'],row['side_penalty'],row['iterations']);sim,S,_,_=rollout(c,d,X,groups,w=w,count=35)
   audit=m.contributions(np.concatenate([X,S],axis=-1),w,groups+['management']*len(STATE_NAMES),d,mask)
   if category!='all' and not audit['core_largest']:continue
   leaders[category]=dict(**row,weights=w.tolist(),names=names+STATE_NAMES,groups=groups+['management']*len(STATE_NAMES),training_contribution_audit=audit);break
 assert len(leaders)==3
 dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders,code_sha256=m.sha(__file__),base_code_sha256=m.sha(m.__file__)))
 print('Frozen',{k:dict(pack=v['pack'],arch=v['arch'],inner=v['inner_validation'],iterations=v['iterations']) for k,v in leaders.items()},flush=True)

def evaluate():
 meta,packs,states=m.load();_,_,px,opening,c0,_=m.s.load();frozen=m.read(P/'FROZEN.json');fh=m.sha(P/'FROZEN.json');results={}
 assert frozen['code_sha256']==m.sha(__file__) and frozen['base_code_sha256']==m.sha(m.__file__)
 assert all(m.sha(f)==h for f,h in m.read(m.P/'SOURCE_HASHES.json').items())
 for name,row in frozen['leaders'].items():
  d=packs[row['pack']];c=m.a.context_for(d,c0);X,names,groups=m.design(d,states,row['arch']);w=np.array(row['weights'])
  sim,S,local,scores=rollout(c,d,X,groups,w=w);chosen=[[c['symbols'][j] for j in ch] for ch in sim['choices']]
  met,slices,eq,trades=m.old.performance(px,opening,c,sim,scores,'close');os,_,_,osc=rollout(c,d,X,groups,w=w,execution='open');om,oss,_,_=m.old.performance(px,opening,c,os,osc,'open')
  mm={k:m.matching(chosen,meta['periods'],mask) for k,mask in m.old.masks(69).items() if k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=mm,rule=row,validation_contribution_audit=m.contributions(np.concatenate([X,S],axis=-1),w,groups+['management']*len(STATE_NAMES),d,np.arange(69)>=35))
  dump(name+'_TRADES.json',trades);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]);dump(name+'_STATES.json',sim['events'])
  details=[];G=groups+['management']*len(STATE_NAMES);XX=np.concatenate([X,S],axis=-1)
  for i,p in enumerate(meta['periods']):
   ss=d['symbols'][i];ranked=sorted(np.flatnonzero(d['eligible'][i]),key=lambda j:(-local[i,j],ss[j]));rank={ss[j]:k+1 for k,j in enumerate(ranked)};target=list(map(m.e.actual,p['holdings']));new=chosen[i];records=[]
   for ticker in sorted(set(target)|set(new)):
    if ticker not in ss:records.append(dict(ticker=m.e.display(ticker),reason='outside_proxy_pool'));continue
    j=ss.index(ticker);records.append(dict(ticker=m.e.display(ticker),rank=rank.get(ticker),selected=ticker in new,author=ticker in target,components={g:float(sum(XX[i,j,k]*w[k] for k,gg in enumerate(G) if gg==g)) for g in set(G)}))
   details.append(dict(date=p['start'],end=p['end'],split='training' if i<35 else 'validation',author=p['holdings'],selected=list(map(m.e.display,new)),details=records))
  dump(name+'_MONTHS.json',details)
  print(name,dict(full_cagr=met['cagr'],full_sharpe=met['sharpe'],validation=slices['validation34'],matching=mm),flush=True)
 prior=m.read(m.O/'monthly_style_inverse/RESULTS.json');results.update({k:prior[k] for k in ['author','qqq']});dump('RESULTS.json',results)
 assert m.sha(P/'FROZEN.json')==fh
 changes=[f for f,h in m.read(m.P/'PROTECTED_BEFORE.json').items() if m.sha(f)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=fh,independent_replay_verified=True,autonomous_state=True,teacher_labels_train_prefix_only=True,selection='first35_only',outer_reused_history=True,complete_reproduction=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','evaluate']);globals()[ap.parse_args().phase]()
