"""Author-guided inverse ranking, without inherited numerical factor budgets.

This learns explanatory scoring coefficients from supplied holdings, not market
return prediction coefficients. All-history fitting is explicitly retrospective.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, itertools, json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
P=Path(__file__).resolve().parent
O=P.parent
sys.path.insert(0,str(O/'monthly_score_architecture'))
import architecture as s
a,old,e,policy,strict=s.a,s.old,s.e,s.policy,s.strict
read,sha=s.read,s.sha

def dump(name,obj):
 (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else z)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
 immutable='Supplied holdings, monthly fixed-share equal-weight execution and author guidance. Core momentum excludes last42 sessions or2 calendar months. Zero costs for reproduction.',
 relaxed='No50/30/20/10 factor budgets, no sum-to-one, no fixed295-price warmup or positive momentum gate, no requirement that industry-relative performance excludes recent2months. Coefficients are common across securities; no ticker/date/month-label features.',
 hypotheses='Momentum times volatility powers -1,-.5,0,.5,1; raw/log and calendar/trading anchors; stock versus own sector ETF at21/63/126/252 and skip42 horizons; risk, volume and price-path corrections; observable regimes and nonlinear monotonic core bases.',
 pool='Original proxyTop32 and previously approvedTop35 boundary control; theme17 and core17 ETF assumptions. Top35 is a sensitivity, not a claim that author used35. Original17 list and exact cap reconstruction remain unknown.',
 models='Nonnegative monotonic core coefficients, signed side factors. Pairwise Ridge margin loss; data-fitting inverse coefficients are not claimed to be original annual five-year market-return Ridge. No new ML-return forecasts this experiment.',
 selection='Inner first17/eval6 and first23/eval12. First35 refit, later34 frozen evaluation. Select by held-out holdings, not returns. Latter34 previously viewed; not pristine OOS. Show sets and list order separately. Also all69 refit solely for explanatory ceiling.',
 guidance_check='No arbitrary numeric weight cap; audit core contribution cross-sectional dispersion versus each risk/industry/shape group. Candidate guide screen requires core to be largest on training. This operational screen is an assumption, not author-disclosed.',
 grid='4pool/ETF x2normalizations x3architectures x4ridge strengths x3order masses x2side penalties x3retentions=1728 policies; shared fits reused.',
 protections='New research directory only. No modifications to previous results, source holdings, prices, monitors, plans or ledgers.'))
 protected=read(O/'monthly_adaptive_core/PROTECTED_BEFORE.json')
 protected.update({str(O/'monthly_supplied_ledger_audit'/n):None for n in ['SOURCE.txt','PERIODS.json']})
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in protected})

def feature_bank(px,raw):
 r=px.pct_change(fill_method=None);v=r.rolling(63).std().replace(0,np.nan)
 core,side=e.features(px,raw);f={};group={}
 for n in [252,294]:
  m=px.shift(42)/px.shift(n)-1
  for kind,z in [('raw',m),('log',np.log1p(m))]:
   for power in [-1.,-.5,0.,.5,1.]:
    name=f'core_{n}_{kind}_v{power:g}';f[name]=z*(v/.02).pow(power);group[name]='core'
  for suffix in ['mean_sharpe','trend_quality']:
   name=f'core_{suffix}_{n}';f[name]=core[f'{suffix}{n}_skip42'];group[name]='core'
 for n in [12,14]:
  m=e.previous.calendar_momentum(px,n)
  for power in [-.5,0.,.5]:
   name=f'core_calendar{n}_v{power:g}';f[name]=m*(v/.02).pow(power);group[name]='core'
 for k in ['vol21','vol63','vol126','downside63','idio63','beta63','vol_ratio21_126']:
  f[k]=side[k];group[k]='risk'
 for k in ['cmf21','cmf63','signed_volume63','volume_ratio21_126','dollar_volume_ratio21_126','close_location21','high252_distance','jump_concentration63','acceleration_skip42']:
  f[k]=side[k];group[k]='shape'
 relative={f'own_sector_{n}':px/px.shift(n)-1 for n in [21,63,126,252]}
 relative.update({f'own_sector_skip42_{n}':px.shift(42)/px.shift(n)-1 for n in [126,252]})
 group.update({k:'industry' for k in relative})
 return f,relative,group

def normalize(tab,mode):
 if mode=='rank':return tab.rank(pct=True).fillna(.5)-.5
 scale=tab.std(ddof=0).replace(0,np.nan)
 return ((tab-tab.mean())/scale).clip(-3,3).fillna(0)/3

def prepare():
 px,raw,members,periods,sources,pools=e.load_prices();f,relative,groups=feature_bank(px,raw)
 sectors_file=O/'index_union/alias_resolved_inputs/sectors.csv.gz'
 sectors=pd.read_csv(sectors_file,index_col=0,parse_dates=True,low_memory=False);sources[str(sectors_file)]=sha(sectors_file)
 dates=[p['start'] for p in periods];styles=old.market_styles(px,dates)
 names=list(f)+list(relative);meta=dict(periods=periods,packs={},groups=groups)
 for pool,etf,norm in itertools.product([32,35],['theme17','core17'],['rank','zclip']):
  symbols=[sorted({e.actual(x) for x in pools[str(pool)][date]}|members[etf][1]) for date in dates]
  X=np.zeros((69,max(map(len,symbols)),len(names)));good=np.zeros(X.shape[:2],bool);labels=np.zeros_like(good);order=np.full((69,3),-1,int)
  coverage=[]
  for i,(date,ss) in enumerate(zip(dates,symbols)):
   bm=strict.benchmark_map(date,ss,sectors,members[etf][1]);tab=pd.DataFrame({k:z.loc[date,ss] for k,z in f.items()})
   for k,z in relative.items():tab[k]=strict.relative_row(z,date,ss,bm)
   # Warmup is defined by actually observed long features, not a hard-coded
   # continuous295-bar requirement. No future data-availability eligibility.
   valid=tab[[k for k in names if groups[k]=='core']].notna().all(axis=1)&px.loc[date,ss].gt(0)
   q=normalize(tab.where(valid,axis=0),norm);X[i,:len(ss)]=q[names];good[i,:len(ss)]=valid
   for j,ticker in enumerate(periods[i]['holdings']):
    t=e.actual(ticker)
    if t in ss:
     k=ss.index(t);labels[i,k]=True
     if good[i,k]:order[i,j]=k
   coverage.append(dict(date=date,unavailable=[t for j,t in enumerate(periods[i]['holdings']) if order[i,j]<0]))
  key=f'top{pool}_{etf}_{norm}'
  np.savez_compressed(P/(key+'.npz'),X=X,eligible=good,labels=labels,order=order)
  meta['packs'][key]=dict(symbols=symbols,names=names,groups=[groups[k] for k in names],pool=pool,etf=etf,norm=norm,coverage=coverage)
 # State covariates known by signal close, never full-period normalized.
 state=np.column_stack([styles.spy_above200.astype(float),styles.highvol.astype(float),
                       (1+np.tanh(styles.growth_relative63.to_numpy()/.08))/2,
                       (1+np.tanh(styles.smallcap_relative63.to_numpy()/.08))/2])
 np.savez_compressed(P/'STATES.npz',states=state)
 meta['state_names']=['spy_above200','highvol','growth_relative63','smallcap_relative63']
 dump('INPUTS.json',meta)
 for mod in [s,a,old,e,policy,strict]:sources[str(Path(mod.__file__))]=sha(mod.__file__)
 dump('SOURCE_HASHES.json',sources)
 print('Prepared',len(meta['packs']),'packs,',len(names),'base features',flush=True)

def load():
 meta=read(P/'INPUTS.json');packs={k:dict(np.load(P/(k+'.npz')),**v) for k,v in meta['packs'].items()}
 states=np.load(P/'STATES.npz')['states']
 return meta,packs,states

def design(d,states,arch):
 names=list(d['names']);groups=list(d['groups']);X=d['X'].copy()
 if arch=='regime':
  # Nonnegative state*core coefficients preserve monotonicity in core.
  ks=['core_252_raw_v0','core_294_log_v0','core_calendar14_v-0.5','vol63','downside63','own_sector_63','own_sector_skip42_252','cmf21','cmf63']
  extra=[]
  for state in range(states.shape[1]):
   for k in ks:
    j=d['names'].index(k);extra.append(d['X'][:,:,j]*states[:,state,None]);names.append(f'{k}__state{state}');groups.append(d['groups'][j])
  X=np.concatenate([X,np.stack(extra,axis=-1)],axis=-1)
 elif arch=='spline':
  ks=['core_252_raw_v0','core_294_log_v0','core_calendar14_v-0.5','vol63','downside63','own_sector_63','own_sector_skip42_252','cmf21','cmf63']
  extra=[]
  for k in ks:
   j=d['names'].index(k)
   for knot in [-.25,0.,.25]:
    extra.append(np.maximum(d['X'][:,:,j]-knot,0));names.append(f'{k}__hinge{knot}');groups.append(d['groups'][j])
  X=np.concatenate([X,np.stack(extra,axis=-1)],axis=-1)
 return X,names,groups

def pairs(X,d,mask,order_mass):
 diffs=[];weights=[]
 for i in np.flatnonzero(mask):
  selected=[int(k) for k in d['order'][i] if k>=0]
  other=np.flatnonzero(d['eligible'][i]&~d['labels'][i])
  for seq,mass in [([(u,int(v)) for u in selected for v in other],1-order_mass), (list(itertools.combinations(selected,2)),order_mass)]:
   if mass==0 or not seq:continue
   diffs.extend([X[i,u]-X[i,v] for u,v in seq]);weights.extend([mass/len(seq)]*len(seq))
 w=np.array(weights);w/=w.sum()
 return np.stack(diffs),w

def fit(X,d,mask,groups,reg,order_mass,side_penalty):
 D,pw=pairs(X,d,mask,order_mass)
 # Quadratic form reuses Gram matrix, avoiding costly repeated event products.
 gram=np.einsum('ni,n,nj->ij',D,pw,D,optimize=False)
 rhs=np.einsum('ni,n->i',D,pw,optimize=False)
 penalty=np.array([1. if g=='core' else side_penalty for g in groups])
 gram+=reg*np.diag(penalty)
 def fun(w):
  gw=np.einsum('ij,j->i',gram,w,optimize=False)
  return float(np.dot(w,gw)-2*np.dot(w,rhs)),2*(gw-rhs)
 bounds=[(0,None) if g=='core' else (None,None) for g in groups]
 opt=minimize(fun,np.zeros(X.shape[-1]),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=40))
 assert opt.success,opt.message
 return opt.x

def contributions(X,w,groups,d,mask):
 result={}
 for g in sorted(set(groups)):
  ii=[j for j,v in enumerate(groups) if v==g]
  val=np.einsum('ijk,k->ij',X[:,:,ii],w[ii],optimize=False)
  result[g]=float(np.mean([np.std(val[i,d['eligible'][i]]) for i in np.flatnonzero(mask)]))
 result['core_largest']=bool(result['core']>=max(v for k,v in result.items() if k!='core'))
 return result

def choices(local,d,buffer):
 result=[];held=[]
 for i in range(len(local)):
  ranked=sorted(np.flatnonzero(d['eligible'][i]),key=lambda j:(-local[i,j],d['symbols'][i][j]))
  names=d['symbols'][i];ranks={names[j]:k for k,j in enumerate(ranked)}
  keep=[t for t in held if ranks.get(t,999)<buffer]
  keep=sorted(keep,key=lambda t:ranks[t])[:3]
  for j in ranked:
   if len(keep)==3:break
   if names[j] not in keep:keep.append(names[j])
  held=sorted(keep,key=lambda t:ranks[t]) if len(keep)>=2 else []
  result.append(list(held))
 return result

def matching(ch,periods,mask):
 result=dict(months=int(mask.sum()),set_exact=0,order_exact=0,shared_slots=0)
 for i in np.flatnonzero(mask):
  target=list(map(e.actual,periods[i]['holdings']))
  result['set_exact']+=set(ch[i])==set(target);result['order_exact']+=ch[i]==target;result['shared_slots']+=len(set(ch[i])&set(target))
 return result

def key_for(row,mode='mixed'):
 m=row['inner_validation'];n=m['months'];slots=m['shared_slots']/(3*n);sets=m['set_exact']/n;orders=m['order_exact']/n
 score=.6*slots+.25*sets+.15*orders if mode=='mixed' else .6*slots+.4*sets
 return score,-(row['arch']!='linear'),-row['buffer'],row['reg'],-row['id']

def train():
 meta,packs,states=load();rows=[]
 for packkey,arch in itertools.product(packs,['linear','regime','spline']):
  d=packs[packkey];X,names,groups=design(d,states,arch)
  for reg,order_mass,sidepenalty in itertools.product([.0001,.001,.01,.1],[0.,.25,.5],[1.,16.]):
   summaries={b:[] for b in [3,5,7]};audits=[]
   for before,until in [(17,23),(23,35)]:
    mask=np.arange(69)<before;vm=(np.arange(69)>=before)&(np.arange(69)<until)
    w=fit(X,d,mask,groups,reg,order_mass,sidepenalty);scores=np.einsum('ijk,k->ij',X[:until],w,optimize=False)
    audits.append(contributions(X,w,groups,d,mask))
    for b in summaries:summaries[b].append(matching(choices(scores,d,b),meta['periods'],vm))
   for b,mm in summaries.items():
    summed={k:sum(m[k] for m in mm) for k in mm[0]}
    rows.append(dict(id=len(rows),pack=packkey,arch=arch,reg=reg,order_mass=order_mass,side_penalty=sidepenalty,buffer=b,inner_validation=summed,core_largest_inner=all(z['core_largest'] for z in audits)))
  print('Completed',len(rows),'policies',packkey,arch,flush=True)
 dump('INNER_TRIALS.json',rows)
 categories={'mixed_all':(rows,'mixed'),'set_only':([r for r in rows if r['order_mass']==0],'sets'),
             'core_primary':([r for r in rows if r['core_largest_inner']],'mixed'),
             'top32_core_primary':([r for r in rows if r['core_largest_inner'] and r['pack'].startswith('top32')],'mixed')}
 leaders={}
 for category,(rr,criterion) in categories.items():
  # Find best inner candidate still core-primary after first35 refit.
  for row in sorted(rr,key=lambda z:key_for(z,criterion),reverse=True):
   d=packs[row['pack']];X,names,groups=design(d,states,row['arch']);mask=np.arange(69)<35
   w=fit(X,d,mask,groups,row['reg'],row['order_mass'],row['side_penalty']);audit=contributions(X,w,groups,d,mask)
   if 'primary' in category and not audit['core_largest']:continue
   leaders[category]=dict(**row,weights=w.tolist(),names=names,groups=groups,training_contribution_audit=audit);break
 assert len(leaders)==4
 dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders,code_sha256=sha(__file__)))
 print('Frozen', {k:dict(pack=v['pack'],arch=v['arch'],inner=v['inner_validation'],core_primary=v['training_contribution_audit']['core_largest']) for k,v in leaders.items()},flush=True)

def evaluate():
 meta,packs,states=load();frozen=read(P/'FROZEN.json');fh=sha(P/'FROZEN.json')
 assert frozen['code_sha256']==sha(__file__)
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items())
 _,_,px,opening,c0,_=s.load();results={}
 for name,row in frozen['leaders'].items():
  d=packs[row['pack']];X,names,groups=design(d,states,row['arch']);w=np.array(row['weights']);local=np.einsum('ijk,k->ij',X,w,optimize=False)
  c=a.context_for(d,c0);scores=np.full(c['eligible'].shape,-np.inf)
  for i,ss in enumerate(d['symbols']):scores[i,[c['ids'][t] for t in ss]]=local[i,:len(ss)]
  cfg=dict(buffer=row['buffer'],gap=0.,exit=dict(name='none'));sim=policy.simulate(c,scores,np.ones_like(scores),cfg,detailed=True)
  chosen=choices(local,d,row['buffer']);assert chosen==[[c['symbols'][j] for j in ch] for ch in sim['choices']]
  met,slices,eq,trades=old.performance(px,opening,c,sim,scores,'close')
  os=policy.simulate(c,scores,np.ones_like(scores),cfg,execution='open');om,oss,_,_=old.performance(px,opening,c,os,scores,'open')
  mm={k:matching(chosen,meta['periods'],mask) for k,mask in old.masks(69).items() if k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=mm,rule=row)
  dump(name+'_TRADES.json',trades);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  details=[]
  for i,p in enumerate(meta['periods']):
   ss=d['symbols'][i];ranked=sorted(np.flatnonzero(d['eligible'][i]),key=lambda j:(-local[i,j],ss[j]));rank={ss[j]:k+1 for k,j in enumerate(ranked)}
   target=list(map(e.actual,p['holdings']));new=chosen[i];prior=chosen[i-1] if i else []
   records=[]
   for ticker in sorted(set(target)|set(new)):
    if ticker not in ss:records.append(dict(ticker=e.display(ticker),reason='outside_proxy_pool'));continue
    j=ss.index(ticker);components={g:float(sum(X[i,j,k]*w[k] for k,gg in enumerate(groups) if gg==g)) for g in set(groups)}
    records.append(dict(ticker=e.display(ticker),rank=rank.get(ticker),selected=ticker in new,author=ticker in target,components=components))
   details.append(dict(date=p['start'],end=p['end'],split='training' if i<35 else 'validation',author=p['holdings'],selected=list(map(e.display,new)),entered=list(map(e.display,set(new)-set(prior))),exited=list(map(e.display,set(prior)-set(new))),details=records))
  dump(name+'_MONTHS.json',details)
  # Upper explanatory check: same architecture fitted on all69, never used as
  # validation or prospective recommendation, no performance selection.
  w_all=fit(X,d,np.ones(69,bool),groups,row['reg'],row['order_mass'],row['side_penalty'])
  ch_all=choices(np.einsum('ijk,k->ij',X,w_all,optimize=False),d,row['buffer'])
  results[name]['retrospective_fit']=dict(matching=matching(ch_all,meta['periods'],np.ones(69,bool)),contributions=contributions(X,w_all,groups,d,np.ones(69,bool)),weights=w_all.tolist())
  print(name,dict(full_cagr=met['cagr'],full_sharpe=met['sharpe'],validation=slices['validation34'],match=mm,retrospective=results[name]['retrospective_fit']['matching']),flush=True)
 prior=read(O/'monthly_style_inverse/RESULTS.json');results.update({k:prior[k] for k in ['author','qqq']})
 dump('RESULTS.json',results)
 assert sha(P/'FROZEN.json')==fh
 changes=[f for f,h in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=fh,independent_replay_verified=True,selection='first35_only',outer_reused_history=True,complete_reproduction=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['initialize','prepare','train','evaluate']);globals()[parser.parse_args().phase]()
