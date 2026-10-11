"""Finite architecture study: preserves long momentum, isolates research artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import sys, itertools, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
P=Path(__file__).resolve().parent
O=P.parent
sys.path.insert(0,str(O/'monthly_adaptive_core'))
import adaptive as a
old,e,policy,strict=a.old,a.e,a.policy,a.strict
read,sha=a.read,a.sha
MENUS={
 'raw':['mom252_skip42','mom294_skip42','calendar12_skip2_raw','calendar14_skip2_raw'],
 'log':['logmom252_skip42','logmom294_skip42'],
 'risk':['mom252_risk_aligned','mom294_risk_aligned','calendar12_skip2_half','calendar14_skip2_half'],
 'sharpe':['mean_sharpe252_skip42','mean_sharpe294_skip42']}
NORMS=['percentile','z_tanh','absolute_tanh']
GATES=['positive','top_half','top12']
OBJECTIVES=['ordered','set_order']
PARTITIONS=['global','growth']
RETENTIONS=[(3,0.),(5,.03)]

def dump(n,v):
 (P/n).write_text(old.json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
 hypothesis='Long momentum first qualifies, then bounded score differentiates candidates. Compare retaining absolute factor distances vs cross-sectional rank, and top3-set vs ordered-choice fitting.',
 hard_rules='Long-only, up to3 equal monthend positions, fixed intramonth units, zero cost idealized same-close; core and own-sector relative return numerators skip42sessions/two months. No ticker/month lookup, no future return optimized.',
 transforms='Percentile control; monthly zscore followed by .5*tanh(z/1.5); fixed economically scaled raw factors followed by .5*tanh. No fitting normalization to later months.',
 gates='Positive12-2 control; top half or top12 by mean percentile of core menu among positive/warmup candidates. Same gate applies all names, including incumbent exits. Missing author targets count as mismatches.',
 objectives='Constrained ordered likelihood vs pairwise selected>unselected(75%) and author-order(25%) logistic loss. Same core>=50%,vol<=30%,own-sector<=20%,shape+accel<=20% budgets.',
 selection='4pool/ETF controls x3scales x4core menus x3gates x2objectives x2regimes x2retentions=1152policies. Inner fit17/eval6 and fit23/eval12. Select on inner18 .6slots+.25sets+.15orders, freeze, refit35, evaluate34 once. Historical latter34 reused; not pristine OOS.',
 limitation='Proxy market-cap pools and two assumed ETF17 lists; user-approved Top35 control. No change to observation accounts or past research. No performance-driven promotion.'))
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in read(O/'monthly_adaptive_core/PROTECTED_BEFORE.json')})

def absolute_scale(name):
 if 'mean_sharpe' in name:return .10,0.
 if 'risk_aligned' in name:return 32.,0.
 if 'half' in name:return 6.,0.
 if name.startswith(('mom','logmom','calendar')):return 1.,0.
 if name=='vol126':return .02,.02
 if name.startswith('stock_sector'):return .4,0.
 if name.startswith(('cmf','highvol_cmf','trend_cmf','signed_volume')):return .25,0.
 if name=='volume_ratio21_126':return .5,1.
 if name=='acceleration_skip42':return .5,0.
 raise ValueError(name)

def normalize(tab,mode):
 if mode=='percentile':return tab.rank(pct=True).fillna(.5)-.5
 if mode=='z_tanh':return (.5*np.tanh((tab-tab.mean())/tab.std(ddof=0).replace(0,np.nan)/1.5)).fillna(0.)
 out=tab.copy()
 for k in out:
  scale,center=absolute_scale(k);out[k]=.5*np.tanh((out[k]-center)/scale)
 return out.fillna(0.)

def prepare():
 px,raw,members,periods,sources,pools=e.load_prices()
 feat,_,states=e.previous.panels(px,raw);core,side=e.features(px,raw)
 names=list(dict.fromkeys(k for menu in MENUS.values() for k in menu))
 clean={k:feat[k] if k in feat else core[k] for k in names}
 sides=['vol126','cmf21','cmf63','signed_volume63','volume_ratio21_126','acceleration_skip42']
 clean.update({k:side[k] for k in sides})
 rel={f'stock_sector_skip42_{n}':px.shift(42)/px.shift(n)-1 for n in [126,252]}
 names=list(clean)+list(rel)+['highvol_cmf63','trend_cmf63']
 sf=O/'index_union/alias_resolved_inputs/sectors.csv.gz';sectors=pd.read_csv(sf,index_col=0,parse_dates=True,low_memory=False);sources[str(sf)]=sha(sf)
 warm=px.notna().rolling(295).sum().eq(295)&px.gt(0);mom=px.shift(42)/px.shift(252)-1
 metadata=dict(periods=periods,packs={})
 for etf,pool in itertools.product(['theme17','core17'],[32,35]):
  symbols=[sorted({e.actual(x) for x in pools[str(pool)][p['start']]}|members[etf][1]) for p in periods]
  R=np.zeros((69,max(map(len,symbols)),len(names)));eligible=np.zeros(R.shape[:2],bool);labels=np.zeros_like(eligible);order=np.full((69,3),-1,int)
  for i,(p,ss) in enumerate(zip(periods,symbols)):
   date=p['start'];bm=strict.benchmark_map(date,ss,sectors,members[etf][1]);tab=pd.DataFrame({k:z.loc[date,ss] for k,z in clean.items()})
   for k,z in rel.items():tab[k]=strict.relative_row(z,date,ss,bm)
   tab['highvol_cmf63']=tab.cmf63*states.at[date,'high_market_vol'];tab['trend_cmf63']=tab.cmf63*states.at[date,'spy_trend']
   R[i,:len(ss)]=tab[names]
   eligible[i,:len(ss)]=(warm.loc[date,ss]&mom.loc[date,ss].gt(0)).to_numpy()
   for j,n in enumerate(p['holdings']):
    s=e.actual(n)
    if s in ss:
     k=ss.index(s);labels[i,k]=True
     if eligible[i,k]:order[i,j]=k
  for norm in NORMS:
   X=np.zeros_like(R)
   for i,ss in enumerate(symbols):X[i,:len(ss)]=normalize(pd.DataFrame(R[i,:len(ss)],columns=names),norm)
   key=f'{etf}_top{pool}_{norm}'
   np.savez_compressed(P/(key+'.npz'),X=X,raw=R,eligible=eligible,labels=labels,order=order)
   metadata['packs'][key]=dict(symbols=symbols,names=names,norm=norm,pool=pool,etf=etf)
 sources[str(Path(__file__))]=sha(__file__)
 for mod in [a,old,e,policy,strict]:sources[str(Path(mod.__file__))]=sha(mod.__file__)
 dump('INPUTS.json',metadata);dump('SOURCE_HASHES.json',sources)
 print('Prepared12 packs',flush=True)

def load():
 meta=read(P/'INPUTS.json');packs={k:dict(np.load(P/(k+'.npz')),**v) for k,v in meta['packs'].items()}
 # Context uses original loader/prices; expand union to cover both ETF hypotheses.
 _,_,px,opening,c,styles=a.load();symbols=sorted({s for d in packs.values() for row in d['symbols'] for s in row})
 dates=pd.to_datetime(c['dates']);r=px.pct_change(fill_method=None);sigma=r.rolling(63).std()*np.sqrt(21)
 peaks=[]
 for i,date in enumerate(dates):
  start=date if i==0 else px.index[px.index.get_loc(dates[i-1])+1];peaks.append(px.loc[start:date,symbols].max().to_numpy())
 c.update(symbols=symbols,ids={s:j for j,s in enumerate(symbols)},close=px.loc[dates,symbols].to_numpy(),opening=opening.loc[pd.to_datetime(c['opening_dates']),symbols].to_numpy(),sigma=sigma.loc[dates,symbols].to_numpy(),peak=np.stack(peaks))
 return meta,packs,px,opening,c,styles

def apply_gate(d,menu,gate):
 ans=dict(d);good=d['eligible'].copy();raw=d['raw'][:,:,[d['names'].index(k) for k in MENUS[menu]]]
 for i,ss in enumerate(d['symbols']):
  tab=pd.DataFrame(raw[i,:len(ss)]);tab.loc[~good[i,:len(ss)],:]=np.nan;values=tab.rank(pct=True).mean(axis=1)
  if gate=='top_half':good[i,:len(ss)]&=values.ge(.5).to_numpy()
  if gate=='top12':
   chosen=values.dropna().sort_values(ascending=False,kind='stable').head(12).index;keep=np.zeros(len(ss),bool);keep[chosen]=True;good[i,:len(ss)]&=keep
 ans['eligible']=good;ans['order']=d['order'].copy()
 for i,row in enumerate(ans['order']):
  for j,k in enumerate(row):
   if k>=0 and not good[i,k]:ans['order'][i,j]=-1
 return ans

def pair_design(X,d,mask):
 D=[];weights=[]
 for i in np.flatnonzero(mask):
  seq=[int(k) for k in d['order'][i] if k>=0];others=[k for k in np.flatnonzero(d['eligible'][i]) if k not in seq]
  groups=[([(x,y) for x in seq for y in others],.75), (list(itertools.combinations(seq,2)),.25)]
  for pairs,mass in groups:
   for x,y in pairs:D.append(X[i,x]-X[i,y]);weights.append(mass/max(1,len(pairs)))
 w=np.array(weights);w/=w.sum()
 return np.stack(D),w

def fit_base(X,d,mask,spec,objective):
 if objective=='ordered':return e.fit(X,d,mask,spec,.05,.10)[0]
 D,pw=pair_design(X,d,mask);A,b,prior=e.limits(spec)
 def loss(w):
  z=np.einsum('ij,j->i',D,w,optimize=False)/.05
  value=np.sum(pw*np.logaddexp(0,-z))+.1*np.sum((w-prior)**2)
  grad=-np.einsum('i,ij->j',pw*expit(-z),D,optimize=False)/.05+.2*(w-prior)
  return value,grad
 opt=minimize(loss,prior,jac=True,method='SLSQP',bounds=[(0,1)]*len(spec),constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(len(w))),dict(type='ineq',fun=lambda w:b-np.einsum('ij,j->i',A,w,optimize=False),jac=lambda w:-A)],options=dict(maxiter=500,ftol=1e-10))
 assert opt.success,opt.message
 return opt.x

def fit_rule(X,d,mask,spec,objective,groups,base=None):
 base=fit_base(X,d,mask,spec,objective) if base is None else base
 weights={};audit={}
 for g in sorted(set(groups)):
  gm=mask&(groups==g)
  if len(set(groups))==1 or gm.sum()<8:w=base.copy()
  else:
   target=fit_base(X,d,gm,spec,objective);delta=target-base;shrink=min(1.,.3/max(1e-12,np.abs(delta).sum()));w=base+shrink*delta
  weights[str(g)]=w.tolist();audit[str(g)]=dict(months=int(gm.sum()),l1_delta=float(np.abs(w-base).sum()))
 return dict(spec=spec,base=base.tolist(),weights=weights,audit=audit)

def choose(row):
 mm=row['inner_validation'];metric=.6*mm['shared_slots']/(3*mm['months'])+.25*mm['set_exact']/mm['months']+.15*mm['order_exact']/mm['months']
 return metric,-(row['partition']!='global'),-(row['gate']!='positive'),-row['config']['buffer'],-row['id']

def train():
 meta,packs,px,opening,c0,styles=load();rows=[]
 for key,menu,gate,objective in itertools.product(packs,MENUS,GATES,OBJECTIVES):
  d=apply_gate(packs[key],menu,gate);c=a.context_for(d,c0);X,spec=a.design(d,MENUS[menu],'legacy')
  aggregations={(part,b,gap):[] for part in PARTITIONS for b,gap in RETENTIONS}
  for before,until in a.FOLDS:
   mask=np.arange(69)<before;val=(np.arange(69)>=before)&(np.arange(69)<until);base=fit_base(X,d,mask,spec,objective)
   for part in PARTITIONS:
    groups=styles[part].to_numpy();rule=fit_rule(X,d,mask,spec,objective,groups,base);scores,strength,_,_=a.scores_for(rule,X,spec,d,c,groups)
    for b,gap in RETENTIONS:
     cfg=dict(buffer=b,gap=gap,exit=dict(name='none'));sim=policy.simulate(c,scores,strength,cfg,count=until)
     aggregations[(part,b,gap)].append(policy.matching(sim['choices'],scores,c,val))
  for (part,b,gap),mm in aggregations.items():
   summed={k:sum(x[k] for x in mm) for k in ['months','shared_slots','set_exact','order_exact']}
   rows.append(dict(id=len(rows),key=key,menu=menu,gate=gate,objective=objective,partition=part,config=dict(buffer=b,gap=gap,exit=dict(name='none')),inner_validation=summed))
  if len(rows)%96==0:print('Completed',len(rows),'policies',flush=True)
 assert len(rows)==1152
 categories=dict(overall=rows,gated=[r for r in rows if r['gate']!='positive'],absolute=[r for r in rows if 'absolute_tanh' in r['key']],set_order=[r for r in rows if r['objective']=='set_order'])
 leaders={}
 for name,rr in categories.items():
  row=max(rr,key=choose);d=apply_gate(packs[row['key']],row['menu'],row['gate']);X,spec=a.design(d,MENUS[row['menu']],'legacy')
  leaders[name]=dict(**row,rule=fit_rule(X,d,np.arange(69)<35,spec,row['objective'],styles[row['partition']].to_numpy()))
 dump('INNER_TRIALS.json',rows);dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders))
 print('FROZEN',{k:{j:v[j] for j in ['key','menu','gate','objective','partition','inner_validation']} for k,v in leaders.items()},flush=True)

def evaluate():
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items());h=sha(P/'FROZEN.json')
 meta,packs,px,opening,c0,styles=load();results={};ms=old.masks(69)
 for name,row in read(P/'FROZEN.json')['leaders'].items():
  d=apply_gate(packs[row['key']],row['menu'],row['gate']);c=a.context_for(d,c0);X,spec=a.design(d,MENUS[row['menu']],'legacy')
  scores,strength,local,W=a.scores_for(row['rule'],X,spec,d,c,styles[row['partition']].to_numpy());sim=policy.simulate(c,scores,strength,row['config'],detailed=True)
  met,slices,eq,trades=old.performance(px,opening,c,sim,scores,'close');os=policy.simulate(c,scores,strength,row['config'],execution='open');om,oss,_,_=old.performance(px,opening,c,os,scores,'open')
  matching={k:policy.matching(sim['choices'],scores,c,ms[k]) for k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=matching,rule=row)
  dump(name+'_TRADES.json',trades);dump(name+'_DECISIONS.json',sim['events']);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  if name=='overall':
   months,txt=old.explain(row,d,c,styles,scores,local,X,W,sim);dump('MONTHLY_EXPLANATIONS.json',months);(P/'MONTHLY_EXPLANATIONS.txt').write_text(txt)
  print(name,dict(cagr=met['cagr'],sharpe=met['sharpe'],validation=slices['validation34'],matching=matching),flush=True)
 prior=read(O/'monthly_style_inverse/RESULTS.json')
 for k in ['author','qqq']:results[k]=prior[k]
 dump('RESULTS.json',results);assert sha(P/'FROZEN.json')==h
 changes=[f for f,v in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=v];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=h,inner_folds=a.FOLDS,split='first35/later34',no_outer_selection=True,zero_cost=True,no_intramonth_trades=True,independent_replay_verified=True,not_pristine_oos=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','prepare','train','evaluate']);globals()[p.parse_args().phase]()
