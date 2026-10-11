"""Retest small causal return-Ridge and price-volume layers after budget removal."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,json,sys
import numpy as np
import pandas as pd
import open_inverse as m
P=Path(__file__).resolve().parent/'minor_layers';P.mkdir(exist_ok=True)
sys.path.insert(0,str(m.O/'monthly_pattern_recheck'))
import patterns

def dump(n,z):
 (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def prepare():
 assert not (P/'PLAN.json').exists()
 meta,packs,states=m.load();px,raw,_,_,sources,_=m.e.load_prices();shapes=patterns.shape_only(px,raw)
 rows=sorted(m.read(m.P/'INNER_TRIALS.json'),key=m.key_for,reverse=True);seeds=[];seen=set()
 for row in rows:
  # Same training-only seed selection, no outer results or returns.
  key=(row['pack'],row['arch'],row['reg'],row['order_mass'],row['side_penalty'])
  if key in seen:continue
  seen.add(key);seeds.append(row)
  if len(seeds)==6:break
 dump('SEEDS.json',seeds)
 forecastroot=m.O/'monthly_causal_ridge';pred=np.load(forecastroot/'PREDICTIONS.npz');fm=m.read(m.O/'monthly_adaptive_core/INPUTS.json')
 audit=m.read(forecastroot/'FORECAST_AUDIT.json')
 assert all(z['latest_label_end']<=z['cutoff'] for z in audit)
 source={str(forecastroot/f):m.sha(forecastroot/f) for f in ['FORECAST_AUDIT.json','PREDICTIONS.npz','MARKET_TRAINING.json']}
 source[str(Path(patterns.research.__file__))]=m.sha(patterns.research.__file__)
 missing={}
 for key,d in packs.items():
  shape=np.zeros((69,d['X'].shape[1],len(patterns.FEATURES)));forecast=np.zeros((6,69,d['X'].shape[1]));miss=0
  fkey=f"top{d['pool']}";fss=fm['packs'][fkey+'_percentile']['symbols']
  for i,ss in enumerate(d['symbols']):
   tab=pd.DataFrame({k:z.loc[meta['periods'][i]['start'],ss] for k,z in shapes.items()})
   shape[i,:len(ss)]=m.normalize(tab.where(pd.Series(d['eligible'][i,:len(ss)],index=tab.index),axis=0),d['norm'])
   for j,t in enumerate(ss):
    if t in fss[i]:forecast[:,i,j]=pred[fkey][:,i,fss[i].index(t)]
    else:miss+=1
  np.savez_compressed(P/(key+'.npz'),shape=shape,forecast=forecast);missing[key]=miss
 dump('SOURCE_HASHES.json',source);dump('MISSING_FORECAST_SYMBOL_MONTHS.json',missing)
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),seeds='6 specifications from new experiment first35-only inner matching; no outer returns used.',patterns='6 anti-chasing/volume/tail-resilience features, re-fit signed inverse coefficients with no inherited group caps. No hard pullback entry requirement.',forecast='Reuse6 causal yearly5year Ridge variants trained on market returns; future label end strictly<=annual cutoff. Original training pool/positive-momentum filter remains a forecast limitation. Missing ETF predictions neutral and counted.',blend='Rank-normalized base mixed with ranked return-Ridge at0/2.5/5/10/20%. Global or only weak-SPY-MA200 condition. <=20 is a tested small-layer range, not claimed author parameter.',selection='First17/eval6 andfirst23/eval12.6seeds x2pattern menus x(1zero+6forecasts x4weights x2modes) x3retentions=1764 configurations. First35 refit/freeze/evaluate34; reused validation.',code_sha256=m.sha(__file__)))

def setup(row,packs,states):
 d=packs[row['seed']['pack']];X,names,groups=m.design(d,states,row['seed']['arch']);extra=np.load(P/(row['seed']['pack']+'.npz'))
 if row['patterns']:
  X=np.concatenate([X,extra['shape']],axis=-1);names+=list(patterns.FEATURES);groups+=['shape']*len(patterns.FEATURES)
 return d,X,names,groups,extra['forecast']

def combine(local,d,forecast,alpha,weak):
 answer=np.zeros_like(local)
 for i,ss in enumerate(d['symbols']):
  base=pd.Series(local[i,:len(ss)]).where(d['eligible'][i,:len(ss)]).rank(pct=True).fillna(.5).to_numpy()-.5
  w=alpha*weak[i];answer[i,:len(ss)]=(1-w)*base+w*forecast[i,:len(ss)]
 return answer

def train():
 meta,packs,states=m.load();rows=[]
 for sid,seed in enumerate(m.read(P/'SEEDS.json')):
  for usepatterns in [False,True]:
   template=dict(seed=seed,patterns=usepatterns);d,X,names,groups,pred=setup(template,packs,states)
   options=[(-1,0.,'global')]+list(itertools.product(range(6),[.025,.05,.1,.2],['global','weak']))
   allmm={(forecast,alpha,mode,b):[] for forecast,alpha,mode in options for b in [3,5,7]};audits=[]
   for before,until in [(17,23),(23,35)]:
    mask=np.arange(69)<before;vm=(np.arange(69)>=before)&(np.arange(69)<until)
    w=m.fit(X,d,mask,groups,seed['reg'],seed['order_mass'],seed['side_penalty']);local=np.einsum('ijk,k->ij',X,w,optimize=False);audits.append(m.contributions(X,w,groups,d,mask))
    for forecast,alpha,mode in options:
     forecastscore=pred[forecast] if forecast>=0 else np.zeros_like(local);weak=np.ones(69) if mode=='global' else 1-states[:,0];score=combine(local,d,forecastscore,alpha,weak)
     for b in [3,5,7]:allmm[(forecast,alpha,mode,b)].append(m.matching(m.choices(score[:until],d,b),meta['periods'],vm))
   for (forecast,alpha,mode,b),mm in allmm.items():
    rows.append(dict(id=len(rows),seed_id=sid,seed=seed,patterns=usepatterns,forecast=forecast,alpha=alpha,forecast_mode=mode,buffer=b,arch=seed['arch'],reg=seed['reg'],inner_validation={k:sum(z[k] for z in mm) for k in mm[0]},core_largest_inner=all(z['core_largest'] for z in audits)))
  print('Completed',len(rows),'layer policies',flush=True)
 dump('INNER_TRIALS.json',rows);leaders={}
 cats={'all':rows,'core_primary':[r for r in rows if r['core_largest_inner']],'with_ridge':[r for r in rows if r['core_largest_inner'] and r['alpha']>0]}
 for name,rr in cats.items():
  for row in sorted(rr,key=m.key_for,reverse=True):
   d,X,names,groups,pred=setup(row,packs,states);seed=row['seed'];w=m.fit(X,d,np.arange(69)<35,groups,seed['reg'],seed['order_mass'],seed['side_penalty']);audit=m.contributions(X,w,groups,d,np.arange(69)<35)
   if name!='all' and not audit['core_largest']:continue
   leaders[name]=dict(**row,weights=w.tolist(),names=names,groups=groups,training_contribution_audit=audit);break
 assert len(leaders)==3
 dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders,code_sha256=m.sha(__file__),base_code_sha256=m.sha(m.__file__)))
 print('Frozen',{k:dict(pack=v['seed']['pack'],patterns=v['patterns'],alpha=v['alpha'],mode=v['forecast_mode'],inner=v['inner_validation']) for k,v in leaders.items()},flush=True)

def evaluate():
 meta,packs,states=m.load();frozen=m.read(P/'FROZEN.json');fh=m.sha(P/'FROZEN.json');results={}
 assert frozen['code_sha256']==m.sha(__file__) and frozen['base_code_sha256']==m.sha(m.__file__)
 assert all(m.sha(f)==h for f,h in m.read(P/'SOURCE_HASHES.json').items())
 _,_,px,opening,c0,_=m.s.load()
 for name,row in frozen['leaders'].items():
  d,X,names,groups,pred=setup(row,packs,states);w=np.array(row['weights']);local=np.einsum('ijk,k->ij',X,w,optimize=False);weak=np.ones(69) if row['forecast_mode']=='global' else 1-states[:,0]
  score=combine(local,d,pred[row['forecast']] if row['forecast']>=0 else np.zeros_like(local),row['alpha'],weak)
  c=m.a.context_for(d,c0);scores=np.full(c['eligible'].shape,-np.inf)
  for i,ss in enumerate(d['symbols']):scores[i,[c['ids'][t] for t in ss]]=score[i,:len(ss)]
  cfg=dict(buffer=row['buffer'],gap=0.,exit=dict(name='none'));sim=m.policy.simulate(c,scores,np.ones_like(scores),cfg,detailed=True);chosen=m.choices(score,d,row['buffer']);assert chosen==[[c['symbols'][j] for j in ch] for ch in sim['choices']]
  met,slices,eq,trades=m.old.performance(px,opening,c,sim,scores,'close');os=m.policy.simulate(c,scores,np.ones_like(scores),cfg,execution='open');om,oss,_,_=m.old.performance(px,opening,c,os,scores,'open')
  mm={k:m.matching(chosen,meta['periods'],mask) for k,mask in m.old.masks(69).items() if k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=mm,rule=row)
  dump(name+'_TRADES.json',trades);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()]);dump(name+'_MONTHS.json',[dict(date=p['start'],end=p['end'],author=p['holdings'],selected=list(map(m.e.display,chosen[i]))) for i,p in enumerate(meta['periods'])])
  print(name,dict(full_cagr=met['cagr'],full_sharpe=met['sharpe'],validation=slices['validation34'],matching=mm),flush=True)
 prior=m.read(m.O/'monthly_style_inverse/RESULTS.json');results.update({k:prior[k] for k in ['author','qqq']});dump('RESULTS.json',results)
 assert m.sha(P/'FROZEN.json')==fh;changes=[f for f,h in m.read(m.P/'PROTECTED_BEFORE.json').items() if m.sha(f)!=h];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=fh,independent_replay_verified=True,selection='first35_only',outer_reused_history=True,complete_reproduction=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','train','evaluate']);globals()[ap.parse_args().phase]()
