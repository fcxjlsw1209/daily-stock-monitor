"""Bounded price-volume pattern correction, strict monthly reconstruction."""
from pathlib import Path
from datetime import datetime,timezone
import sys,argparse,itertools
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_score_architecture'))
import architecture as s
a,old,e,policy=s.a,s.old,s.e,s.policy
sys.path.insert(0,str(O/'monthly_multilayer_reconstruction'))
import research
read,sha=s.read,s.sha
FEATURES={'rebound21_atr':-1.,'positive_gap5_vol':-1.,'shakeout_recovery5':1.,'dry_consolidation':1.,'failed_rebounds5':-1.,'market_tail_residual252':1.}
MENUS=dict(chase=['rebound21_atr','positive_gap5_vol'],volume=['shakeout_recovery5','dry_consolidation','failed_rebounds5'],all=list(FEATURES))
RETENTION=[(3,0.),(5,.03),(7,.03)]

def dump(n,v):
 (P/n).write_text(old.json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 dump('SEEDS.json',read(O/'monthly_regime_tree_inverse/SEEDS.json'))
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),hypothesis='Retest author-mentioned anti-chasing, gap, shakeout recovery, dry consolidation and failed-rebound shapes under corrected own-sector benchmark and chronological35/34 split.',
 hard_rules='Core and sector-relative return numerators skip42sessions/two months; no recent return ranking or daily pullback gate. Monthend only, fixed intramonth units, zero costs. Patterns compete within combined20% shape+acceleration budget; core>=50%.',
 features=FEATURES,formulas_source=str(Path(research.__file__)),unused='Ignore sector correlation outputs from shared builder; no absolute sector-ETF strength. No calendar-event guesses or author cost/profit features.',
 training='Same6 earlier inner-only scoring seeds,3pattern menus,global/growth partitions,3retentions=108policies. Inner fit17/eval6 plus fit23/eval12. Select by .6slots+.25sets+.15orders, refit35/freeze/validate34. No returns to select; reused history not untouched OOS.',
 limitations='Assumed17ETF list and market-cap proxy pools; these hypotheses are not author code. No observation-account changes.'))
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in read(O/'monthly_score_architecture/PROTECTED_BEFORE.json')})

def shape_only(px,raw):
 # The two sector-correlation features of the reusable builder are deliberately ignored.
 rr=dict(raw,sector_return=px.pct_change(fill_method=None));allfeat=research.shape_features(px,rr)
 return {k:allfeat[k] for k in FEATURES}

def transform(tab,norm):
 if norm!='absolute_tanh':return s.normalize(tab,norm)
 scales={'rebound21_atr':(3.,3.),'positive_gap5_vol':(2.,0.),'shakeout_recovery5':(1.,0.),'dry_consolidation':(.3,0.),'failed_rebounds5':(5.,0.),'market_tail_residual252':(.01,0.)}
 out=tab.copy()
 for k in out:
  scale,center=scales[k];out[k]=.5*np.tanh((out[k]-center)/scale)
 return out.fillna(0.)

def prepare():
 meta,packs,px,opening,c,styles=s.load();raw=e.load_prices()[1];shapes=shape_only(px,raw);missing={}
 for key,d in packs.items():
  S=np.zeros((69,d['X'].shape[1],len(FEATURES)));counts={k:0 for k in FEATURES}
  for i,ss in enumerate(d['symbols']):
   tab=pd.DataFrame({k:v.loc[c['dates'][i],ss] for k,v in shapes.items()})
   for k in tab:counts[k]+=int(tab[k].isna().sum())
   S[i,:len(ss)]=transform(tab,d['norm'])
  np.savez_compressed(P/(key+'.npz'),shape=S);missing[key]=counts
 sources=read(O/'monthly_close_exit_recheck/SOURCE_HASHES.json');sources[str(Path(__file__))]=sha(__file__);sources[str(Path(research.__file__))]=sha(research.__file__)
 dump('SOURCE_HASHES.json',sources);dump('MISSING_PATTERNS.json',missing);print('Prepared patterns12packs',flush=True)

def load():
 meta,packs,px,opening,c,Z=s.load()
 for key,d in packs.items():
  d['X']=np.concatenate([d['X'],np.load(P/(key+'.npz'))['shape']],axis=2);d['names']=list(d['names'])+list(FEATURES)
 return meta,packs,px,opening,c,Z

def setup(seed,menu,packs,c0):
 d=s.apply_gate(packs[seed['key']],seed['menu'],seed['gate']);c=a.context_for(d,c0);X,spec=a.design(d,s.MENUS[seed['menu']],'legacy')
 addon=[(k,FEATURES[k],'shape') for k in MENUS[menu]]
 Y=np.stack([d['X'][:,:,d['names'].index(k)]*sign for k,sign,_ in addon],axis=-1)
 return d,c,np.concatenate([X,Y],axis=2),spec+addon

def choose(r):
 m=r['inner_validation'];score=.6*m['shared_slots']/(3*m['months'])+.25*m['set_exact']/m['months']+.15*m['order_exact']/m['months']
 return score,-(r['partition']!='global'),-len(MENUS[r['patterns']]),-r['config']['buffer'],-r['id']

def train():
 _,packs,px,opening,c0,Z=load();rows=[]
 for sid,seed in enumerate(read(P/'SEEDS.json')):
  for menu in MENUS:
   d,c,X,spec=setup(seed,menu,packs,c0);agg={(part,b,g):[] for part in ['global','growth'] for b,g in RETENTION}
   for before,until in a.FOLDS:
    mask=np.arange(69)<before;val=(np.arange(69)>=before)&(np.arange(69)<until);base=s.fit_base(X,d,mask,spec,seed['objective'])
    for part in ['global','growth']:
     groups=Z[part].to_numpy();rule=s.fit_rule(X,d,mask,spec,seed['objective'],groups,base);scores,strength,_,_=a.scores_for(rule,X,spec,d,c,groups)
     for b,g in RETENTION:
      cfg=dict(buffer=b,gap=g,exit=dict(name='none'));sim=policy.simulate(c,scores,strength,cfg,count=until);agg[(part,b,g)].append(policy.matching(sim['choices'],scores,c,val))
   for (part,b,g),mm in agg.items():rows.append(dict(id=len(rows),seed_id=sid,seed=seed,patterns=menu,partition=part,config=dict(buffer=b,gap=g,exit=dict(name='none')),inner_validation={k:sum(v[k] for v in mm) for k in ['months','shared_slots','set_exact','order_exact']}))
  print('Pattern policies completed',len(rows),flush=True)
 leaders={}
 for name,group in [('overall',rows),('ungated',[r for r in rows if r['seed']['gate']=='positive'])]:
  row=max(group,key=choose);d,c,X,spec=setup(row['seed'],row['patterns'],packs,c0);rule=s.fit_rule(X,d,np.arange(69)<35,spec,row['seed']['objective'],Z[row['partition']].to_numpy());leaders[name]=dict(**row,rule=rule)
 dump('INNER_TRIALS.json',rows);dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders));print('FROZEN',{n:{k:r[k] for k in ['seed_id','patterns','partition','config','inner_validation']} for n,r in leaders.items()},flush=True)

def evaluate():
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items());h=sha(P/'FROZEN.json');_,packs,px,opening,c0,Z=load();ms=old.masks(69);results={}
 for name,row in read(P/'FROZEN.json')['leaders'].items():
  d,c,X,spec=setup(row['seed'],row['patterns'],packs,c0);scores,strength,local,W=a.scores_for(row['rule'],X,spec,d,c,Z[row['partition']].to_numpy());sim=policy.simulate(c,scores,strength,row['config'],detailed=True)
  met,slices,eq,trades=old.performance(px,opening,c,sim,scores,'close');os=policy.simulate(c,scores,strength,row['config'],execution='open');om,oss,_,_=old.performance(px,opening,c,os,scores,'open');matching={k:policy.matching(sim['choices'],scores,c,ms[k]) for k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=matching,rule=row);dump(name+'_TRADES.json',trades);dump(name+'_DECISIONS.json',sim['events']);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  if name=='overall':
   er=dict(row['seed'],partition=row['partition'],config=row['config'],rule=row['rule']);months,txt=old.explain(er,d,c,Z,scores,local,X,W,sim);dump('MONTHLY_EXPLANATIONS.json',months);(P/'MONTHLY_EXPLANATIONS.txt').write_text(txt)
  print(name,dict(cagr=met['cagr'],sharpe=met['sharpe'],validation=slices['validation34'],matching=matching),flush=True)
 prior=read(O/'monthly_style_inverse/RESULTS.json')
 for k in ['author','qqq']:results[k]=prior[k]
 dump('RESULTS.json',results);assert sha(P/'FROZEN.json')==h;changes=[f for f,v in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=v];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=h,split='first35/later34',zero_cost=True,no_intramonth_trades=True,independent_replay_verified=True,not_pristine_oos=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','prepare','train','evaluate']);globals()[p.parse_args().phase]()
