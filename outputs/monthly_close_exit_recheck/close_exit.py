"""Monthend-only endogenous-position exit policies; finite train35/validate34 study."""
from pathlib import Path
from datetime import datetime, timezone
import sys,argparse,itertools
import numpy as np
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_score_architecture'))
import architecture as s
a,old,e,policy=s.a,s.old,s.e,s.policy
read,sha=s.read,s.sha
RETENTIONS=[(3,0.),(5,0.),(5,.03),(7,.03)]
EXITS=[dict(name='none')]
for loss,profit in itertools.product([.10,.20],[.25,.50]):EXITS.append(dict(name='fixed',loss=loss,profit=profit))
for adaptive,loss,profit,trail in itertools.product([False,True],[1.5,2.5,3.5],[None,3.,5.],[2.,4.]):
 EXITS.append(dict(name='adaptive' if adaptive else 'sigma',loss_k=loss,profit_k=profit,trail_k=trail,adaptive=adaptive))

def dump(n,v):
 (P/n).write_text(old.json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')

def initialize():
 assert not (P/'PLAN.json').exists()
 rows=sorted(read(O/'monthly_score_architecture/INNER_TRIALS.json'),key=s.choose,reverse=True);seeds=[];seen=set()
 for row in rows:
  key=tuple(row[k] for k in ['key','menu','gate','objective','partition'])
  if key not in seen:seen.add(key);seeds.append(row)
  if len(seeds)==6:break
 dump('SEEDS.json',seeds)
 dump('PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
 hypothesis='Author hinted market/volatility/signal-dependent exits. User confirms no intramonth trades. Test such thresholds strictly at monthend, followed by next-ranked eligible replacements.',
 seeds='Six unique scoring specs chosen only from architecture first35 inner18 matching; ignore outer returns and matches.',
 grid=dict(scoring_seeds=6,exit_rules=len(EXITS),retentions=RETENTIONS,policies=6*len(EXITS)*len(RETENTIONS)),
 thresholds=EXITS,hard_rules='Monthend close only, fixed shares between monthends. Do not use author prior holdings/costs/peaks. Use own simulated weighted cost and past closing peak. Exit names cannot be repurchased at same close; may requalify next month. Equal weight up to3, fewer than2 means cash. Zero costs.',
 adaptive='Existing rule: monthly sigma=63daily std*sqrt21; weak SPYtrend .8, highmarketvol .9, strong core quartile1.25, weak half.8. Loss5-50%,profit10-100%,trail8-60% caps. These are research hypotheses, not original author values.',
 training='Fit17/eval6 and fit23/eval12; aggregate18 matching, no returns to select. Refit chosen scorer35, freeze exits, then later34 validation. Reused history not untouched OOS.',
 objective='Same .6slot+.25set+.15ordered matching; ties prefer none exits, smaller buffer, earlier seed/rule.',
 protection='Observation ledgers/rules/plans/fingerprints untouched. No promotion. Same-close execution is idealized reproduction, next-open sensitivity separate.'))
 dump('PROTECTED_BEFORE.json',{f:sha(f) for f in read(O/'monthly_score_architecture/PROTECTED_BEFORE.json')})
 sources=read(O/'monthly_score_architecture/SOURCE_HASHES.json');sources[str(Path(__file__))]=sha(__file__)
 for n in ['INNER_TRIALS.json','INPUTS.json']+[k+'.npz' for k in read(O/'monthly_score_architecture/INPUTS.json')['packs']]:sources[str(O/'monthly_score_architecture'/n)]=sha(O/'monthly_score_architecture'/n)
 dump('SOURCE_HASHES.json',sources)

def choose(row):
 mm=row['inner_validation'];v=.6*mm['shared_slots']/(3*mm['months'])+.25*mm['set_exact']/mm['months']+.15*mm['order_exact']/mm['months']
 return v,-(row['config']['exit']['name']!='none'),-row['config']['buffer'],-row['id']

def get_score(seed,packs,c0,styles,mask):
 d=s.apply_gate(packs[seed['key']],seed['menu'],seed['gate']);c=a.context_for(d,c0);X,spec=a.design(d,s.MENUS[seed['menu']],'legacy');groups=styles[seed['partition']].to_numpy()
 rule=s.fit_rule(X,d,mask,spec,seed['objective'],groups);score,strength,local,W=a.scores_for(rule,X,spec,d,c,groups)
 return d,c,X,spec,rule,score,strength,local,W

def train():
 _,packs,px,opening,c0,styles=s.load();seeds=read(P/'SEEDS.json');rows=[]
 for seed_id,seed in enumerate(seeds):
  agg={(b,gap,ei):[] for b,gap in RETENTIONS for ei in range(len(EXITS))}
  for before,until in a.FOLDS:
   d,c,X,spec,rule,scores,strength,_,_=get_score(seed,packs,c0,styles,np.arange(69)<before)
   mask=(np.arange(69)>=before)&(np.arange(69)<until)
   for b,gap,ei in agg:
    cfg=dict(buffer=b,gap=gap,exit=EXITS[ei]);sim=policy.simulate(c,scores,strength,cfg,count=until)
    agg[(b,gap,ei)].append(policy.matching(sim['choices'],scores,c,mask))
  for (b,gap,ei),mm in agg.items():
   summed={k:sum(z[k] for z in mm) for k in ['months','shared_slots','set_exact','order_exact']}
   rows.append(dict(id=len(rows),seed_id=seed_id,seed=seed,config=dict(buffer=b,gap=gap,exit=EXITS[ei]),inner_validation=summed))
  print('Finished policies',len(rows),flush=True)
 leaders={}
 for name,group in [('overall',rows),('no_exit',[r for r in rows if r['config']['exit']['name']=='none']),('explicit_exit',[r for r in rows if r['config']['exit']['name']!='none'])]:
  row=max(group,key=choose);rule=get_score(row['seed'],packs,c0,styles,np.arange(69)<35)[4];leaders[name]=dict(**row,rule=rule)
 dump('INNER_TRIALS.json',rows);dump('FROZEN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),leaders=leaders))
 print('FROZEN',{n:{k:r[k] for k in ['seed_id','config','inner_validation']} for n,r in leaders.items()},flush=True)

def evaluate():
 assert all(sha(f)==h for f,h in read(P/'SOURCE_HASHES.json').items());h=sha(P/'FROZEN.json');_,packs,px,opening,c0,styles=s.load();ms=old.masks(69);results={}
 for name,row in read(P/'FROZEN.json')['leaders'].items():
  seed=row['seed'];d=s.apply_gate(packs[seed['key']],seed['menu'],seed['gate']);c=a.context_for(d,c0);X,spec=a.design(d,s.MENUS[seed['menu']],'legacy')
  scores,strength,local,W=a.scores_for(row['rule'],X,spec,d,c,styles[seed['partition']].to_numpy());sim=policy.simulate(c,scores,strength,row['config'],detailed=True)
  met,slices,eq,trades=old.performance(px,opening,c,sim,scores,'close');os=policy.simulate(c,scores,strength,row['config'],execution='open');om,oss,_,_=old.performance(px,opening,c,os,scores,'open')
  matching={k:policy.matching(sim['choices'],scores,c,ms[k]) for k in ['training','validation','all']}
  results[name]=dict(metrics=met,slices=slices,next_open=om,next_open_slices=oss,matching=matching,rule=row)
  dump(name+'_TRADES.json',trades);dump(name+'_DECISIONS.json',sim['events']);dump(name+'_DAILY.json',[dict(date=str(t.date()),equity=float(v)) for t,v in eq.items()])
  if name=='overall':
   er=dict(seed,config=row['config'],rule=row['rule']);months,txt=old.explain(er,d,c,styles,scores,local,X,W,sim);dump('MONTHLY_EXPLANATIONS.json',months);(P/'MONTHLY_EXPLANATIONS.txt').write_text(txt)
  print(name,dict(cagr=met['cagr'],sharpe=met['sharpe'],validation=slices['validation34'],matching=matching),flush=True)
 prior=read(O/'monthly_style_inverse/RESULTS.json')
 for k in ['author','qqq']:results[k]=prior[k]
 dump('RESULTS.json',results);assert sha(P/'FROZEN.json')==h;changes=[f for f,v in read(P/'PROTECTED_BEFORE.json').items() if sha(f)!=v];assert not changes,changes
 dump('VALIDATION.json',dict(protected_changes=changes,source_hashes_verified=True,frozen_hash=h,inner_folds=a.FOLDS,split='first35/later34',zero_cost=True,own_position_state_only=True,no_intramonth_trades=True,independent_replay_verified=True,not_pristine_oos=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','train','evaluate']);globals()[p.parse_args().phase]()
