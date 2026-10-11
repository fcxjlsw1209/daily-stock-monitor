"""Same-close model AND execution diagnostic; no tradeability claims."""
import rebuild as b
import joint
from clock_execution import run_account
import numpy as np,pandas as pd,json
P=b.P/'close_target';P.mkdir(exist_ok=True)

def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('No overwrite')
 b.dump(P/'CONFIG.json',dict(declared_before_results=True,previous74runs_seen=True,cases=['experts','risk','joint_raw','joint_rank'],cost_bps=[0,10],model_target='Own-day close to close5sessions later; same-close conditional outcome/path history; conservative six-session feature and training maturity retained.',execution='Own-day final OHLCV signal filled same close; remains non-executable diagnostic without intraday signal freezing/auction evidence.',other_rules='Unchanged reconstruction10daycap,3ATRprofit,2ATRstop,fourgroups,replace after2sessions,positive modeled opportunities.',no_optimization=True,data_certified=False))
 _,d,_=b.prepare();f=b.features(d,'same_close_diagnostic');idx=f['valid'].index;cols=f['valid'].columns;models={};audit={}
 for mode in ['experts','risk','joint_raw','joint_rank']:
  models[mode]={}
  for pop in ['entry','hold']:
   if mode.startswith('joint'):
    pred,a=joint.predictions(f,mode.split('_')[1],pop)
   else:
    pr,_,a=b.ridge_fit_predict(f['X'],f['y'].to_numpy(),(f['event'] if pop=='entry' else f['strong']).to_numpy(),idx,f['vol'].to_numpy() if mode=='risk' else None);pred=pd.DataFrame(pr,index=idx,columns=cols)
   models[mode][pop+'_experts']=pred;audit[mode+'_'+pop]=a
  print('close-target trained',mode,flush=True)
 b.dump(P/'TRAINING_AUDIT.json',audit);summary={};(P/'runs').mkdir(exist_ok=True)
 for mode,model in models.items():
  for cost in [0,10]:
   key=f'{mode}_same_close_diagnostic_cost{cost}';pl,ff=b.planner(b.BASE,f,model,'same_close_diagnostic');out=run_account(d,ff,pl,b.START,b.END,cost,'same_close_diagnostic');b.dump(P/'runs'/f'{key}.json',out)
   if out['status']!='complete':raise ValueError(out.get('error'))
   summary[key]=b.metrics(out);b.dump(P/'SUMMARY.json',summary);print(key,summary[key]['full'],flush=True)
 b.dump(P/'CHECKS.json',dict(complete=True,runs=len(summary)))
if __name__=='__main__':main()
