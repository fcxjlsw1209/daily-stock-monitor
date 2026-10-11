"""Declared second-stage risk/entry neighborhood, all outcomes retained."""
from pathlib import Path
import json,hashlib
import frontier as z
P=z.P/'expansion'
VARIANTS={
 'stop225':dict(sl=2.25),'stop275':dict(sl=2.75),'stop3':dict(sl=3.),'stop35':dict(sl=3.5),'stop4':dict(sl=4.),
 'profit25':dict(tp=2.5),'profit35':dict(tp=3.5),
 'threshold73':dict(threshold=.73),'threshold74':dict(threshold=.74),'threshold76':dict(threshold=.76),'threshold77':dict(threshold=.77),
 'cap9':dict(cap=9),'cap11':dict(cap=11),'min_age5':dict(min_age=5),
 'stop25_age3':dict(sl=2.5,min_age=3),'stop3_age3':dict(sl=3.,min_age=3),
 'stop25_profit4':dict(sl=2.5,tp=4.),'stop3_profit4':dict(sl=3.,tp=4.),
 'stop3_cap12':dict(sl=3.,cap=12),'stop3_cap15':dict(sl=3.,cap=15)}
CONFIG=dict(stage='after partial first-stage exploration',declared_before_this_batch_results=True,
 variants=VARIANTS,cost_bps=[10,30],base=z.BASE,
 rationale='Prior2.5ATR stop had higher recent return but poorer drawdown; test broader stop sensitivity and its holding/turnover interactions. Prior threshold.72/.78 were poor, inspect intervening rounded neighborhoods without suppressing failures.',
 recent_goal_is_reused_history=True,no_robustness_claim_from_target_hit=True,no_observer_promotion=True,
 data_certified=False,performance_publishable=False,selection='Same development criterion as first-stage. Target hits separately identified as in-sample only.')

def declare():
 P.mkdir(exist_ok=True)
 p=P/'CONFIG.json'
 if p.exists():assert json.loads(p.read_text())==CONFIG
 else:p.write_text(json.dumps(CONFIG,indent=2)+'\n')

def main():
 declare()
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite second-stage outcomes')
 assert json.loads((z.P/'CHECKS.json').read_text())['complete']
 protected=z.protected_hashes();previous={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in z.P.glob('*_cost*.json')}
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 x=z.extras(d,f);summary={}
 for name,overlay in VARIANTS.items():
  for cost in CONFIG['cost_bps']:z.save(P,f'{name}_cost{cost}',z.run({**z.BASE,**overlay},cost,d,f,base,gaps,x),summary)
 original=json.loads((z.P/'SUMMARY.json').read_text());eligible=[k for k in VARIANTS if summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=original['shock_control_cost30']['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 hits=[k for k,v in summary.items() if v['last252']['return_']>=1]
 (P/'SELECTION.json').write_text(json.dumps(dict(development_selected=selected,eligible=eligible,in_sample_target_hits=hits,criterion=CONFIG['selection']),indent=2)+'\n')
 assert protected==z.protected_hashes()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in previous.items())
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,first_stage_unchanged=True,no_leverage=True),indent=2)+'\n')
 print('EXPANSION_COMPLETE',len(summary),'selected',selected,'hits',hits,flush=True)

if __name__=='__main__':
 import sys
 if '--declare-only' in sys.argv:declare()
 else:main()
