"""Validate tiny improvements and risk tradeoffs; do not force a new winner."""
import json
import numpy as np
import growth as g
z=g.z;P=g.P/'validation'
CASES={'control25':{},'crash_entry':g.VARIANTS['crash_entry'],'strong_cap60':g.VARIANTS['strong_cap60']}
CONFIG=dict(declared_after_first_two_stages=True,declared_before_validation=True,cases=CASES,cost50=True,
 starts=['2022-01-03','2024-01-02'],cost_bps=[10,30],crash_neighbors=[1.25,1.75],
 explanation='Crash entry improves CAGR only0.061percentagepoint. Strong cap60 is highest first-stage full CAGR but fails35%DD gate. Validate both without promotion or reselection.',
 prefix_end='2023-12-29',prefix_cases=['control25','strong_cap60'],all_history_reused=True,data_certified=False,performance_publishable=False,no_observation_changes=True)
def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite validation')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=g.protected_hashes();d,f,base,gaps,x,ctx,_=g.prepare(P);summary={}
 for name,extra in CASES.items():
  cfg={**g.BASE,**extra};z.save(P,f'{name}_cost50',g.run(cfg,50,d,f,base,gaps,x,ctx),summary)
  for start in CONFIG['starts']:
   for cost in [10,30]:z.save(P,f'fresh{start[:4]}_{name}_cost{cost}',g.run(cfg,cost,d,f,base,gaps,x,ctx,start=start),summary)
 for sigma in [1.25,1.75]:
  altered=dict(ctx);spy=d['adj']['SPY'];ret=ctx['ret']['SPY'];crash=(spy/spy.shift(63)-1).lt(0)&ret.rolling(20).std().gt(sigma*ret.rolling(126).std());altered['no_crash']=~crash
  for cost in [10,30]:z.save(P,f'crash_sigma{str(sigma).replace(".","")}_cost{cost}',g.run({**g.BASE,**CASES['crash_entry']},cost,d,f,base,gaps,x,altered),summary)
 for name in CONFIG['prefix_cases']:
  out=g.run({**g.BASE,**CASES[name]},10,d,f,base,gaps,x,ctx,end=CONFIG['prefix_end']);ref=json.loads((g.P/f'{name}_cost10.json').read_text())
  assert out['orders']==[v for v in ref['orders'] if v['date']<=CONFIG['prefix_end']]
  np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in ref['daily'] if v['date']<=CONFIG['prefix_end']],rtol=1e-12,atol=1e-7)
  z.save(P,f'prefix2023_{name}_cost10',out,summary)
 assert protected==g.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,prefix_invariant=True,no_leverage=True),indent=2)+'\n');print('VALIDATION_COMPLETE',len(summary),flush=True)
if __name__=='__main__':main()
