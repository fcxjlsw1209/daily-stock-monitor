"""Validate observed stop neighbors without replacing the declared first-hit candidate."""
import json
import frontier as z
P=z.P/'neighbor_validation'
VARIANTS={'stop15':dict(tp=None,cap=30,sl=1.5),'stop25':dict(tp=None,cap=30,sl=2.5)}
CONFIG=dict(declared_after_robustness=True,variants=VARIANTS,stress_cost_bps=[50,100],cash_restarts=['2024-01-02','2025-10-06'],cost_bps=[10,30],
 reason='Neighbor stop1.5 had highest recent return; stop2.5 had higher full CAGR. Validate both, disclose data snooping, keep first-hit candidate unchanged.',
 no_further_search=True,no_observation_changes=True,data_certified=False,performance_publishable=False)
def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite neighbor validation')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=z.protected_hashes()
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 x=z.extras(d,f);summary={}
 for name,overlay in VARIANTS.items():
  for cost in [50,100]:z.save(P,f'{name}_cost{cost}',z.run({**z.BASE,**overlay},cost,d,f,base,gaps,x),summary)
  for start,label in [('2024-01-02','fresh2024'),('2025-10-06','fresh252')]:
   for cost in [10,30]:z.save(P,f'{label}_{name}_cost{cost}',z.run({**z.BASE,**overlay},cost,d,f,base,gaps,x,start=start),summary)
 assert protected==z.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,no_leverage=True),indent=2)+'\n')
 print('NEIGHBORS_VALIDATED',len(summary),flush=True)
if __name__=='__main__':main()
