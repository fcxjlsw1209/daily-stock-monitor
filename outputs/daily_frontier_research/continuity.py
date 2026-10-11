"""Sixth batch explicitly changes minimum-position behavior, never hidden in controls."""
from pathlib import Path
import json
import frontier as z
P=z.P/'continuity'
VARIANTS={
 'control':{},
 'retain_lone':dict(keep_lone=True),
 'slots_thirds':dict(min_positions=1,budget_floor=3),
 'slots_halves':dict(min_positions=1,budget_floor=2),
 'retain_lone_no_shock':dict(keep_lone=True,shock=False),
 'slots_thirds_no_shock':dict(min_positions=1,budget_floor=3,shock=False),
 'slots_halves_no_shock':dict(min_positions=1,budget_floor=2,shock=False),
 'slots_halves_stop25':dict(min_positions=1,budget_floor=2,sl=2.5),
 'slots_thirds_stop25':dict(min_positions=1,budget_floor=3,sl=2.5),
 'retain_lone_stop25':dict(keep_lone=True,sl=2.5),
 'retain_lone_cap15':dict(keep_lone=True,cap=15),
 'retain_lone_profit4':dict(keep_lone=True,tp=4.),
 'slots_halves_cap15':dict(min_positions=1,budget_floor=2,cap=15)}
CONFIG=dict(declared_after_176_runs=True,declared_before_this_batch_results=True,variants=VARIANTS,cost_bps=[10,30],
 explicit_scope_change='Original min2 rule is a hypothesis, not an immutable live-rule migration. These cases can retain or initiate a lone holding; report separately from original rule.',
 retain_lone='Only remove forced fewer_than_two_holdings liquidation when at least one incumbent survives mandatory exits. Still require2 targets to start from empty; no lone replacement after all incumbents exit.',
 slots_thirds='Allow1 qualified candidate; each new entry budget prior equity/max(target_count,3). No topping up incumbents. Max3; cash retained.',
 slots_halves='Allow1 qualified candidate; each new entry budget prior equity/max(target_count,2), so single entry at most50%,two50%each,three33.3%each; no incumbent topup.',
 unchanged='Same event eligibility,score.75,four25%groups,Ridge gating,price history,next-open fills,costs and issuer uniqueness. All other overrides explicit in variants.',
 no_leverage=True,no_observation_changes=True,data_certified=False,performance_publishable=False,
 target_policy=z.CONFIG['target_policy'],development_selection=z.CONFIG['development_selection'])

def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite continuity batch')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=z.protected_hashes()
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 x=z.extras(d,f);summary={}
 for name,overlay in VARIANTS.items():
  for cost in CONFIG['cost_bps']:
   out=z.run({**z.BASE,**overlay},cost,d,f,base,gaps,x)
   if name=='control':
    ref=json.loads((z.P/f'shock_control_cost{cost}.json').read_text());assert out['orders']==ref['orders'];z.np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in ref['daily']],rtol=1e-12,atol=1e-7)
   z.save(P,f'{name}_cost{cost}',out,summary)
 control=summary['control_cost30'];eligible=[k for k in VARIANTS if k!='control' and summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=control['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 hits=[k for k,v in summary.items() if v['last252']['return_']>=1]
 (P/'SELECTION.json').write_text(json.dumps(dict(development_selected=selected,eligible=eligible,in_sample_target_hits=hits),indent=2)+'\n')
 assert protected==z.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,default_orders_reproduced=True,scope_change_disclosed=True,no_leverage=True),indent=2)+'\n')
 print('CONTINUITY_COMPLETE',len(summary),'selected',selected,'hits',hits,flush=True)

if __name__=='__main__':main()
