"""Fifth disclosed batch: factorial interactions and explicit two-name concentration."""
from pathlib import Path
import itertools,json,hashlib
import frontier as z
P=z.P/'interactions'
VARIANTS={f'th{int(th*100)}_sl{int(sl*10)}_age{age}':dict(threshold=th,sl=sl,min_age=age)
 for th,sl,age in itertools.product([.73,.75],[2.,2.5],[2,3])}
VARIANTS.update(two_names=dict(positions=2),two_names_th73=dict(positions=2,threshold=.73),
 two_names_gap10=dict(positions=2,gap=.1),two_names_no_shock=dict(positions=2,shock=False))
CONFIG=dict(declared_after_152_runs=True,declared_before_this_batch_results=True,variants=VARIANTS,cost_bps=[10,30],
 data_certified=False,performance_publishable=False,no_observer_changes=True,no_leverage=True,group_weights=[.25]*4,
 factorial='Threshold.73/.75,stop2/2.5ATR,replacement minimum age2/3: all8 combinations, no selectively omitted failures.',
 concentration='Two-name cases cap holdings at2 despite3 eligible, initial equal budgets; explicitly different concentration risk, not an alpha-only improvement.',
 target_policy=z.CONFIG['target_policy'],development_selection=z.CONFIG['development_selection'])

def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite interaction batch')
 (P/'CONFIG.json').write_text(json.dumps(CONFIG,indent=2)+'\n');protected=z.protected_hashes()
 oldp=z.t.P;z.t.P=P
 try:d,f,base,gaps,_=z.t.prepare()
 finally:z.t.P=oldp
 x=z.extras(d,f);summary={}
 for name,overlay in VARIANTS.items():
  for cost in CONFIG['cost_bps']:
   out=z.run({**z.BASE,**overlay},cost,d,f,base,gaps,x)
   if name=='th75_sl20_age2':
    ref=json.loads((z.P/f'shock_control_cost{cost}.json').read_text())
    assert out['orders']==ref['orders'];z.np.testing.assert_allclose([v['equity'] for v in out['daily']],[v['equity'] for v in ref['daily']],rtol=1e-12,atol=1e-7)
   z.save(P,f'{name}_cost{cost}',out,summary)
 control=json.loads((z.P/'SUMMARY.json').read_text())['shock_control_cost30'];eligible=[k for k in VARIANTS if summary[k+'_cost10']['development']['max_drawdown']>=-.35 and summary[k+'_cost30']['development']['cagr']>=control['development']['cagr']]
 selected=max(eligible,key=lambda k:summary[k+'_cost10']['development']['cagr']) if eligible else None
 hits=[k for k,v in summary.items() if v['last252']['return_']>=1]
 (P/'SELECTION.json').write_text(json.dumps(dict(development_selected=selected,eligible=eligible,in_sample_target_hits=hits),indent=2)+'\n')
 assert protected==z.protected_hashes()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,default_orders_reproduced=True,no_leverage=True,two_name_risk_disclosed=True),indent=2)+'\n')
 print('INTERACTIONS_COMPLETE',len(summary),'selected',selected,'hits',hits,flush=True)

if __name__=='__main__':main()
