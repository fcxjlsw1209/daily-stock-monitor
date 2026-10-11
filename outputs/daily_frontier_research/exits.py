"""Seventh batch: daily management with rebound continuation and causal trailing exits."""
import json
import frontier as z
P=z.P/'exits'
VARIANTS={
 'control':{},
 'no_profit_cap10':dict(tp=None),
 'profit6_cap20':dict(tp=6.,cap=20),
 'no_profit_cap20':dict(tp=None,cap=20),
 'no_profit_cap30':dict(tp=None,cap=30),
 'no_profit_cap20_protect':dict(tp=None,cap=20,protect_winner=True),
 'no_profit_cap20_gap20':dict(tp=None,cap=20,gap=.20),
 'trail15_cap20':dict(tp=None,cap=20,trail_atr=1.5),
 'trail2_cap20':dict(tp=None,cap=20,trail_atr=2.),
 'trail3_cap20':dict(tp=None,cap=20,trail_atr=3.),
 'trail2_cap30':dict(tp=None,cap=30,trail_atr=2.),
 'trail3_cap30':dict(tp=None,cap=30,trail_atr=3.),
 'trail2_protect':dict(tp=None,cap=20,trail_atr=2.,protect_winner=True),
 'trail3_protect':dict(tp=None,cap=20,trail_atr=3.,protect_winner=True),
 'trail3_stop25':dict(tp=None,cap=20,trail_atr=3.,sl=2.5),
 'ma5_cap20':dict(tp=None,cap=20,ma_exit=5),
 'ma10_cap20':dict(tp=None,cap=20,ma_exit=10),
 'ma20_cap30':dict(tp=None,cap=30,ma_exit=20),
 'no_profit_cap20_lone':dict(tp=None,cap=20,keep_lone=True),
 'trail3_cap20_lone':dict(tp=None,cap=20,trail_atr=3.,keep_lone=True)}
CONFIG=dict(declared_after_prior_batches=True,declared_before_this_batch_results=True,variants=VARIANTS,cost_bps=[10,30],
 entry='Unchanged baseline four25%groups and original pullback/score.75/shock2 gates. Long-only,max3,min2 except explicit retain_lone variants; no leverage.',
 trailing='High-water CLOSE from entry through previous session, in consistent split-adjusted signal units; activate after max close>=entry+1entryATR; next-open exit below peak minus trailing multiple*entryATR. No intraday stop fill.',
 moving_average='Simple mean of last5/10/20 closes through prior close; exit only age>=2; next-open fill. Original protective stop and rank exit remain.',
 scope='No fixed profit target uses tp=null; each longer cap remains actively checked daily, not monthly fixed sleeve. Hard stop2ATR except explicit2.5, rank.35, optional paired replacement unchanged unless stated.',
 data_certified=False,performance_publishable=False,no_observation_changes=True,target_policy=z.CONFIG['target_policy'],development_selection=z.CONFIG['development_selection'])
def main():
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite exits batch')
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
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,default_orders_reproduced=True,no_leverage=True),indent=2)+'\n')
 print('EXITS_COMPLETE',len(summary),'selected',selected,'hits',hits,flush=True)
if __name__=='__main__':main()
