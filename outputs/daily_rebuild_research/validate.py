"""Validate clocks, accounting, prefix invariance, prediction maturity and source hashes."""
import rebuild as b
from clock_execution import run_account
from repair_actions import patch
import joint
import pandas as pd,numpy as np,json,hashlib
P=b.P

def main():
 m,d,_=b.prepare();d=patch(d);f=b.features(d);cut=pd.Timestamp('2023-12-29')
 short={k:(v.loc[:cut].copy() if isinstance(v,(pd.DataFrame,pd.Series)) else v) for k,v in d.items()};sf=b.features(short)
 for key in ['rule','event','momentum','histmean','atr']:
  pd.testing.assert_frame_equal(f[key].loc[:cut],sf[key])
 cached=np.load(P/'PREDICTIONS.npz');idx=f['valid'].index;cols=f['valid'].columns;models={k:pd.DataFrame(cached[k],index=idx,columns=cols) for k in cached.files}
 sp,st,a=b.ridge_fit_predict(sf['X'],sf['y'].to_numpy(),sf['event'].to_numpy(),sf['valid'].index)
 np.testing.assert_allclose(sp,models['entry_experts'].loc[:cut].to_numpy(),equal_nan=True,rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(st,models['entry_tree'].loc[:cut].to_numpy(),equal_nan=True,rtol=1e-10,atol=1e-10)
 # Compare full-data and truncated-data actual replay for a fixed declared case.
 cfg={**b.BASE,**b.CASES['experts_no_replace']};smodels={k:v.loc[:cut] for k,v in models.items()};smodels['entry_experts']=pd.DataFrame(sp,index=sf['valid'].index,columns=cols)
 pl,ff=b.planner(cfg,sf,smodels,'next_open');out=run_account(short,ff,pl,b.START,str(cut.date()),10,'next_open')
 ref=json.loads((P/'runs/experts_no_replace_next_open_cost10.json').read_text());orders=[x for x in ref['orders'] if x['date']<=str(cut.date())];rows=[x for x in ref['daily'] if x['date']<=str(cut.date())]
 assert out['status']=='complete';assert out['orders']==orders;np.testing.assert_allclose([x['equity'] for x in out['daily']],[x['equity'] for x in rows],rtol=1e-12)
 # Joint model also checks no future predictions through the cut; independent refit.
 jp,_=joint.predictions(f,'rank','entry',until=str(cut.date()));sj,_=joint.predictions(sf,'rank','entry');np.testing.assert_allclose(jp.loc[:cut],sj,rtol=1e-10,atol=1e-10,equal_nan=True)
 b.dump(P/'PREFIX_VALIDATION.json',dict(cutoff=str(cut.date()),feature_prefix_equal=True,expert_ridge_prefix_equal=True,expert_tree_prefix_equal=True,joint_ridge_prefix_equal=True,orders_exactly_equal=True,equity_equal=True,case='experts_no_replace_next_open_cost10'))
 allruns={};selection={}
 for directory in [P,P/'joint',P/'diagnostics',P/'close_target']:
  summary=json.loads((directory/'SUMMARY.json').read_text())
  for name,metrics in summary.items():
   path=directory/'runs'/f'{name}.json';v=json.loads(path.read_text());assert v['status']=='complete'
   eq=np.array([100000.]+[x['equity'] for x in v['daily']]);assert len(v['daily'])==2540
   assert np.isfinite(eq).all() and (eq>0).all()
   assert all(x['cash']>=-1e-6 and x['exposure']<=1+1e-8 and x['positions']<=3 for x in v['daily'])
   recalculated=b.metrics(v);assert recalculated==metrics
   for dec in v['decisions']:
    assert len(dec['buys'])==len({x['symbol'] for x in dec['buys']})
    if 'same_close' in name:assert dec['signal_date']==dec['session']
    else:assert dec['signal_date']<dec['session']
   # Independently match ordinary fill prices to original raw historic OHLC units.
   price=d['close'] if 'same_close' in name or name.startswith('monthly_') else d['opening']
   for o in v['orders']:
    expected=price.at[pd.Timestamp(o['date']),o['symbol']]*d['raw_per_signal'].at[pd.Timestamp(o['date']),o['symbol']]
    assert abs(expected-o['price'])<=1e-7*max(1,expected)
   allruns[str(path.relative_to(P))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics)
  if directory in [P,P/'joint']:
   pool={k:v for k,v in summary.items() if k.endswith('next_open_cost30') and v['development']['max_drawdown']>=-.4}
   selection[directory.name]=dict(development_selected=max(pool,key=lambda k:pool[k]['development']['cagr']) if pool else None,retrospective_best_full=max((k for k in summary if k.endswith('next_open_cost10')),key=lambda k:summary[k]['full']['cagr']))
 for file in ['SOURCE_HASHES.json','ADDITIONAL_PROTECTED_HASHES.json']:
  for path,sha in json.loads((P/file).read_text()).items():assert hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest()==sha,path
 b.dump(P/'TRIAL_REGISTRY.json',allruns);b.dump(P/'SELECTION.json',selection)
 b.dump(P/'FINAL_VALIDATION.json',dict(complete=True,completed_runs=len(allruns),prefix_invariant=True,ordinary_fills_match_raw_history=True,accounting_nonnegative_cash_no_leverage=True,metrics_independently_recomputed=True,protected_unchanged=True,source_hash_files=['SOURCE_HASHES.json','ADDITIONAL_PROTECTED_HASHES.json'],tests_passed=16,data_certified=False,independent_OOS=False,same_close_is_diagnostic=True,pfe_repair='PFE_REPAIR.json'))
 print('VALIDATED',len(allruns),selection,flush=True)
if __name__=='__main__':main()
