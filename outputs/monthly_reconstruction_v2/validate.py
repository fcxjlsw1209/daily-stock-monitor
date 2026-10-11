import json
import numpy as np,pandas as pd
from research import P,O,load,raw_features,Simulator,train_models,fixed_rules,dump,metrics,sha

def net_metrics(nav):
 # Initial capital is 1 before first entry fee; do not normalize that fee away.
 r=nav.pct_change().dropna();r=pd.concat([pd.Series([float(nav.iloc[0]-1)]),r]);n=len(nav)-1
 return dict(start=str(nav.index[0].date()),end=str(nav.index[-1].date()),wealth=float(nav.iloc[-1]),cagr_252=float(nav.iloc[-1]**(252/n)-1),max_dd=float((nav/np.maximum(1,nav.cummax())-1).min()),sharpe=float(r.mean()/r.std()*np.sqrt(252)))

def main():
 px,fs,pools=load();raw=raw_features(px,fs)
 # Re-fit an actual model with all data strictly truncated after June2020.
 full,fulltree,audit=train_models(px,raw,pools['core17'],36,'rank');end=pd.Timestamp('2020-07-01');smallpx=px.loc[:end];smallfs={k:v.loc[:end] for k,v in fs.items()};smaller=raw_features(smallpx,smallfs);part,parttree,_=train_models(smallpx,smaller,pools['core17'].loc[:end],36,'rank');idx=part.index
 diff=(full.reindex(idx)-part).abs().max().max();treediff=(fulltree.reindex(idx)-parttree).abs().max().max();assert diff<1e-10 and treediff<1e-10
 # Prefix features agree independently of model. No future normalization leakage.
 fdelta=max(float((raw[k].reindex(idx)-smaller[k]).abs().max().max()) for k in smaller)
 assert fdelta<1e-10
 controls={};curves={}
 for pool,name in [('theme17','m252_42'),('theme17','m294_63'),('core17','m252_42')]:
  member=pools[pool];rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()};score=fixed_rules(rank,raw)[name];sim=Simulator(px,fs,member);gate=raw['m252_42']>0
  for lag in [0,1]:
   for cost in [0,10,30]:
    nav,orders=sim.run(score,gate,cost_bps=cost,signal_lag=lag);tag=f'{pool}_{name}_lag{lag}_cost{cost}';controls[tag]=dict(**net_metrics(nav),signal_lag=lag,cost_bps_roundtrip=cost,embedded_leverage=False);curves[tag]=nav
 dump(P/'EXECUTION_CONTROLS.json',controls);pd.DataFrame(curves).to_csv(P/'EXECUTION_CONTROL_NAV.csv.gz')
 source=json.loads((P/'SOURCE_HASHES.json').read_text());changes=[f for f,h in source.items() if sha(f)!=h];protected=json.loads((O/'monthly_curve_reverse/PROTECTED_BEFORE.json').read_text());protected_changes=[f for f,h in protected.items() if sha(f)!=h];tests=(P/'TESTS.txt').read_text();assert 'Ran 12 tests' in tests and tests.rstrip().endswith('OK');assert not changes and not protected_changes
 result=dict(prefix_cutoff='2020-07-01',feature_prefix_max_difference=fdelta,ridge_prefix_max_difference=float(diff),stump_prefix_max_difference=float(treediff),all_model_labels_mature_at_signal=True,unit_tests_passed=12,execution_control_cases=len(controls),protected_files=len(protected),protected_unchanged=True,source_hashes_verified=len(source),author_strategy_reproduced=False,causal_target_achieved=False,independent_oos=False,teacher_path_uses_future=True)
 dump(P/'VALIDATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
