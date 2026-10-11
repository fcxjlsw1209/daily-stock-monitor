"""Compare new feature path with frozen strategies; not a PIT historical backtest."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'factor_research'))
from backtest_factors import load_inputs
from backtest_swing import MAP
from features import six_features

def imported(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 data,raw=load_inputs();a=raw['adj_close'];membership=pd.DataFrame(True,index=a.index,columns=a.columns)
 sectors=pd.DataFrame({s:pd.Series(MAP[data.sector[s]],index=a.index) for s in a.columns})
 actual=six_features(data,raw,membership,sectors);checks=[]
 paths={'baseline':'daily_stock_monitor','shadow':'pressure_shadow_monitor','resilience':'resilience_shadow_monitor'}
 for name,folder in paths.items():
  old=imported('frozen_'+name,P.parent/folder/'strategy.py').features(data,raw)
  pd.testing.assert_frame_equal(actual[name]['score'],old['score'])
  pd.testing.assert_frame_equal(actual[name]['mask'],old['masks']['broad']);checks.append(name)
 for name,folder in [('accel','accel_shadow_monitor'),('high52','high52_shadow_monitor'),('accel40','accel40_shadow_monitor')]:
  module=imported('frozen_'+name,P.parent/folder/'frontier_features.py');old=module.all_features(data,raw)['high52' if name=='high52' else 'accel']
  pd.testing.assert_frame_equal(actual[name]['score'],old['score'])
  pd.testing.assert_frame_equal(actual[name]['mask'],old['masks']['trend']);checks.append(name)
 # Time-varying synthetic membership tests only, not genuine market-cap history.
 dynamic=membership.copy();dynamic.loc['2022':,a.columns[:20]]=False
 full=six_features(data,raw,dynamic,sectors)
 cut='2023-12-29'
 from dataclasses import replace
 small=replace(data,**{k:getattr(data,k).loc[:cut] for k in ['close','opening','dividends','volume','sector_prices']})
 prefix=six_features(small,{k:v.loc[:cut] for k,v in raw.items()},dynamic.loc[:cut],sectors.loc[:cut])
 for name,f in full.items():
  for key in ['score','mask']:pd.testing.assert_frame_equal(f[key].loc[:cut],prefix[name][key])
  assert not (f['mask']&~dynamic).any().any()
  assert not f['score'].where(~dynamic).notna().any().any()
 out=dict(frozen_feature_equivalence=checks,dynamic_prefix_invariance='passed',no_nonmember_scores='passed',genuine_pit_backtest_run=False,live_migration_complete=False)
 (P/'feature_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
