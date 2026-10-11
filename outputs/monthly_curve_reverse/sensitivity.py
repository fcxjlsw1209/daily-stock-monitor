"""Universe and entry-gate sensitivity, NOT a search for target annual return."""
import json
import numpy as np,pandas as pd
from extract import P,O,read_prices,sha
from inverse import load_observations,run_month

def main():
 px,member,_=read_prices();f=O/'daily_rebuild_research/snapshot_pool/membership.csv.gz';old=pd.read_csv(f,index_col=0,parse_dates=True).reindex(index=member.index,columns=member.columns,fill_value=False)
 cal=json.loads((P/'CALIBRATION.json').read_text());obs=load_observations();mom=px.shift(42)/px.shift(252)-1;rank=mom.where(member).rank(axis=1,pct=True)
 months=['2020-07','2020-08','2020-11','2023-08','2025-09','2026-03','2026-04','2026-05','2026-06']
 variants={'prior_snapshot_top32':old,'positive_r12_2':member&(mom>0),'top_half_positive_r12_2':member&(mom>0)&(rank>=.5)}
 config=dict(months=months,purpose='Inspect the most discriminating or persistent-stock months from first pass. Ex-post-selected diagnostic months, no statistical significance or OOS claim.',variants=['Use prior market-cap snapshot unscaled, versus price-scaled main pool','Require positive 12-to-2-month momentum','Also require upper-half 12-to-2 momentum'],snapshot_source=str(f),snapshot_sha256=sha(f))
 (P/'SENSITIVITY_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n');results={}
 for name,m in variants.items():
  results[name]=[]
  for month in months:
   r=run_month(month,px,m,obs,cal);results[name].append(r);print(name,month,r.get('best'),flush=True)
 (P/'SENSITIVITY.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
