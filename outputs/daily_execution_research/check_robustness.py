"""Declared follow-up after initial results; no reselecting on recent returns."""
from pathlib import Path
import json,hashlib
import numpy as np
import run_execution as e
P=Path(__file__).resolve().parent/'robustness'

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite robustness results')
 configs=[]
 for gap in [.10,.20]:
  for cost in [10,30]:configs.append(dict(key=f'gap{int(gap*100)}_cost{cost}',kind='matched_replacement',cost=cost,gap=gap,start='2019-01-02',sector_cap=None))
 for cost in [10,30]:configs.append(dict(key=f'matched_sector2_cost{cost}',kind='matched_replacement',cost=cost,gap=.15,start='2019-01-02',sector_cap=2))
 for kind in ['baseline','validated_base','matched_replacement']:
  configs.append(dict(key=f'{kind}_cost50',kind=kind,cost=50,gap=.15,start='2019-01-02',sector_cap=None))
  for cost in [10,30]:configs.append(dict(key=f'fresh2024_{kind}_cost{cost}',kind=kind,cost=cost,gap=.15,start='2024-01-02',sector_cap=None))
 (P/'CONFIG.json').write_text(json.dumps(dict(runs=configs,phase='Post initial-development selection checks',data_certified=False,
  rule='Do not promote neighboring gap or combination based on recent performance; retain gap.15 and show every outcome',
  combination='Matched replacement plus sector2 entered because each passed declared development/cost criterion. This combination was proposed after initial results.',
  fresh_start='Start100000cash on2024-01-02, train/validate using pre2024 mature history as needed; 2024+ already viewed, not pristine OOS'),indent=2)+'\n')
 protected=e.protected_hashes();source_hash=hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
 m,d,f,base,choice,original,validated,audit,choices=e.prepare_features();summary={}
 for cfg in configs:
  sc=original if cfg['kind']=='baseline' else validated;mask=f['event']&sc.ge(.75)
  out=e.run_variant(cfg['kind'],cfg['cost'],d,f,sc,mask,cfg['start'],'2026-10-06',gap=cfg['gap'],sector_cap=cfg['sector_cap'])
  key=cfg['key'];(P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
  summary[key]=e.prior.metrics(out);summary[key]['diagnostics']=e.diagnostics(out)
  (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(key,'CAGR',summary[key]['full']['cagr'],'DD',summary[key]['full']['max_drawdown'],'LAST',summary[key]['last252']['return_'],flush=True)
 assert protected==e.protected_hashes();assert source_hash==hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),protected_unchanged=True,implementation_sha256=source_hash,data_certified=False),indent=2)+'\n')
 print('DONE',flush=True)

if __name__=='__main__':main()
