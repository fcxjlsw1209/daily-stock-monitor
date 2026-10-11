"""Follow-up scope frozen after stage1, retaining its development selection."""
from pathlib import Path
import json,hashlib
import timing as t
P=t.P/'robustness'

def main():
 P.mkdir(exist_ok=True)
 if (P/'SUMMARY.json').exists():raise RuntimeError('Do not overwrite completed follow-up')
 selection=json.loads((t.P/'SELECTION.json').read_text())
 assert selection['selected']=='stop_pause3' and set(selection['eligible'])=={'shock_pause2','stop_pause3'}
 cfg=dict(declared_after_stage1=True,retain_selection='stop_pause3',candidates=selection['eligible'],
  costs_stress=[50],cash_restart='2024-01-02',neighbors={'shock_days':[1,3],'stop_pause':[2,4]},
  combinations=['shock_pause2 plus stop_pause3','shock_pause2 with fixed gap .10'],
  reason='Both passed the original development/cost filter. Follow up both, no winner chosen using recent returns or neighbor outcomes.',
  no_reselection=True,no_observation_changes=True,data_certified=False)
 (P/'CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
 protected=t.protected_hashes();original={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in t.P.glob('*_cost*.json')}
 d,f,score,gaps,_=t.prepare();summary={}
 def save(key,out):
  (P/(key+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')
  summary[key]=t.a.old.prior.metrics(out);summary[key]['diagnostics']=t.a.old.diagnostics(out)
  (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
  v=summary[key];print(key,'CAGR',v['full']['cagr'],'DD',v['full']['max_drawdown'],'RECENT',v['last252']['return_'],flush=True)
 for kind in ['stable_control','shock_pause2','stop_pause3']:
  save(kind+'_cost50',t.run(kind,50,d,f,score,gaps))
  for cost in [10,30]:save('fresh2024_'+kind+f'_cost{cost}',t.run(kind,cost,d,f,score,gaps,start='2024-01-02'))
 for n in [1,3]:
  for cost in [10,30]:save(f'shock_days{n}_cost{cost}',t.run('shock_pause2',cost,d,f,score,gaps,shock_days=n))
 for n in [2,4]:
  for cost in [10,30]:save(f'stop_pause{n}_cost{cost}',t.run('stop_pause3',cost,d,f,score,gaps,pause=n))
 # Causal mask combination; leave original f untouched. No refit or altered score.
 masked=dict(f);masked['event']=t.entry_inputs(d,f,score,'shock_pause2')[2]
 for cost in [10,30]:
  save(f'shock_plus_stop_cost{cost}',t.run('stop_pause3',cost,d,masked,score,gaps))
  save(f'shock_fixed10_cost{cost}',t.run('fixed10',cost,d,masked,score,gaps))
 assert protected==t.protected_hashes()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in original.items())
 (P/'CHECKS.json').write_text(json.dumps(dict(complete=True,runs=len(summary),original_results_unchanged=True,protected_unchanged=True,selection_retained=True),indent=2)+'\n')
 print('FOLLOWUP_COMPLETE',len(summary),flush=True)

if __name__=='__main__':main()
