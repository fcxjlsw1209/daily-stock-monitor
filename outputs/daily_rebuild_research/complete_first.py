import rebuild as b
import json,numpy as np,pandas as pd
from clock_execution import run_account
from repair_actions import patch
P=b.P
m,d,old=b.prepare();d=patch(d);f=b.features(d);cached=np.load(P/'PREDICTIONS.npz');models={k:pd.DataFrame(cached[k],index=f['valid'].index,columns=f['valid'].columns) for k in cached.files}
summary=json.loads((P/'SUMMARY.json').read_text())
# Preserve the stopped run and exact failure; complete four cases with same signals.
failed=P/'runs/experts_wide_gate_next_open_cost0.json'
if failed.exists() and json.loads(failed.read_text())['status']=='blocked':
 target=P/'runs/experts_wide_gate_next_open_cost0.blocked.json'
 if not target.exists():failed.rename(target)
for clock,cost in [('next_open',0),('next_open',10),('next_open',30),('same_close_diagnostic',0)]:
 key=f'experts_wide_gate_{clock}_cost{cost}'
 if key in summary:continue
 cfg={**b.BASE,**b.CASES['experts_wide_gate']};pl,ff=b.planner(cfg,f,models,clock)
 out=run_account(d,ff,pl,b.START,b.END,cost,clock);b.save(key,out,summary)
b.dump(P/'CHECKS.json',dict(complete=True,runs=len(summary),execution_repair='PFE distribution, recorded in PFE_REPAIR.json',protected_unchanged=json.loads((P/'SOURCE_HASHES.json').read_text())==b.protected(),data_certified=False))
assert len(summary)==40
