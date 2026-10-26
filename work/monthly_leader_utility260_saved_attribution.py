"""Post-selection diagnostic using saved zero-cost NAV only; no fitting or trades."""
from pathlib import Path
import json, hashlib, math
R=Path(__file__).resolve().parents[1]
O=R/'outputs/monthly_leader_utility260'
A=('incumbent_local67','unconditional_leader_utility','state_leader_utility')
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
expected='00a4104d1c237ae408f1acff9a7c0215304f9b3d9bac3747acbefc2a485e52fe'
assert H(O/'results/COMPLETION.json')==expected
c=read(O/'results/COMPLETION.json')
for name,digest in c['artifacts_sha256'].items():assert H(O/'results'/name)==digest
meta=read(O/'prepared/META.json');orders=read(O/'prepared/ORDERS.json')
nav={a:{r['date']:r['equity'] for r in read(O/'results'/(a+'_same_close_zero_DAILY.json'))} for a in A}
result=read(O/'results/RESULTS.json');out={}
for candidate,reference in ((A[1],A[0]),(A[2],A[0]),(A[2],A[1])):
 rows=[]
 for i,(start,end) in enumerate(zip(meta['dates'],meta['ends'])):
  gr=nav[reference][end]/nav[reference][start]
  gc=nav[candidate][end]/nav[candidate][start]
  logdiff=math.log(gc/gr)
  r=orders[reference][i]['symbols'];s=orders[candidate][i]['symbols'];changed=set(r)!=set(s)
  if not changed:assert abs(logdiff)<1e-11
  if changed:rows.append(dict(signal_date=start,end=end,segment='first88' if i<88 else 'later34',reference=r,candidate=s,reference_return=gr-1.,candidate_return=gc-1.,log_relative_wealth_change=logdiff))
 total=sum(r['log_relative_wealth_change'] for r in rows)
 direct=math.log(result['arms'][candidate]['same_close_zero']['metrics']['final_equity']/result['arms'][reference]['same_close_zero']['metrics']['final_equity'])
 assert abs(total-direct)<1e-10
 out[candidate+'__vs__'+reference]=dict(changed_months=len(rows),relative_terminal_wealth=math.exp(total),first88_relative_wealth=math.exp(sum(r['log_relative_wealth_change'] for r in rows if r['segment']=='first88')),later34_relative_wealth=math.exp(sum(r['log_relative_wealth_change'] for r in rows if r['segment']=='later34')),changed=rows,largest_drags=sorted(rows,key=lambda r:r['log_relative_wealth_change'])[:5])
p=R/'work/monthly_leader_utility260_SAVED_ATTRIBUTION.json';assert not p.exists()
p.write_text(json.dumps(dict(schema='monthly_leader_utility260_saved_attribution_v1',post_selection=True,used_for_selection=False,new_models_fitted=0,new_orders_computed=0,zero_cost_only=True,results_completion_sha256=expected,comparisons=out),indent=2)+'\n')
print(json.dumps({k:{q:v[q] for q in ('changed_months','relative_terminal_wealth','first88_relative_wealth','later34_relative_wealth','largest_drags')} for k,v in out.items()}))
