"""Freeze exact source and fixed-reference bindings; never computes factors."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
R=Path(__file__).resolve().parents[1]
U=R/'outputs/monthly_same_report145_union'
P=R/'outputs/monthly_same_report144'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
completion=read(P/'COMPLETION.json')
assert completion['status']=='completed_independently_verified'
state=read(R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')
assert state['current']=='monthly_same_report144' and not state['registered_queue']
assert len(state['completed'])==144 and sum(x['trials'] for x in state['completed'])==932
data=read(U/'DATA.json');audit=read(U/'AUDIT.json')
assert audit['passed'] and audit['stage']=='conditional_union_source_admission'
assert audit['data_sha256']==sha(U/'DATA.json')
assert data['counts']['events']==44 and data['counts']['raw_values']==176
assert all(audit[x]==0 for x in ('financial_calculations','model_fits','ranks_computed','returns_computed','network_requests'))
draft=R/'work/monthly_same_report145_proposal.DRAFT.json';spec=read(draft)
assert spec['round']==145 and spec['actual_new_returns']==0 and spec['actual_new_factors']==0
spec['status']='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION'
spec['created_utc']=datetime.now(timezone.utc).isoformat()
spec['draft_sha256']=sha(draft)
spec['design_review']={'path':'work/after144_round145_design_review.json','sha256':sha(R/'work/after144_round145_design_review.json')}
s=spec['sources'];s['directory']='outputs/monthly_same_report145_union'
s['dataset_sha256']=sha(U/'DATA.json');s['audit_sha256']=sha(U/'AUDIT.json')
s['hashes']={f:sha(U/f) for f in ('DATA.json','AUDIT.json','PLAN.json','MANIFEST.json')}
s['expected_census']=data['counts']
s['packages']=audit['packages']
s['conditional_limitations']=audit['limitations']
spec['reference144']['hashes']={f:sha(P/f) for f in ('PLAN.json','PRETRAIN_ORDERS.json','TRAIN_OPEN_TRIALS.json')}
spec['reference144']['completion']={'path':'outputs/monthly_same_report144/COMPLETION.json','sha256':sha(P/'COMPLETION.json')}
spec['target']='Selected model full122-month same-close zero-cost CAGR >=0.70 over2016-08-17 through2026-09-24 with independent verification. No target guarantee; rejected challengers do not count.'
encoded=(json.dumps(spec,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
assert 'PENDING' not in encoded.decode()
path=R/'work/monthly_same_report145_proposal.json'
with path.open('xb') as f:f.write(encoded)
print(json.dumps({'proposal_sha256':sha(path),'census':data['counts'],'new_factor_calculations':0}))
