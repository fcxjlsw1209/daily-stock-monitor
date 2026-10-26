"""Write the independent code-review receipt; no outcome or market execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
import monthly_profitability141_outcome_review as outcome
R=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proposal=R/'work/monthly_profitability141_proposal.json'
assert sha(proposal)=='80901aab35840451f289e58ca90caefefa751c717561e8cb1175d39012fd7a58'
log=R/'work/monthly_profitability141_outcome_prereview_tests.log'
text=log.read_text();assert re.search(r'Ran 87 tests in ',text) and text.rstrip().endswith('OK')
sources={str(p):sha(p)for p in outcome.source_files()}
closer={str(R/'work'/n):sha(R/'work'/n)for n in ['complete_monthly_profitability141.py','test_complete_monthly_profitability141.py']}
record={
 'schema_version':1,'round':141,'stage':'preregistration_outcome_review','passed':True,
 'created_utc':datetime.now(timezone.utc).isoformat(),'proposal_sha256':sha(proposal),
 'synthetic_only':True,'real_141_feature_or_return_calculations':0,'source_hashes':sources,
 'closure_reviewed_source_hashes':closer,
 'synthetic_tests':{'outcome_review':{'executed_by_this_reviewer':True,'passed':87,'failures':0,'errors':0,'log':{'path':str(log.relative_to(R)),'sha256':sha(log)}},'completion':{'executed_by_this_reviewer':False,'root_reported_passed':46,'root_reported_duration_seconds':16.518,'note':'Unchanged closer independently read; no duplicate execution. Root will separately freeze final suite log.'}},
 'reviewed_contracts':[
  'Outcome imports only its hash-pinned independent accounting helper, not producer factor/selector/replay modules; all market execution is behind explicit CLI.',
  'Nine close/open/cost scenarios reconstruct NAV, fixed units, trades, holdings, metrics and slices; training selection checks the profitability challenger against both incumbent and growth control on first88 only.',
  'All366 monthly orders,5978 original cells and all3 added matrices are checked; shared availability, separate reports, period continuity, staleness and exact missing/zero fallback are enforced.',
  'Dataset,meta,sessions,proposal admission hashes and exact source closure are verified before replay, then artifact/source hashes are rechecked after review.',
  'Outcome now explicitly carries frozen proposal interpretation_limits, two-issuer source_scope, first_public_version_verified=false, conditional source-admission label and causal identity assumption.',
  'Closer requires independent prepare/outcome evidence, frozen plans/orders, source/dataset bindings, diagnostic-start chronology, full protected inventory, and registered positive synthetic test counts.',
  'Closer requires exact predecessor140rounds920trials and exact round141 queue; appends only3 trials to reach141rounds923trials, preserves unrelated central metadata and only promotes the selected challenger if it passed the fixed gate.',
  'Completion publication is exclusive and fsynced; state update checks predecessor digest, publishes atomically, and supports exact-completion-SHA idempotent resume with reverified inputs and protected members.'
 ],
 'limited_edits':{'files':['work/monthly_profitability141_outcome_review.py','work/test_monthly_profitability141_outcome_review.py'],'purpose':'Carry conditional source interpretation into final independent outcome and test that it cannot claim first-version or SEC certification.','new_synthetic_tests':4},
 'remaining_blockers':[],
 'scope_limits':['This approves preregistration code contracts and synthetic verification, not real outcome values.','No real round141 feature,rank,order,return or closer execution was performed by this reviewer.','Issuer source admission remains conditional current-retrieval evidence; first-public-version certification remains false.','Central state and observer files were not modified.']}
path=R/'work/monthly_profitability141_outcome_prereview.json'
with path.open('x')as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'path':str(path.relative_to(R)),'sha256':sha(path),'passed':True,'source_hashes':sources,'closure_reviewed_source_hashes':closer}))
