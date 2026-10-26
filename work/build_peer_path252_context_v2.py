"""Prospective source-policy-only revision. No new factor/orders/returns."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import json,hashlib,platform
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def write(p,value):
    with (ROOT/p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
assert platform.python_version()=='3.9.6'
policy_path='work/monthly_peer_path252_v2_LISTING_POLICY.json'
assert sha(policy_path)=='919a698dafe5fafc7b0516ac7fb613d1ff5292e0a831f049dc134b8864ca1e91'
policy=read(policy_path)
context=read('work/monthly_peer_path252_CONTEXT.json');source=read('work/monthly_peer_path252_SOURCE_CONTRACT.json')
gaps=read('work/monthly_peer_path252_SOURCE_GAPS.json');failure=read('work/monthly_peer_path252_PREPARE_FAILURE.json')
assert gaps['unique_gap_count']==4 and gaps['returns_computed'] is False and failure['new_features_computed'] is False
stamp=datetime.now(timezone.utc).isoformat()
extra=[policy_path,'work/monthly_peer_path252_CONTEXT.json','work/monthly_peer_path252_SOURCE_CONTRACT.json',
       'outputs/monthly_peer_path252/PLAN.json','work/monthly_peer_path252_PREPARE_FAILURE.json',
       'work/monthly_peer_path252_SOURCE_GAPS.json','work/monthly_peer_path252_source_gap_audit.py',
       'work/build_peer_path252_context_v2.py']
for doc in (context,source):
    for path in extra:doc['input_sha256'][path]=sha(path)
revision=dict(version=2,prior_plan_sha256=sha('outputs/monthly_peer_path252/PLAN.json'),prior_source_failure_sha256=sha('work/monthly_peer_path252_PREPARE_FAILURE.json'),
    gap_audit_sha256=sha('work/monthly_peer_path252_SOURCE_GAPS.json'),listing_policy=dict(path=policy_path,sha256=sha(policy_path)),
    source_only=True,prior_new_features=0,prior_new_orders=0,prior_new_returns=0,prior_full_slot_observations=816992,
    unchanged_strategy_fields=['arms','hypothesis','causal_scope','original_control','exact_math','scoring','budget','selection','execution','validation'],
    strategy_trial_count=3,prior_source_attempts=1,trial_count_reason='Same predeclared strategy and all its numeric parameters; prior attempt failed before factor/orders/returns. Source failure retained separately, no erased attempts.',
    rule='Keep exact old history_bound and all maturity/cohort fields. Add path_history_bound copied from old bound, except same-issuer SPCX at query>=2026-08-05 uses official actual first trade2026-06-12. Prior queries retain original lower bound. Any old-mature stock still requires every211 valid prices; new bound must not silently change maturity or support.')
source.update(schema='monthly_peer_path252_metadata_source_contract_v2',created_utc=stamp,status='SOURCE_POLICY_REFINED_DAILY_ADMISSION_PENDING',source_revision=deepcopy(revision),raw_source_bytes_rehashed=False,
    prior_source_only_inventory_reviewed=True,actual_mature211_coverage='V1_DIAGNOSTIC_FOUND_NO_MATURE_GAPS_V2_INDEPENDENT_ADMISSION_PENDING')
source['source_policy']['warmup']=revision['rule']+' For structurally warmup stocks every pre-path-bound slot must be explicitly missing, every at/after-path-bound slot must be valid finite positive real nonbool. Invalid present values always fail. No filled price, endpoint substitution, hidden ticker omission or first-valid-quote date inference.'
source['source_policy']['old_certificate']+=' v1 daily source attempt failed on a lower-bound/realized-first-trade conflation; it is preserved, not reclassified as successful.'
source['listing_policy']=revision['listing_policy']
source_path='work/monthly_peer_path252_SOURCE_CONTRACT_v2.json';write(source_path,source)
context.update(schema='monthly_peer_path252_context_v2',created_utc=stamp,status='PROSPECTIVE_SOURCE_REVISION_NOT_REGISTERED',source_revision=deepcopy(revision),actual_prices_loaded=True,
    source_coverage_before_registration='V1_SOURCE_ONLY_INVENTORY_REVIEWED_V2_ADMISSION_PENDING')
context['source']=deepcopy(context['source']);context['source'].update(path=source_path,sha256=sha(source_path),schema=source['schema'],source_policy=source['source_policy'])
context['input_sha256'][source_path]=sha(source_path)
context['phase_order'][0]='Immutable v2 contract/code/tests/review and PLAN before anyactual new252 feature/orders/returns; prior v1 source-only failed attempt and full gap inventory remain recorded.'
context['limitations'].append('Official8/4 first-trade confirmation is visible only from8/5; cannot rewrite6/30 or7/31 historical source knowledge. Actual-first-trade is distinct from June8 lower-bound announcement.')
write('work/monthly_peer_path252_CONTEXT_v2.json',context)
print(json.dumps({'context_sha256':sha('work/monthly_peer_path252_CONTEXT_v2.json'),'source_contract_sha256':sha(source_path),'new_features':0,'new_orders':0,'new_returns':0}))
