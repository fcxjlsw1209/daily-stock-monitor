"""Create separate origin-corrected contracts; no market numeric evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def write(p,d):
 with (ROOT/p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(path=p,sha256=sha(ROOT/p))),flush=True)
fixed={'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json':'d250fc4d42b6089c62f4b1c4a032b872db7a1dd3cc99f9c133b8c8421c8c0eaf','work/calendar_clock248_FAILURE_REVIEW.json':'f90f478008e2361ac24c45b92142776649eeb054e172399e424418d36417ecc0','work/monthly_calendar_clock248_context.json':'23a68af34718f1c4bf854b6e2b36b468f6130907cf7d97757a2a691f3c6c465f','work/calendar_clock248_SOURCE_REVIEW.json':'6fa0eb06b79d1a250d16629b8351cb7a0fd647f52245148d31eab80429c0a19f','outputs/monthly_calendar_clock248/PLAN.json':'28e567db41e562c4a31e33104e3c2c45392b9bd9197b95ce16eaff866219708e'}
for p,h in fixed.items():assert sha(ROOT/p)==h,p
axis=read('work/calendar_clock248_ORIGIN_DIAGNOSTIC.json');assert axis['actual_origin']=='2008-01-02' and axis['manifest_start']=='2010-01-04' and axis['raw_sources_exact'] is True
assert all(axis[k] is False for k in ['original_features_recomputed','candidate_features_computed','orders_computed','returns_computed','protected_files_modified'])
source=read('work/calendar_clock248_SOURCE_REVIEW.json');context=read('work/monthly_calendar_clock248_context.json')
assert source['panel_first_session']==context['panel_first_session']=='2010-01-04'
for p,h in source['input_sha256'].items():assert sha(Path(p))==h,p
more=list(fixed)+['work/calendar_clock248_origin_diagnostic.py','work/calendar_clock248_v2_COPY_RECEIPT.json','outputs/monthly_calendar_clock248/PREPARE_FAILURE.json','work/prepare_calendar_clock248_v2_contract.py']
origin=dict(status='CORRECTED_BEFORE_ANY_CANDIDATE_FEATURES',failed_attempt='monthly_calendar_clock248',diagnostic_path='work/calendar_clock248_ORIGIN_DIAGNOSTIC.json',diagnostic_sha256=fixed['work/calendar_clock248_ORIGIN_DIAGNOSTIC.json'],failure_review_path='work/calendar_clock248_FAILURE_REVIEW.json',failure_review_sha256=fixed['work/calendar_clock248_FAILURE_REVIEW.json'],manifest_reported_start='2010-01-04',actual_loader_first_session='2008-01-02',manifest_reporting_start_is_not_axis_proof=True,original64_sources_unchanged=True,original_source_was_loaded_for_axis_diagnostic=True,original_cached_pre_was_computed_in_failed_attempt=True,candidate_features_orders_returns_previously_computed=False,formula_changes=False,tolerance_changes=False,original_U0_core_control_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE')
source.update(schema='calendar_clock248_v2_metadata_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),panel_first_session='2008-01-02',origin_correction=origin)
for p in more:source['input_sha256'][str(ROOT/p)]=sha(ROOT/p)
write('work/calendar_clock248_v2_SOURCE_REVIEW.json',source)
context.update(schema='monthly_calendar_clock248_v2_context_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='PROSPECTIVE_ORIGIN_REPAIR_NOT_REGISTERED',panel_first_session='2008-01-02',origin_correction=origin)
for p in more+['work/calendar_clock248_v2_SOURCE_REVIEW.json']:context['input_sha256'][p]=sha(ROOT/p)
context['limitations'].append('Original248 failed before source-core/calendar candidate numerical computation due stale manifest reporting start. This version corrects only full-history origin using the same original64 source files; original frozen248 remains unchanged. Runtime source/core/control gates remain mandatory and unrelaxed.')
write('work/monthly_calendar_clock248_v2_context.json',context)
