from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
ROOT=Path('/Users/shiweili/code/monthly-stock-research')
DEST=ROOT/'work/expanded_error228_CODE_REVIEW.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(name):return json.loads((ROOT/name).read_text())
assert not DEST.exists(),'Review is write-once'
expected={
'work/monthly_expanded_error228_host.py':'31d171df75bbff371c1c921cf375b8ae6c86d13cb3b9de8320da14bab8c0b3c1',
'work/test_monthly_expanded_error228_host.py':'bbdd2d88cf6342a2c0b682e5c5c55a0f5b74df12f4e9ab4c9718c4f67c25b811',
'work/monthly_expanded_error228_adapter.py':'be9e217a196c8b484b8aa4ede156f5a632c1d02de01aacfd19fa2281d597ebb2',
'work/test_monthly_expanded_error228_adapter.py':'8762d7c49ba828c71a8558fb9c896ece883f74fc23bed9e3ae1470704836f158',
'work/monthly_expanded_error228_independent.py':'0adb29ed8564db4e38785d6df574d65a8d4a5302d8feca002a6fde839308f31c',
'work/test_monthly_expanded_error228_independent.py':'f80d114f5afd68ed3c815ba66f2a5797bd2877fee3bc92f48b0dfbc3ed4015dd'}
CTX='work/monthly_expanded_error228_context.json';CTX_SHA='5831d860f78d87cf47bee1dac8267f180615a2add1c64b01b787ed4bc5dcf747'
TEST='work/monthly_expanded_error228_TESTS.json';TEST_SHA='7872225522e2e8062844a48eeab61d47114135c99145b63433ec1a010f740f6f'
SOURCE='work/expanded_error228_SOURCE_REVIEW.json';SOURCE_SHA='d5981dbb6797894f46e54fc92e5b8702426b1b2c8a993ee655d2b818636b65db'
pins={}
def bind(name,digest):
    p=Path(name);p=(p if p.is_absolute() else ROOT/p).resolve()
    assert p.is_relative_to(ROOT),str(p)
    assert p!=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    assert sha(p)==digest,'Changed reviewed input '+str(p)
    key=str(p.relative_to(ROOT));assert key not in pins or pins[key]==digest
    pins[key]=digest
for name,digest in expected.items():bind(name,digest)
for name,digest in ((CTX,CTX_SHA),(TEST,TEST_SHA),(SOURCE,SOURCE_SHA)):bind(name,digest)
context=read(CTX);tests=read(TEST);source=read(SOURCE)
assert tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==129
assert source['passed'] is True and source['source_only'] is True and source['metadata_only'] is True
assert source['context_sha256']==CTX_SHA and source['evidence_sha256']==context['evidence_sha256']
for key in ('source_npz_decoded','actual_support_computed','actual_returns_computed','observers_modified'):assert source[key] is False
for name,digest in tests['code_sha256'].items():bind(name,digest)
for name,digest in expected.items():assert tests['code_sha256'][str(ROOT/name)]==digest
for name,digest in context['evidence_sha256'].items():bind(name,digest)
for name in ('work/expanded_error228_code_review.py','run_local.sh','local_run.py','local.sandbox.sb','work/monthly_ineligible_source227_model.py',
             'work/monthly_ineligible_source227_independent.py'):
    bind(name,sha(ROOT/name))
review=dict(schema='monthly_expanded_error228_static_code_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),
 passed=True,unresolved_blockers=[],context_sha256=CTX_SHA,code_sha256=expected,input_sha256=dict(sorted(pins.items())),
 reviewer_scope=dict(primary_reviewer='nested199_code_audit',
   adapter_and_adapter_test_author=True,host_and_host_test_author=False,independent_checker_author=False,
   independent_host_static_review=True,
   numerical_adapter_independence='asset213_independent separately rebuilt adapter inputs/outputs without importing producer numerical implementation; both fixtures and all physical prefixes crossed byte-exactly',
   second_host_review='asset213_independent also reported no blockers after independent static inspection of host, adapter and source bindings',
   source_scope='Static code and literal JSON/hash checks only. No actual NPZ decoded, historical error support measured, portfolio replay or registration performed by reviewer.'),
 synthetic_validation=dict(formal_receipt=TEST,formal_receipt_sha256=TEST_SHA,formal_tests=129,formal_exit_code=0,
   formal_session=56826,owned_adapter_tests=21,owned_adapter_session=12786,owned_adapter_exit_code=0,
   independently_authored_adapter_tests=27,host_tests=30,frozen_model_tests=26,frozen_oracle_tests=25,
   exact_six_code_hashes_match_formal_receipt=True,all_formal_code_hashes_reverified=True,
   runtime=tests['runtime']),
 checks=dict(
  source227_plan_completion_prepared_manifest_and_independent_receipts_bound=True,
  exact_old206_and_new227_query_training_axes_checked=True,
  old_forecast_bytes_and_nonadded_label_bytes_preserved=True,
  added_labels_only_original_ineligible=True,
  original_common_good_union_added_only=True,
  historical_use_original_or_newly_available_only=True,
  old_available_rows_outside_original_use_not_newly_admitted=True,
  all_returned_buffers_detached_and_host_snapshots_original_inputs=True,
  original_train_vol63_and_nonlabel_fields_preserved=True,
  expanded_labels_used_only_for_historical_risk_not_source_model_training=True,
  frozen226_canonical_risk_and_independent_gram_oracle_reused=True,
  frozen226_provenance_flags_explicitly_interpreted_for_expanded_supplied_mask=True,
  no_source_models_or_labels_or_raw_features_recomputed_by_adapter_or_host=True,
  original27_arrays_current_eligibility_pre_score_core_and_support_preserved=True,
  fixed_top2_rank3_to5_gap_point02_min18_and_shared_complete_history_preserved=True,
  ready_effect_replacement_uses_one_original_episode_and_existing_budget=True,
  initial_domain_and_history_fallback_exact_incumbent=True,
  constant_supported_risks_zero_effect_not_incumbent_fallback=True,
  physical_source_query_metadata_original_arrays_and_future_label_prefixes=True,
  physical_model_prefixes=[0,1,53,88,122],decision_prefixes=123,
  statistical_parameter_count21_fields_and_independent_receipts_checked=True,
  first88_dual_reference_guards_and_changed_support_frozen_before_later34=True,
  same_close_zero_next_open_zero_and_next_open_cost_point0005_separated=True,
  independent9_paths45_slices_and18_training_metric_groups_required=True,
  exact7_prepared_and38_result_artifact_manifests_required=True,
  immutable_plan_test_code_context_runtime_source_gates_precede_actual_inputs=True,
  full21444_protected_archive_register_close_and_central_cas_required=True,
  central_strategy_entries_before=224,central_trials_before=1189,
  central_strategy_entries_after=225,central_trials_after=1192,
  separate_source227_zero_strategy_trial_record_required=True,
  observer_writes_forbidden_and_only_training_admitted_candidate_can_update_best=True),
 tolerance_policy='No tolerance introduced by228 adapter: dtype/shape/bytes exact. Frozen226 canonical risk/effect/selection checks remain exact; its separate Gram diagnostic tolerance1e-12 and pre-existing independent accounting conventions are unchanged. No actual outcome comparisons performed here.',
 actual_features_computed=False,actual_model_fits_computed=False,actual_portfolio_returns_computed=False,
 actual_support_computed=False,actual_npz_decoded=False,production_files_modified=False,
 strategy_registered_by_reviewer=False,observers_modified=False,
 resolved_findings=[],
 limitations=['Reviewer authored the adapter; this is not a claim that the adapter author independently validated their own implementation. The separate checker author supplies that independence.',
 'The returned train dictionary is a risk-history compatibility view and must never train original Ridge models.',
 'Expanded same-query history does not add missing outside-query identities, certify strictPIT, or demonstrate adequate actual shared-H support before registration.',
 'Source227 forecasts outside original eligibility can be less reliable; relative clipped/month-centered target errors are not realized portfolio return variance.',
 'Actual cache parity, shared history, orders, nine-path accounting and21444 protected bytes remain required post-registration checks; this receipt records code safeguards and synthetic validation.'])
with DEST.open('x') as f:json.dump(review,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=str(DEST),sha256=sha(DEST),input_pins=len(pins),tests=tests['tests'],passed=True)))
