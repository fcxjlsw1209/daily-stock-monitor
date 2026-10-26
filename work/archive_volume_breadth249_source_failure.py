"""Archive a registered pre-candidate failure without claiming completed results."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import monthly_volume_breadth249_host as h
root=h.ROOT
expected='f3f4a6d9f00c2aacf6877fd39f0763b446873afb0588806972013735a4d8a9f8'
review=root/'work/volume_breadth249_FAILURE_REVIEW.json'
assert h.sha(review)=='70ba41f08901f2950f02f4f8e0f7f09a80b096bffcd0d3f27715b09af36af6cd'
review_record=h.read(review)
assert review_record['status']=='FAILURE_CONFIRMED_NOT_PREPARED'
for evidence in [h.OUT/'PREPARE_FAILURE.json',root/'work/volume_breadth249_SOURCE_DIAGNOSTIC.json']:
 pins=review_record['input_sha256'];expected_evidence=pins.get(str(evidence),pins.get(str(evidence.relative_to(root))))
 assert expected_evidence is not None and h.sha(evidence)==expected_evidence
h.gate(expected)
failure=h.read(h.OUT/'PREPARE_FAILURE.json')
assert failure['status']=='FAILED' and failure['plan_sha256']==expected and failure['error']=='; '.join(['Positive required volume']*4)
assert review_record['passed'] is True and review_record['verdict']['no_blocker_to_recording_this_failure'] is True
for key in ('new_measure_calls','new_block_covariances_computed','new_target_ranks_computed','candidate_funded_pre_scores_computed','candidate_final_scores_computed','new_orders_computed','new_strategy_portfolio_returns_computed'):
 assert review_record['actual_prepare_scope'][key]==0
assert not any((h.OUT/name).exists() for name in ['prepared','selected','results','COMPLETION.json','ABORTED.json'])
protected=h.read(h.old.PROTECTED);h.old.verify(protected);assert len(protected)==21444
before=h.CENTRAL.read_bytes();state=json.loads(before);h.registration_check(state,expected)
assert len(state['completed'])==244 and sum(x['trials'] for x in state['completed'])==1245
prior=h.read(h.OUT/'STATE_BEFORE.json')
assert state['completed']==prior['completed']
assert state['best_training_selected_candidate']==prior['best_training_selected_candidate']
h.predecessor_binding(state)
receipt=dict(status='ABORTED_DURING_SOURCE_PREPARATION',created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=expected,
 failure_sha256=h.sha(h.OUT/'PREPARE_FAILURE.json'),review_sha256=h.sha(review),
 source_diagnostic_sha256=h.sha(root/'work/volume_breadth249_SOURCE_DIAGNOSTIC.json'),
 configured_arms=3,actual_candidate_features=0,actual_candidate_orders=0,actual_candidate_returns=0,
 original_sources_loaded=True,original_cached_pre_score_computed=True,original_episode_calls=0,
 protected_files=21444,observers_modified=False,completed_rounds_unchanged=244,completed_trial_count_unchanged=1245,
 replacement='monthly_volume_breadth249_v2',reason='Frozen positive-volume admission failed on native stored zero at first ordinary event. Exact economic semantics remain unknown. Separate v2 uniform unresolved-zero whole-event fallback preserves raw data and formulas/weights/guards.',
 preservation='Original PLAN,source,code,tests andfailure remain byte-for-byte. No completed/numerical-performance claim for this attempt.',
 prior_central_sha256=hashlib.sha256(before).hexdigest(),archive_script_sha256=h.sha(Path(__file__)))
h.dump(h.OUT/'ABORTED.json',receipt)
state.setdefault('prefit_failed_registrations',[]).append(dict(directory=h.OUT.name,stage='prepare',candidate_returns_observed=False,candidate_features_computed=False,configured_arms=3,new_factor_or_return_trials=0,original_sources_loaded=True,original_cached_pre_score_computed=True,reason=receipt['reason'],replacement=receipt['replacement'],archive=str((h.OUT/'ABORTED.json').relative_to(root)),archive_sha256=h.sha(h.OUT/'ABORTED.json')))
state.update(current=h.PRIOR.name,current_status='completed_independently_verified_with_prefit_failure_archived',registered_queue=[],updated_utc=datetime.now(timezone.utc).isoformat(),last_goal_turn_classification='progress',next_step='Register separately reviewed volume_breadth249_v2 unknown-zero availability policy; preserve failed249 and all scoring thresholds.')
h.old.commit_state(before,state,'volume_breadth249_source_failure_archive')
print(json.dumps(dict(status=receipt['status'],archive_sha256=h.sha(h.OUT/'ABORTED.json'),completed=244,trials=1245)),flush=True)
