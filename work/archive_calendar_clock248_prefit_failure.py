"""Archive a registered pre-candidate failure without claiming completed results."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import monthly_calendar_clock248_host as h
root=h.ROOT
expected='28e567db41e562c4a31e33104e3c2c45392b9bd9197b95ce16eaff866219708e'
review=root/'work/calendar_clock248_FAILURE_REVIEW.json'
assert h.sha(review)=='f90f478008e2361ac24c45b92142776649eeb054e172399e424418d36417ecc0'
review_record=h.read(review)
assert review_record['status']=='CONFIRMED_METADATA_ORIGIN_BINDING_ERROR_FAIL_CLOSED'
for evidence in [h.OUT/'PREPARE_FAILURE.json',root/'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json']:
 pins=review_record['input_sha256'];expected_evidence=pins.get(str(evidence),pins.get(str(evidence.relative_to(root))))
 assert expected_evidence is not None and h.sha(evidence)==expected_evidence
h.gate(expected)
failure=h.read(h.OUT/'PREPARE_FAILURE.json')
assert failure['status']=='FAILED' and failure['plan_sha256']==expected and failure['error']=='Full original rolling history origin'
assert not any((h.OUT/name).exists() for name in ['prepared','selected','results','COMPLETION.json','ABORTED.json'])
protected=h.read(h.old.PROTECTED);h.old.verify(protected);assert len(protected)==21444
before=h.CENTRAL.read_bytes();state=json.loads(before);h.registration_check(state,expected)
assert len(state['completed'])==243 and sum(x['trials'] for x in state['completed'])==1243
prior=h.read(h.OUT/'STATE_BEFORE.json')
assert state['completed']==prior['completed']
assert state['best_training_selected_candidate']==prior['best_training_selected_candidate']
h.predecessor_binding(state)
receipt=dict(status='ABORTED_BEFORE_CANDIDATE_FEATURES',created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=expected,
 failure_sha256=h.sha(h.OUT/'PREPARE_FAILURE.json'),review_sha256=h.sha(review),
 axis_diagnostic_sha256=h.sha(root/'work/calendar_clock248_ORIGIN_DIAGNOSTIC.json'),
 configured_arms=2,actual_candidate_features=0,actual_candidate_orders=0,actual_candidate_returns=0,
 original_sources_loaded=True,original_cached_pre_score_computed=True,original_episode_calls=0,
 protected_files=21444,observers_modified=False,completed_rounds_unchanged=243,completed_trial_count_unchanged=1243,
 replacement='monthly_calendar_clock248_v2',reason='Manifest reporting start2010 was misbound as loader axis origin; unchanged original64-file loader axis begins2008-01-02. Separate version required; no altered formula or tolerance.',
 preservation='Original PLAN,source,code,tests andfailure remain byte-for-byte. No completed/numerical-performance claim for this attempt.',
 prior_central_sha256=hashlib.sha256(before).hexdigest(),archive_script_sha256=h.sha(Path(__file__)))
h.dump(h.OUT/'ABORTED.json',receipt)
state.setdefault('prefit_failed_registrations',[]).append(dict(directory=h.OUT.name,stage='prepare',candidate_returns_observed=False,candidate_features_computed=False,configured_arms=2,new_factor_or_return_trials=0,original_sources_loaded=True,original_cached_pre_score_computed=True,reason=receipt['reason'],replacement=receipt['replacement'],archive=str((h.OUT/'ABORTED.json').relative_to(root)),archive_sha256=h.sha(h.OUT/'ABORTED.json')))
state.update(current=h.PRIOR.name,current_status='completed_independently_verified_with_prefit_failure_archived',registered_queue=[],updated_utc=datetime.now(timezone.utc).isoformat(),last_goal_turn_classification='progress',next_step='Register separately reviewed calendar_clock248_v2 origin repair; preserve failed248 evidence and all scoring thresholds.')
h.old.commit_state(before,state,'calendar_clock248_prefit_failure_archive')
print(json.dumps(dict(status=receipt['status'],archive_sha256=h.sha(h.OUT/'ABORTED.json'),completed=243,trials=1243)),flush=True)
