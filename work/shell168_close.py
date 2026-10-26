"""Close data-only sensitivity without adding a strategy trial or changing models."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_sensitivity168'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
EXPECTED_CENTRAL = '687ff7f34cf2a52bf68531cda1e70c9555d5b33bc229404ed254a434e0c2abc6'
PINS = {
    'PLAN.json': 'c50e7a4edd9b83a23efe7a290f6fa82d5f920141b02aff1aed34520191c7d2a6',
    'DATA_COMPARISON.json': '6b6afc5fa84a3abcbeb05e108e2aecb35c6fc30e5c85c274c0b2455ac6980c32',
    'INDEPENDENT_DATA_REVIEW.json': 'de653dbf52dd0d907f2f70c985b0ef81476aebe503a62b74239dcb5843f6c74e',
    'INPUT_IDENTITY.json': 'e77267d13fd09595976a3f99fe61ab47c27bae4f0387db3f6833de0d2669e555',
    'INDEPENDENT_IDENTITY_REVIEW.json': '80cfc7b5c3bf6e10159a28652472a6b31309665f3afb4d23b7dc09a75f7e3800',
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    with p.open('x') as f:
        json.dump(d, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    assert sha(CENTRAL) == EXPECTED_CENTRAL, 'Concurrent central change; inspect before resuming'
    state = json.loads(CENTRAL.read_text())
    docs = {}
    for name, expected in PINS.items():
        assert sha(OUT / name) == expected, name
        docs[name] = json.loads((OUT / name).read_text())
    identity = docs['INPUT_IDENTITY.json']
    assert identity['passed'] and identity['summary']['mapped_member_set_changes'] == 0
    assert identity['summary']['numerical_recomputation_needed'] is False
    for key in ['features_computed', 'model_fits', 'orders_computed', 'returns_computed']:
        assert identity[key] == 0
    bound = dict(identity['input_sha256'])
    bound.update(identity['proof_code_sha256'])
    for name, expected in bound.items():
        p = ROOT / name
        assert p.resolve().is_relative_to(ROOT), name
        assert sha(p) == expected, name
    assert docs['INDEPENDENT_DATA_REVIEW.json']['status'] == 'PASS_SINGLE_DECLARED_PROXY_SENSITIVITY_ONLY'
    assert docs['INDEPENDENT_IDENTITY_REVIEW.json']['verdict'] == 'PASS_SINGLE_DECLARED_PROXY_INPUT_IDENTITY_ONLY'
    assert docs['INDEPENDENT_IDENTITY_REVIEW.json']['may_archive_this_sensitivity'] is True
    assert len(state['completed']) == 167 and sum(r['trials'] for r in state['completed']) == 1016
    assert not state['registered_queue']
    novelty = ROOT / 'work/after_shell168_novelty_review.json'
    assert sha(novelty) == 'a5abf30ce41dd3a305e1afb49b960dd6e8395786f4f261025471cf855b934e7d'
    stamp = datetime.now(timezone.utc).isoformat()
    completion = dict(schema='shell168_data_sensitivity_completion_v1', completed_utc=stamp,
        status='INDEPENDENTLY_VERIFIED_CONDITIONAL_INPUT_EQUIVALENCE',
        scope='Only Shell cap proxy changes; original price/feature sources and all other issuers held fixed.',
        artifacts=PINS, verified_bound_input_count=len(bound),
        data_summary=docs['DATA_COMPARISON.json']['summary'], identity_summary=identity['summary'],
        independent_data_review=True, independent_identity_review=True,
        inference='Reordered cap lists normalize to identical event and training inputs; existing deterministic model/orders are unchanged in this scenario.',
        new_strategy_trials=0, new_returns_computed=False, new_model_fits=0,
        closure_retry='Initial closer used status instead of verdict key; stopped before writes, corrected field lookup only.',
        existing_best_not_recomputed=state['best_training_selected_candidate'], target_reached=False,
        limitations=docs['INDEPENDENT_DATA_REVIEW.json']['limitations'] + [
          'This does not establish that replacing model price series with A/B quote data preserves returns.',
          'No full-market Top32, exact net capitalization or original historical vintage certification.'],
        observers_modified=False, source_assumptions=docs['PLAN.json']['assumptions'],
        closer_sha256=sha(Path(__file__)), prior_central_sha256=EXPECTED_CENTRAL)
    write(OUT / 'COMPLETION.json', completion)
    ref = dict(path=str((OUT / 'COMPLETION.json').relative_to(ROOT)),
               sha256=sha(OUT / 'COMPLETION.json'), strategy_trials=0,
               status=completion['status'], full_point_in_time_dataset_complete=False)
    state['data_evidence_completions'].append(ref)
    state['latest_pool_data_progress'] = ref
    state['negative_novelty_reviews'].append(dict(path=str(novelty.relative_to(ROOT)), sha256=sha(novelty)))
    state['next'] = 'Shell conditional cap audit closed: 53 periods, zero normalized membership/input changes. 167 strategy rounds / 1016 trials retained; best CAGR67.666%, target unmet.'
    state['next_step'] = 'Audit independently collected PetroChina A/H price, FX and historical share-class evidence before any separately registered pool reconstruction; no performance-selected multiplier.'
    state['last_goal_turn_classification'] = 'progress'
    state['updated_utc'] = stamp
    backup = OUT / 'CENTRAL_BEFORE.json'
    with backup.open('xb') as f:
        f.write(CENTRAL.read_bytes())
    tmp = CENTRAL.with_name('CONTINUATION_STATUS.shell168.tmp')
    write(tmp, state)
    assert sha(CENTRAL) == EXPECTED_CENTRAL, 'Concurrent central change before atomic replace'
    os.replace(tmp, CENTRAL)
    after = json.loads(CENTRAL.read_text())
    before = json.loads(backup.read_text())
    for key in ['completed', 'best_training_selected_candidate', 'registered_queue', 'current', 'current_status']:
        assert after[key] == before[key], key
    write(OUT / 'CENTRAL_UPDATE.json', dict(updated_utc=stamp, before=EXPECTED_CENTRAL,
        after=sha(CENTRAL), completion_sha256=ref['sha256'], preserved_strategy_registry=True))
    print(json.dumps(dict(status=completion['status'], completion_sha256=ref['sha256'],
                         strategy_rounds=167, strategy_trials=1016, target_reached=False)))


if __name__ == '__main__':
    main()
