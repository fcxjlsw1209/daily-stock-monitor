"""Close the source audit only; do not count it as a strategy experiment."""
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

from liquidity162_audit_v2 import ROOT, OUT, SOURCE_OUT, PLAN_SHA, bound, digest, now, write_new


def main():
    plan = bound()
    extracted = json.loads((OUT/'EXTRACTED.json').read_text())
    review = json.loads((OUT/'INDEPENDENT_REVIEW.json').read_text())
    protection = json.loads((OUT/'PROTECTED_AFTER.json').read_text())
    if not review['passed'] or len(review['records']) != 24 or not all(x['verified'] for x in review['records']):
        raise ValueError('Complete independent review required')
    if review['extraction_sha256'] != digest(OUT/'EXTRACTED.json'):
        raise ValueError('Unreviewed extraction')
    if not protection['passed'] or protection['checked_files'] != 21444:
        raise ValueError('Complete protected-archive check required')
    if [(r['ticker'],r['fiscal_year']) for r in extracted['records']] != [
            (r['ticker'],r['fiscal_year']) for r in plan['source_records']]:
        raise ValueError('Source set changed')
    for result, audit in zip(extracted['records'], review['records']):
        if result['status'] != audit['status']:
            raise ValueError('Source status not independently supported')
    state_path = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    before_raw = state_path.read_bytes()
    state = json.loads(before_raw)
    if (state['current'] != 'monthly_rank161' or state['current_status'] != 'completed_independently_verified'
            or len(state['completed']) != 161 or sum(x['trials'] for x in state['completed']) != 998
            or state['registered_queue']):
        raise ValueError('Central research state advanced; do not overwrite')
    before_hash = digest(state_path)
    counts = dict(Counter(r['status'] for r in extracted['records']))
    record_summary = [{
        'ticker': r['ticker'], 'fiscal_year': r['fiscal_year'], 'status': r['status'],
        'reasons': r['reasons'], 'known_by_at': r['availability']['known_by_at'],
        'available_session': r['availability']['available_session'],
        'currency_status': (r.get('statement') or {}).get('currency_status'),
        'source_url': r['source_binding']['source_url'],
        'independent_scope_caveats': a.get('conditional_coverage', []),
    } for r, a in zip(extracted['records'], review['records'])]
    completion = {
        'status': 'source_only_audit_complete', 'completed_utc': now(),
        'source_plan_sha256': PLAN_SHA, 'source_records': 24, 'status_counts': counts,
        'record_summary': record_summary, 'strategy_trials_added': 0,
        'numerical_factors_computed': 0, 'orders_computed': 0, 'returns_computed': 0,
        'completed_strategy_rounds': 161, 'completed_strategy_trials': 998,
        'best_candidate_unchanged': state['best_training_selected_candidate'],
        'previous_goal_turn_classification': 'progress: source scope was frozen before financial extraction',
        'this_goal_turn_classification': 'progress: original statements extracted and independently reviewed',
        'protected_files_unchanged': 21444, 'live_observer_root_accessed': False,
        'failed_v1_preserved': {'path':str(SOURCE_OUT/'IDENTITY_FAILURE.json'),
            'sha256':digest(SOURCE_OUT/'IDENTITY_FAILURE.json')},
        'evidence_hashes': {name:digest(OUT/name) for name in (
            'SYNTHETIC_TESTS.json', 'EXTRACTION_CODE.json',
            'EXTRACTED.json', 'INDEPENDENT_REVIEW.json', 'PROTECTED_AFTER.json')},
        'limitations': plan['limitations'] + [
            'A candidate is only a narrow source-field proxy, not an admitted strategy factor.',
            'Dollar-symbol-only rows do not establish an ISO currency; no USD conversion was performed.',
            'Listed borrowing carrying values do not certify all upcoming contractual repayments.',
            'Reported investments do not certify unrestricted cash availability.',
            'Rejected years remain unavailable; no zero, interpolation or later-disclosure fill.',
        ],
        'next_action': 'Register a past-only availability/coverage feasibility screen before deciding whether a numerical factor experiment is justified; preserve the 70% target and frozen observers.',
        'central_state_before_sha256': before_hash,
        'finalizer_sha256': digest(__file__),
    }
    write_new('COMPLETION.json', completion)
    backup = state_path.parent/('PRESOURCE162_'+before_hash+'.json')
    with backup.open('xb') as handle:
        handle.write(before_raw)
    state.update(updated_utc=now(), last_goal_turn_classification='progress',
        latest_source_progress={
            'path': str(OUT/'COMPLETION.json'), 'sha256': digest(OUT/'COMPLETION.json'),
            'status': 'source_only_audit_complete', 'records': 24, 'status_counts': counts,
            'strategy_trials_added': 0},
        next='The 24-source liquidity/debt audit is complete. The 70% CAGR target is not reached; the best remains unchanged. No new strategy trial has run.',
        next_step=completion['next_action'])
    if digest(state_path) != before_hash:
        raise ValueError('Concurrent central state change; source completion remains saved')
    with tempfile.NamedTemporaryFile(dir=state_path.parent, delete=False) as handle:
        temp = Path(handle.name)
        handle.write((json.dumps(state, indent=2, sort_keys=True)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if digest(state_path) != before_hash:
            raise ValueError('Concurrent central state change')
        os.replace(temp, state_path)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'completed':completion['status'], 'status_counts':counts,
                      'candidate_years':[(r['ticker'],r['fiscal_year']) for r in record_summary if r['status']=='candidate'],
                      'completion_sha256':digest(OUT/'COMPLETION.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
