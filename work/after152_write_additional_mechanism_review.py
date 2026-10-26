"""Write a bounded metadata/code novelty review. No market calculations."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'work/after152_additional_mechanism_review.json'
PATHS = [
    'work/after152_write_additional_mechanism_review.py',
    'outputs/monthly_target70_research/CONTINUATION_STATUS.json',
    'work/monthly_next_mechanism_source_20261015.json',
    'work/after140_path_novelty.json',
    'work/after150_guidance_novelty.json',
    'outputs/monthly_trend_evidence_research/trend_features.py',
    'outputs/monthly_trend_evidence_research/study.py',
    'outputs/monthly_trend_evidence_research/PLAN.json',
    'outputs/monthly_trend_evidence_research/GRID.json',
    'outputs/monthly_trend_evidence_research/COMPLETION.json',
    'outputs/monthly_quality_research/explore.py',
    'outputs/monthly_path70_research/features.py',
    'outputs/monthly_earnings_factor_research/earnings_features.py',
    'outputs/monthly_earnings_factor_research/study.py',
    'outputs/monthly_earnings_factor_research/PLAN.json',
    'outputs/monthly_earnings_factor_research/VALIDATION.json',
    'outputs/monthly_incumbent_earnings70/PLAN.json',
    'work/monthly_peer_earnings139_factor.py',
    'work/monthly_peer_earnings139_proposal.json',
    'work/monthly_price_delay140_factor.py',
    'work/monthly_price_delay140_proposal.json',
    'work/monthly_price_delay140_novelty.json',
    'outputs/monthly_earnings_repair/EVENTS.json',
    'outputs/monthly_earnings_repair/STATUS.json',
    'outputs/monthly_earnings_repair/SCOPED_READINESS.json',
    'outputs/monthly_earnings_repair/event_schema.py',
    'outputs/monthly_earnings_repair/audit_scope.py',
]
hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PATHS}
completion = json.loads((ROOT / 'outputs/monthly_trend_evidence_research/COMPLETION.json').read_text())
status = json.loads((ROOT / 'outputs/monthly_earnings_repair/STATUS.json').read_text())
readiness = json.loads((ROOT / 'outputs/monthly_earnings_repair/SCOPED_READINESS.json').read_text())
record = {
    'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
    'stage': 'bounded_additional_mechanism_code_review',
    'scope': 'Existing local code, completed-experiment metadata and event-clock admission only; not a full historical archive review.',
    'decisions': {
        'underwater_recovery': {
            'status': 'directly_tested_duplicate_do_not_retry_as_new_mechanism',
            'source': 'outputs/monthly_trend_evidence_research/trend_features.py',
            'function': '_path_recovery',
            'formula': 'On adjusted prices P[t-252:t-42] inclusive (211 observations), keep nondecreasing running high H. Age resets to zero when P >= H*(1-1e-12), otherwise increments by one. recovery_speed=-sum(age)/(211*210).',
            'existing_placement': 'study.score centers finite cross-sectional ranks, supplies zero for missing, and adds budget*(clip(original_support+rank,-.5,.5)-original_support) to the unchanged base. GRID has original65/simple bases and .025/.075 recovery budgets.',
            'completed': completion['completed'], 'existing_study_trials': completion['trials'],
            'existing_study_tests': completion['tests_passed'],
            'distinctions': [
                'Path efficiency is net log movement divided by total absolute daily movement, not waiting duration.',
                'Three-block path stability measures dispersion or weakest block growth, not time under a running high.',
                'Maximum drawdown measures depth; duration measures elapsed time. The direct duration source makes that distinction insufficient novelty.',
                'Peak-recency historical index describes time of the whole-window maximum. Its source package is missing locally; no unverified exact formula is asserted.',
                'Recent rebound episodes are a different local-pattern mechanism, but do not negate the direct long-window recovery duplicate.',
            ],
            'control_limit': 'This review did not establish that the old study residualized recovery against total return and volatility on a strictly matched basis. Adding that control alone does not establish a new mechanism.',
        },
        'formation_period_post_earnings_drift': {
            'status': 'not_ready_for_registration',
            'candidate_under_review': 'Separate all post-announcement day2..20 residual-return observations within a formation window ending t-42 from its remaining observations; no EPS expectations or announcement-gap inference.',
            'nearest_tested': [
                {'source': 'outputs/monthly_earnings_factor_research/study.py', 'formula': 'Latest verified event only; anchor is first session close at/after conservative known_by. event10/event21 equal stock simple return minus same-endpoint SPY simple return. Positive .5*tanh(raw/.05) at .025/.05 budgets; matched_recent10/21 controls have the identical event mask. Paired Ridge adds event10,event21,event21-event10 to plain10.'},
                {'source': 'outputs/monthly_incumbent_earnings70/PLAN.json', 'formula': 'Uses latest-event21 versus matched_recent21 in the incumbent episode-boundary slot, without new source reconstruction.'},
                {'source': 'work/monthly_peer_earnings139_factor.py', 'formula': 'Same frozen-sector peers from original stock pool, distinct issuers, event-anchor age10..63 sessions, at least2 peers: arithmetic mean peer event10 versus peer recent10 and negative own event10/peer_count, common .5*tanh(raw/.05).'},
                {'source': 'work/monthly_price_delay140_factor.py', 'formula': '52 weekly stock returns through the skip42 cutoff; restricted contemporaneous market regression versus market plus4lags. D=1-R2_restricted/R2_full, compared with1-R2_restricted. No earnings dates.'},
            ],
            'verified_difference': 'The inspected implementations use latest-event/current-age reactions or unconditional market delay. They do not partition all observations of the old formation window into post-event versus other-day momentum.',
            'novelty_limit': 'PEAD is already tested. This is at most a different temporal aggregation within that family; no claim that the exact variant is absent from every historical experiment.',
            'event_clock': {
                'records': status['collected_event_records'], 'precision_counts': status['precision_counts'],
                'source': 'outputs/monthly_earnings_repair/EVENTS.json',
                'available_fields': ['event_id','symbol','fiscal_year','fiscal_quarter','publication_date','precision','known_by_at','available_session','source_url','evidence_path','evidence_sha256'],
                'admitted_scope': readiness['scope'],
                'scoped_post_known_drift_data_ready': readiness['scoped_post_known_drift_data_ready'],
                'full_global_earnings_archive_complete': readiness['full_global_earnings_archive_complete'],
                'exact_announcement_gap_or_eps_surprise_ready': readiness['exact_announcement_gap_or_eps_surprise_ready'],
                'meaning': 'Date-only records map conservatively to a later XNYS close. They support a clearly declared post-known window, not exact first-public announcement day2. Current archives are not immutable contemporaneous snapshots.',
            },
            'hard_remaining_evidence': [
                'Certify complete regular-release sequence for every original candidate throughout the whole consumed formation window plus any event-window warmup; latest-event candidate-date readiness does not prove this.',
                'Fix a precise post-known anchor and session interval; do not silently equate it to actual announcement time.',
                'Define residual benchmark/estimation timing, overlapping-event assignment, cutoff clipping and missing-event behavior before any feature calculation. Missing history must not turn unobserved event days into ordinary days.',
            ],
        },
    },
    'historical_archive_limit': {'reported_missing_packages': 32, 'reported_package_scope': 116,
        'scope_note': 'Inherited bounded-archive limitation; not rescanned here. Missing packages cannot be taken as evidence of untested novelty.'},
    'new_trials': 0, 'new_factor_calculations': 0, 'model_fits': 0,
    'new_orders': 0, 'new_returns_calculated': 0,
    'registered_strategy': False, 'observer_files_accessed': False,
    'input_hashes': hashes,
}
with TARGET.open('x') as stream:
    json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'path': str(TARGET), 'sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                  'input_files': len(hashes), 'new_trials': 0}))
