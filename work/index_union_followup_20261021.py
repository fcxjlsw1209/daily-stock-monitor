"""Additive, offline calendar/evidence review; no eligibility or return calculation."""
import hashlib
import importlib
import io
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/index_union'
PRIOR = BASE / 'FOLLOWUP_2026-10-20_SOLS.json'
PRIOR_SHA256 = '91a124890f547f02067ebab6e3f5132be7106b9976ca276b06be9217dbcf2ab5'
TARGET = BASE / 'FOLLOWUP_2026-10-21.json'
REVIEW_DATE = '2026-10-21'
COMMAND = './run_local.sh work/index_union_followup_20261021.py'
SEARCH_BATCHES = [
    {
        'tool': 'web.search_query', 'response_length': 'long',
        'queries': [
            {'q': 'Electronic Arts Nasdaq-100 removal August 2026 index',
             'domains': ['ir.nasdaq.com', 'nasdaqtrader.com', 'globalnewswire.com', 'globenewswire.com']},
            {'q': 'Kraft Heinz Nasdaq-100 September 14 2026 removed',
             'domains': ['ir.nasdaq.com', 'nasdaqtrader.com', 'globenewswire.com']},
            {'q': 'Kraft Heinz Kraft Foods July 2015 Nasdaq-100 Index change',
             'domains': ['ir.nasdaq.com', 'nasdaqtrader.com', 'globenewswire.com']},
        ],
        'reported_outcome': 'Already-known EA ECA2026-545 and KRFT ETA2015-102; irrelevant index terminations and known CTRX/JD 2015 and June 2026 quarterly releases. No index-specific proof for a pending case.',
        'known_relevant_results': [
            {'url': 'https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-545',
             'limited_scope': 'July 31 heading with updated closure language; last trade August 4, suspension August 5, $210. Exchange/transaction facts, not index removal certification.'},
            {'url': 'https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2015-102',
             'limited_scope': 'June 29 heading with July 1 approval language; expected merger July 2 and listing July 6, 2015. Exchange/transaction facts, not index continuity certification.'},
        ],
    },
    {
        'tool': 'web.search_query',
        'queries': [
            {'q': '"Electronic Arts" "August" "2026" "Nasdaq-100" "removed"',
             'domains': ['indexes.nasdaq.com', 'ir.nasdaq.com', 'nasdaqtrader.com', 'globenewswire.com']},
            {'q': '"Kraft Heinz" "September" "2026" "Nasdaq-100"',
             'domains': ['indexes.nasdaq.com', 'ir.nasdaq.com', 'nasdaqtrader.com', 'globenewswire.com']},
            {'q': '"Kraft Heinz" "July 6, 2015" "Index"',
             'domains': ['indexes.nasdaq.com', 'ir.nasdaq.com', 'nasdaqtrader.com']},
        ],
        'reported_outcome': 'Entire tool response: Empty search results / No results found.',
    },
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def checked_path(name):
    path = (ROOT / name).resolve(strict=True)
    path.relative_to(BASE)
    return path


def verify(pins):
    actual = {name: digest(checked_path(name)) for name in sorted(pins)}
    changed = [name for name in pins if actual[name] != pins[name]]
    if changed:
        raise ValueError('Pinned bytes changed: ' + ', '.join(changed))
    return actual


def run_regressions():
    results = []
    for module_name, expected in [('test_membership', 15), ('test_short_history', 3)]:
        module = importlib.import_module(module_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if (not result.wasSuccessful() or result.testsRun != expected
                or result.skipped or result.expectedFailures or result.unexpectedSuccesses):
            raise RuntimeError(log.getvalue())
        results.append({
            'module': relative(Path(module.__file__)),
            'tests_passed': result.testsRun,
            'failed': len(result.failures), 'errors': len(result.errors),
            'skipped': len(result.skipped), 'log': log.getvalue(),
        })
    return results


def calendar_review(prior):
    from audit_short_history import assess
    from research_calendar import sessions

    facts = read(BASE / 'SHORT_HISTORY_SOURCE_FACTS.json')
    if facts['through'] != '2026-10-06':
        raise ValueError('Original research horizon changed')
    expected = {
        'SOLS': (242, 252, 253, '2026-10-21'),
        'VSNT': (203, 213, 214, '2026-12-16'),
        'HONA': (79, 89, 90, '2027-06-16'),
    }
    prior_rows = {row['symbol']: row for row in prior['warmup_bounds']}
    if {row['symbol'] for row in facts['securities']} != set(expected):
        raise ValueError('Unexpected short-history source scope')
    rows = []
    for item in facts['securities']:
        symbol = item['symbol']
        start = item['first_possible_session']
        for source in item['sources']:
            if digest(BASE / source['file']) != source['sha256']:
                raise ValueError('Short-history source bytes changed')
        original = assess(start, facts['through'])
        previous = assess(start, '2026-10-20')
        current = assess(start, REVIEW_DATE)
        threshold = str(sessions(start, '2027-12-31')[252].date())
        observed = tuple(row['maximum_possible_observations']
                         for row in [original, previous, current]) + (threshold,)
        if observed != expected[symbol]:
            raise ValueError('Unexpected calendar boundary: ' + symbol)
        if (original != prior_rows[symbol]['at_original_backtest_end']
                or previous != prior_rows[symbol]['at_latest_complete_session']):
            raise ValueError('Prior calendar review no longer reproduces: ' + symbol)
        if current['can_pass_history_requirement'] != (symbol == 'SOLS'):
            raise ValueError('Unexpected calendar-only threshold result')
        if (not original['new_entry_impact_proven_zero']
                or current['corporate_action_entitlements_waived']):
            raise ValueError('Original scope or entitlement requirement changed')
        rows.append({
            'symbol': symbol,
            'at_original_backtest_end': original,
            'at_previous_review_date': previous,
            'at_review_date': current,
            'earliest_calendar_only_253rd_observation': threshold,
            'calendar_impossibility_bound_ended_since_previous_review': symbol == 'SOLS',
            'actual_quote_history_certified': False,
            'membership_or_identity_certified_by_this_review': False,
            'eligibility_changed': False,
        })
    return rows


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    if digest(PRIOR) != PRIOR_SHA256:
        raise ValueError('Prior additive followup bytes changed')
    if len(SEARCH_BATCHES) != 2:
        raise ValueError('Exact parent search requests have not been supplied')
    prior = read(PRIOR)
    source_pins = prior['source_hashes']
    source_before = verify(source_pins)
    protected = dict(prior['protected_hashes_before_and_after'])
    protected[relative(PRIOR)] = PRIOR_SHA256
    for name in ['POLICY.md', 'ASSUMPTIONS.json', 'AVAILABLE_RESEARCH_READINESS.json',
                 'SHORT_HISTORY_IMPACT.json', 'membership.py']:
        protected[relative(BASE / name)] = digest(BASE / name)
    for name, expected in read(BASE / 'AVAILABLE_RESEARCH_CODE_FINGERPRINTS.json').items():
        protected[relative(BASE / name)] = expected
    protected_before = verify(protected)

    stopped = read(BASE / 'sources/oct20_sols_availability/STOPPED.json')
    if (stopped['host'] != 'www.nasdaq.com' or stopped['http_status'] != 403
            or stopped['further_requests_permitted'] is not False):
        raise ValueError('Source-stop policy changed')
    status = read(BASE / 'STATUS.json')
    readiness = read(BASE / 'BACKTEST_READINESS.json')
    adjudication = read(BASE / 'NDX_EVENT_ADJUDICATION.json')
    if (status['backtest_admitted'] is not False
            or readiness['can_start_requested_backtest'] is not False
            or status['qualified_research_backtest_completed'] is not True
            or len(adjudication['pending']) != 6):
        raise ValueError('Admission or research state changed; reassessment required')

    sys.path.insert(0, str(BASE))
    import numpy as np
    import pandas as pd
    runtime = {'python': platform.python_version(), 'numpy': np.__version__,
               'pandas': pd.__version__, 'bytecode_writes_disabled': sys.dont_write_bytecode}
    if runtime != {'python': '3.9.6', 'numpy': '2.0.2', 'pandas': '2.3.3',
                   'bytecode_writes_disabled': True}:
        raise ValueError('Pinned local runtime mismatch')
    regression_results = run_regressions()
    rows = calendar_review(prior)
    source_after = verify(source_pins)
    protected_after = verify(protected)

    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'review_date': REVIEW_DATE,
        'automation_id': 'top100',
        'scope': 'Additive offline cached-byte and calendar-boundary review; no strategy optimization',
        'previous_followup': relative(PRIOR), 'previous_followup_sha256': PRIOR_SHA256,
        'parent_public_search': {
            'provenance': 'Exact requests and outcome supplied by the parent agent; searches preceded this offline run',
            'local_raw_web_response_saved': False,
            'search_log_is_primary_document_evidence': False,
            'batches': SEARCH_BATCHES,
            'outcome': 'Only already-known ECA2026-545 (EA trading halt August 5) and ETA2015-102 (expected merger/listing July 6) were found; no index-specific decision proof. Exact-phrase second batch returned no results.',
            'new_primary_documents_downloaded': 0,
            'new_membership_evidence_admitted': False,
            'search_nonresult_is_not_proof_no_notice_exists': True,
        },
        'source_stop_respected': {'host': stopped['host'], 'prior_status': 403,
                                  'requests_in_this_offline_run': 0,
                                  'blocked_source_retried': False},
        'source_hashes_before': source_before, 'source_hashes_after': source_after,
        'cached_source_pin_count': len(source_pins),
        'protected_hashes_before': protected_before, 'protected_hashes_after': protected_after,
        'protected_file_count': len(protected),
        'warmup_bounds': rows,
        'boundary_interpretation': 'SOLS reaches a maximum of 253 scheduled sessions on 2026-10-21. This only ends the calendar impossibility bound. The imported can_pass_history_requirement field means the calendar count is large enough, not that observed quotes, consecutive data, index membership, identity, or corporate-action treatment pass. No parent history was spliced. The original through-2026-10-06 zero-new-entry-impact bound remains unchanged for all three securities.',
        'actual_quote_observations_examined': 0,
        'validation': {'command': COMMAND, 'runtime': runtime,
                       'method': 'Existing unittest modules executed in-process under the offline wrapper; no subprocess or full-suite execution',
                       'regressions': regression_results, 'tests_passed': 18,
                       'calendar_boundary_checks': 'Exact original/prior/current counts and 253rd date checked for all three symbols; original and prior results reproduce the pinned prior followup'},
        'strict_backtest_admitted': False, 'readiness_gates_changed': False,
        'readiness_gates': readiness['gates'],
        'pending_membership_cases_unchanged': len(adjudication['pending']),
        'qualified_research_already_completed': True,
        'available_data_research_changed': False,
        'eligibility_changed': False, 'performance_rerun': False,
        'observer_runtime_accessed': False, 'cloud_storage_accessed': False,
        'automation_retained': True, 'automation_configuration_modified': False,
        'notification_decision': 'DONT_NOTIFY',
        'notification_reason': 'Strict admission and completed available-data research are unchanged; the calendar threshold alone changes no eligibility and requires no user action.',
        'script_sha256': digest(Path(__file__)),
    }
    # Verify again after report assembly; the only destination opened for writing is new.
    verify(source_pins)
    verify(protected)
    with TARGET.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    verify(source_pins)
    verify(protected)
    print(json.dumps({'report': relative(TARGET), 'sha256': digest(TARGET),
                      'tests_passed': 18, 'source_pins_verified': len(source_pins),
                      'protected_files_unchanged': len(protected),
                      'calendar_maximum_observations': {row['symbol']: row['at_review_date']['maximum_possible_observations'] for row in rows},
                      'notification_decision': 'DONT_NOTIFY'}))


if __name__ == '__main__':
    main()
