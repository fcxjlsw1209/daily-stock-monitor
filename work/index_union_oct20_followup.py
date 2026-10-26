"""Offline source and warmup-bound review; no admission or performance changes."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/index_union'
sys.path.insert(0, str(BASE))
from audit_short_history import assess
from research_calendar import sessions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = BASE / 'FOLLOWUP_2026-10-20_SOLS.json'
    if target.exists():
        raise FileExistsError(target)
    earlier = json.loads((BASE / 'sources/oct19_dtcc_symbol_bridge/SOURCE_REVIEW.json').read_text())
    protected = earlier['protected_inputs_sha256']
    for name, expected in protected.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Protected input changed: ' + name)

    facts = json.loads((BASE / 'SHORT_HISTORY_SOURCE_FACTS.json').read_text())
    pins = {}
    rows = []
    for item in facts['securities']:
        for source in item['sources']:
            path = BASE / source['file']
            if digest(path) != source['sha256']:
                raise ValueError('Source changed: ' + source['file'])
            pins[str(path.relative_to(ROOT))] = source['sha256']
        dates = sessions(item['first_possible_session'], '2027-12-31')
        threshold = str(dates[252].date())
        rows.append({
            'symbol': item['symbol'],
            'at_original_backtest_end': assess(item['first_possible_session'], '2026-10-06'),
            'at_latest_complete_session': assess(item['first_possible_session'], '2026-10-20'),
            'earliest_calendar_only_253rd_observation': threshold,
            'threshold_is_not_proof_of_quotes_or_membership': True,
        })

    retrospective = json.loads((BASE / 'sources/oct08_validation/sols_retrospective_review.json').read_text())
    exchange = json.loads((BASE / 'sources/oct09_validation/sols_exchange_reviewed_facts.json').read_text())
    for record in (retrospective, exchange):
        source = BASE / record['source_file']
        if digest(source) != record['source_sha256']:
            raise ValueError('SOLS reviewed source changed')
        pins[str(source.relative_to(ROOT))] = record['source_sha256']
    acquisition_dir = BASE / 'sources/oct20_sols_availability'
    review = json.loads((acquisition_dir / 'REVIEW.json').read_text())
    for filename in ('REVIEW.json', 'COLLECTION.json', 'SCOPE.json', 'STOPPED.json',
                     'highlights.html', 'highlights.receipt.json', 'annual.receipt.json'):
        path = acquisition_dir / filename
        pins[str(path.relative_to(ROOT))] = digest(path)
    for name in ('SHORT_HISTORY_SOURCE_FACTS.json', 'audit_short_history.py',
                 'research_calendar.py', 'CALENDAR_EXCEPTIONS.json', 'dynamic_features.py',
                 'test_membership.py', 'test_short_history.py'):
        pins[str((BASE / name).relative_to(ROOT))] = digest(BASE / name)
    calendar = json.loads((BASE / 'CALENDAR_EXCEPTIONS.json').read_text())
    for row in calendar['closures']:
        p = BASE / row['source_file']
        if digest(p) != row['source_sha256']:
            raise ValueError('Calendar closure changed')
        pins[str(p.relative_to(ROOT))] = row['source_sha256']

    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'automation_id': 'top100',
        'scope': 'Cached source-layout audit and calendar-only entry-bound refresh; no optimization',
        'source_hashes': pins,
        'visual_review': {
            'source': retrospective['source_file'], 'page_one_based': 3,
            'method': 'Poppler render inspected by assistant; complete page including table heading and source line',
            'table_subject': 'Off-cycle additions and deletions in 2025; source Nasdaq Global Indexes',
            'effective_addition_date_reported': '2025-10-20',
            'effective_deletion_date_reported': '2025-11-10',
            'symbol_used_in_both_table_columns': 'SOLS',
            'explicit_SOLSV_label_or_CUSIP_in_this_row': False,
            'conclusion': 'The row reports index changes but does not explicitly identify its when-issued constituent security. Coincidence with SOLSV first trading is not a certified identity binding.',
            'publication_limit': 'January 2026 retrospective cannot establish October 2025 advance knowledge; exact historical release timestamp remains unverified.'
        },
        'fresh_acquisition_review': str((acquisition_dir / 'REVIEW.json').relative_to(BASE)),
        'fresh_acquisition_outcome': 'First official Nasdaq HTML URL returned HTTP403; source stopped, second URL not requested; no new primary document or publication metadata admitted.',
        'warmup_bounds': rows,
        'boundary_warning': 'The original through-2026-10-06 warmup result is not an indefinite exclusion. A calendar threshold reaching253 only removes that impossibility proof; valid consecutive quotes, identity, membership and corporate-action gates still apply.',
        'validation': {
            'membership_command': './run_local.sh outputs/index_union/test_membership.py',
            'membership_tests_passed': 15,
            'short_history_command': './run_local.sh outputs/index_union/test_short_history.py',
            'short_history_tests_passed': 3,
            'scope': 'Existing targeted timing/warmup regressions, not the full suite or strategy performance'
        },
        'strict_backtest_admitted': False,
        'readiness_gates_changed': False,
        'pending_membership_cases_unchanged': 6,
        'qualified_research_already_completed': True,
        'performance_rerun': False,
        'observer_runtime_accessed': False,
        'cloud_storage_accessed': False,
        'protected_hashes_before_and_after': protected,
        'script_sha256': digest(Path(__file__)),
    }
    for name, expected in protected.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Protected input changed during review: ' + name)
    with target.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({'report': str(target.relative_to(ROOT)),
                      'sha256': digest(target), 'warmup_bounds': rows,
                      'strict_admission_changed': False}))


if __name__ == '__main__':
    main()
