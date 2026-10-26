"""Persist source-only findings and receipts; no new research trial or outcome math."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

ROOT = Path('/Users/shiweili/code/monthly-stock-research')
BASE = ROOT / 'outputs/monthly_etf_primary_flow_sources'
OUT = BASE / 'SOURCE_FEASIBILITY_SUMMARY.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUT.exists():
        raise RuntimeError('Preserve existing write-once summary')
    expected = {
        'outputs/monthly_target70_research/CONTINUATION_STATUS.json': '67146c139586e4439d172899a713a353d2e734afb11b253e3a7c661e1e3bb079',
        'outputs/monthly_etf_primary_flow_sources/ssga/FEASIBILITY.json': 'aea4f2ff7436574c7385c1006313da58ebd367741f29ef6a8d9d652d0df9ee30',
        'outputs/monthly_etf_primary_flow_sources/ishares/FEASIBILITY.json': 'ea307ef6b297b290c6cd4c339f97a3cdfb86e08ccf8cd122ed6b5970fc07d51a',
        'outputs/monthly_etf_primary_flow_sources/other/FEASIBILITY.json': 'a2bfcd8daaa51597f88a555b80f96075d5761df09656502ef7f84429cff37015',
        'outputs/monthly_etf_primary_flow_sources/CALENDAR_ACCOUNTING_AUDIT.json': '2cf542a5755cab8e99100b36fcf04ca60dd60efc4d3feeaea9a37b559399f482',
        'outputs/monthly_etf_primary_flow_sources/LAGGED_COVERAGE_SUPPLEMENT.json': 'c6b5c49cfb591ffb13ca7137c9b18e3d786dd5bd6e4f9b7f46ece86ca6255ab3',
        'outputs/monthly_etf_primary_flow_sources/ishares_actions/SPLIT_CONVENTION_AUDIT.json': '80c22271244c10fc12b297a30578006e423689acb5a71e01da43264e969a5d8f',
        'work/etf_flow_method_clock_review.json': '20197ece40abfa6628aea33ce7df1176be465853a41be97f190f40e0c0468378',
        'work/etf_flow_ssga_method_review.json': 'ec3a91e7dcd3246f4690b32672519d94d9c9bab08971da2ff5b378b837ae2627',
        'work/etf_flow_source_quality_independent.json': '9b57ec682549bb2b528c5125af121325e8627a6c0aa788eddef3cb464c34d858',
        'work/etf_reported_share_growth_feasibility.json': '3913532e20a37a18cceda43619347cf8c421cf5f22fd0ab0052449a728004855',
        'work/etf_flow_source_quality.py': '8d9cecea170e79b7e645babc2829a5a4ac30c543a07833e6d3afac96f845f010',
        'work/test_etf_flow_source_quality.py': '5e96d232bf3ba9521cff7e11782144d8fe569481da57b49793706d99ea29c200',
    }
    for name, digest in expected.items():
        assert sha(ROOT / name) == digest, name
    receipts, statuses, raw_checked = [], Counter(), 0
    for folder in ('ssga', 'ishares', 'other', 'ssga_method', 'ishares_actions'):
        paths = sorted(set((BASE / folder).glob('*_RETRIEVAL.json')) |
                       set((BASE / folder).glob('*.receipt.json')))
        for path in paths:
            data = json.loads(path.read_text())
            statuses[str(data['status'])] += 1
            item = {'receipt': str(path.relative_to(ROOT)), 'receipt_sha256': sha(path), 'status': data['status']}
            raw_name = data.get('raw_file')
            if raw_name is not None:
                raw = ROOT / raw_name if raw_name.startswith('outputs/') else path.parent / raw_name
                assert raw.resolve().is_relative_to(BASE.resolve()), str(raw)
                assert raw.is_file() and not raw.is_symlink(), str(raw)
                assert sha(raw) == data['sha256'], str(raw)
                assert raw.stat().st_size == data['bytes'], str(raw)
                item.update(raw=str(raw.relative_to(ROOT)), raw_sha256=sha(raw), bytes=data['bytes'])
                raw_checked += 1
            else:
                assert data['status'] not in (200, 201), str(path)
                item['raw_error_body_retained'] = False
            receipts.append(item)
    audit = json.loads((BASE / 'CALENDAR_ACCOUNTING_AUDIT.json').read_text())
    lagged = json.loads((BASE / 'LAGGED_COVERAGE_SUPPLEMENT.json').read_text())
    complete = [r['symbol'] for r in audit['funds'] if r['coverage']['complete_nav_and_shares']]
    complete_lagged = [r['symbol'] for r in lagged['funds'] if r['coverage']['complete_nav_and_shares']]
    assert len(complete) == 13 and set(complete) == set(complete_lagged)
    assert audit['original_registered_price_axis_check']['same_date_axis']
    tested_aum = {r['symbol']: r['accounting']['checked_rows'] for r in audit['funds']
                  if 'checked_rows' in r['accounting']}
    assert len(tested_aum) == 8
    assert all(not r['accounting'].get('coarse_unit_screen_failures') for r in audit['funds'])
    source_files = ['work/after215_etf_primary_flow_source_leads.json',
                    'work/after215_etf_primary_flow_universe.json',
                    'work/etf_reported_share_growth_feasibility.json',
                    'work/etf_flow_ssga_extract.py', 'work/etf_flow_lagged_coverage.py']
    for name in source_files:
        expected[name] = sha(ROOT / name)
    stops = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sorted(BASE.rglob('STOP*.json'))]
    transport = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sorted(BASE.rglob('*TRANSPORT_ERROR.json'))]
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'SOURCE_FEASIBILITY_COMPLETED_NEW_STRATEGY_NOT_REGISTERED',
        'goal_turn_classification': 'progress',
        'progress': 'Collected issuer histories and action evidence; verified daily coverage; identified a concrete NAV/share date mismatch that changes the next research design.',
        'objective_unchanged': '2016-08-17 to 2026-09-24 same-close zero-cost CAGR at least70%, dominant skip42 long momentum and monthly long-only EW3 fixed units.',
        'current_best_zero_cost_cagr': 0.6766617372041686,
        'current_best_sharpe': 1.7348767578075763,
        'target_reached': False, 'new_strategy_trials': 0,
        'new_flow_factor_model_order_return_calculations': 0,
        'frozen_observers_modified': False, 'central_strategy_record_modified': False,
        'complete_raw_nav_and_share_date_coverage': {
            'symbols': complete, 'original_window': ['2016-05-18','2026-09-24'], 'sessions': 2603,
            'hypothetical5session_lag_window': ['2016-05-11','2026-09-24'], 'lagged_sessions': 2608,
            'lagged_symbols': complete_lagged, 'missing_required_dates_or_values': 0,
            'raw_nontrading_rows_retained_separately': True,
            'historical_availability_certified': False},
        'independent_net_assets_coarse_unit_check': tested_aum,
        'no_independent_historical_aum_column': ['IGV','IWM','MTUM','SOXX','TLT'],
        'remaining_direct_share_histories': {
            'ARKK': 'Official initial request403; source stopped, no bypass.',
            'SMH': 'Official2011-2026 NAV+AUM workbook obtained, no direct shares column; no inferred share proxy.',
            'QQQ': 'Historical NAV plus only current share snapshot; insufficient share history.',
            'VGT': 'Single2026-09-30 share snapshot; no daily share history.'},
        'verified_method_findings': [
            'XLE/XLK/XLY source tables preserve native units across their2025-12-05 2:1 split; adjust verified effective events, not raw share changes.',
            'XLF2016 event is XLRE in-kind distribution, not an XLF split; NAV declines must not be labeled redemptions.',
            'IGV/SOXX pre2024-split NAV is restated at official anchor dates. Historical shares basis remains unverified.',
            'At2026-03-31 the IGV/SOXX NAV agrees with annual statements, but statement shares appear in the source2026-04-01 row. Two anchors do not establish a global one-session shift.',
            'SOXX/MTUM2026-11-05 announced splits are outside research/history cutoff; do not apply them early.',
            'Observation dates, retrieval UTC and workbook metadata do not prove original publication clocks.'],
        'decision': 'Do not register or evaluate the original same-row NAV-weighted flow design. Evaluate reported-share-growth feasibility only after source/action semantics and a predeclared missing-data policy are fixed.',
        'next_work': [
            'Use the newly saved method review to choose a valid input definition before any feature values or outcomes.',
            'Resolve or explicitly withhold unsupported share-unit/action intervals; keep all original assets in the portfolio universe and make unsupported corrections neutral.',
            'Declare any reporting lag and latest-vintage limitation prospectively; a lag does not certify strict PIT.',
            'Only then register a candidate and matched control, test causality/units, and run the original segmented zero-cost evaluation.'],
        'validation': {'synthetic_tests': 8, 'command': './run_local.sh work/test_etf_flow_source_quality.py',
                       'observed_exit_code': 0,
                       'independent_xlsx_xml_sample_rows': 89, 'independent_numeric_cells': 239,
                       'all_raw_receipt_sha_checks_passed': True},
        'http_status_counts': dict(statuses), 'http_receipts': len(receipts), 'raw_files_verified': raw_checked,
        'receipt_bindings': receipts, 'source_stop_records': stops, 'transport_error_records': transport,
        'evidence_sha256': expected, 'summary_writer_sha256': sha(Path(__file__))}
    with OUT.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'summary': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
                      'receipts': len(receipts), 'raw_files_verified': raw_checked,
                      'status_counts': dict(statuses), 'new_strategy_trials': 0}))

if __name__ == '__main__':
    main()
