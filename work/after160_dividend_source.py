"""Bounded, offline dividend-source feasibility. Never computes a strategy feature."""
import argparse
import csv
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after160_dividend_source.py'
PLAN = 'work/after160_dividend_source_plan.json'
OUT = 'outputs/after160_dividend_source/FEASIBILITY.json'
META = 'outputs/monthly_local_linear70/META.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
BASE = 'outputs/index_union/'
MANIFEST = BASE + 'input_diagnostics/MANIFEST.json'
VALIDATION = BASE + 'NEW_PRICE_VALIDATION.json'
EXECUTION = BASE + 'EXECUTION_CANDIDATES.json'
STATIC = [META, CAL, MANIFEST, VALIDATION, EXECUTION,
          BASE + 'collect_missing_yahoo.py', BASE + 'normalize_new_prices.py',
          BASE + 'build_execution_candidates.py', BASE + 'price_units.py',
          BASE + 'assemble_input_diagnostics.py',
          'work/venv/lib/python3.9/site-packages/yfinance/scrapers/history.py',
          'work/venv/lib/python3.9/site-packages/yfinance/utils.py']
CHECKLIST = [
    'Freeze all 73 distinct original Local67 stock labels and exact source-manifest matches; never infer aliases.',
    'Check declared original CSV, normalized CSV, execution CSV and provenance receipt SHA-256; retain absent or conflicting references.',
    'Read dates, dividend and split fields only; count zero, nonzero, missing and invalid cells without summing cash or computing ratios.',
    'Check sorted unique dates and exact calendar presence for the fixed t-294 through t-42 source-observation window for each original stock slot.',
    'Require positive original evidence for regular versus special dividend classification; a generic Dividends field is insufficient.',
    'Require explicit dividend currency, share/ADR basis and split-adjustment semantics; auto_adjust=false does not independently certify them.',
    'Distinguish serialized zero versus missing cells from certification that the provider supplied a complete action sequence.',
    'Distinguish event date from publication/known-at date and current downloaded history from contemporaneous vintages.',
    'Retain all old identity, action-completeness, quarantine and source-admission limitations.',
    'Close this candidate if sources fail these checks; no collection, no new factor, score, return, order or strategy trial.'
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def path(name):
    p = Path(name)
    need('..' not in p.parts, 'Parent traversal forbidden')
    p = p if p.is_absolute() else ROOT / p
    need(p.is_relative_to(ROOT) and p.resolve() == p, 'Only canonical local source paths allowed')
    need('CloudStorage' not in p.parts and not any(x.endswith('_monitor') for x in p.parts), 'Forbidden source tree')
    return p


def sha(name):
    h = hashlib.sha256()
    with path(name).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads(path(name).read_text())


def iso(value):
    need(type(value) is str and len(value) == 10 and date.fromisoformat(value).isoformat() == value, 'Invalid ISO date')
    return value


def exclusive(name, obj):
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check_axes(meta, sessions):
    need(len(meta['dates']) == len(meta['symbols']) == 122, 'Original 122 rows required')
    need(len(meta['etfs']) == len(set(meta['etfs'])) == 17, 'Original 17 ETF labels required')
    need(sessions == sorted(set(sessions)), 'Calendar must be unique and sorted')
    for d in sessions:
        iso(d)
    for d, symbols in zip(meta['dates'], meta['symbols']):
        iso(d)
        need(d in sessions and len(symbols) == len(set(symbols)) == 49, 'Original candidate axis invalid')
        need(len(set(symbols) & set(meta['etfs'])) == 17, 'Every row must preserve all 17 ETFs')
    stocks = sorted(set(s for row in meta['symbols'] for s in row) - set(meta['etfs']))
    need(len(stocks) == 73, 'Exactly 73 original stock labels required')
    return stocks


def only(rows, key, value):
    found = [r for r in rows if r.get(key) == value]
    need(len(found) == 1, 'Reference must be unique: ' + str(value))
    return found[0]


def make_plan():
    meta, cal, manifest, validation, execution = [read(x) for x in (META, CAL, MANIFEST, VALIDATION, EXECUTION)]
    stocks = check_axes(meta, cal['sessions'])
    pins = {x: sha(x) for x in STATIC + [SELF]}
    rows = []
    for symbol in stocks:
        candidates = [r for r in manifest['price_sources'] if r['symbol'] == symbol]
        need(len(candidates) <= 1, 'Ambiguous source-manifest route')
        row = dict(symbol=symbol, exact_manifest_match=bool(candidates))
        if candidates:
            selected = candidates[0]
            need(selected['provider'] == 'yahoo', 'Unexpected provider outside this fixed source route')
            norm = only([r for r in validation['files'] if r['provider'] == 'yahoo'], 'symbol', symbol)
            raw = only(execution['files'], 'symbol', symbol)
            need(selected['input_file'] == norm['normalized_file'] == raw['input_file'], 'Normalized route mismatch')
            need(selected['input_sha256'] == norm['normalized_sha256'] == raw['input_sha256'], 'Normalized SHA mismatch')
            need(selected['raw_file'] == raw['file'] and selected['raw_sha256'] == raw['sha256'], 'Execution route mismatch')
            original = str(path(norm['source']).relative_to(ROOT))
            need(original == BASE + 'sources/yahoo_prices/' + symbol + '.csv', 'Unexpected original source path')
            receipt = str(Path(original).with_suffix('.provenance.json'))
            refs = {original: norm['sha256'], BASE + norm['normalized_file']: norm['normalized_sha256'],
                    BASE + raw['file']: raw['sha256'], receipt: sha(receipt)}
            row.update(selected_manifest_route=selected, original=original, receipt=receipt,
                       normalized=BASE + norm['normalized_file'], execution=BASE + raw['file'],
                       source_hashes=refs,
                       inherited_identity_verified=raw['identity_verified'],
                       inherited_action_completeness_verified=raw['action_completeness_verified'])
        rows.append(row)
    return dict(schema='after160_dividend_source_plan_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                source_only=True, checklist=CHECKLIST, input_hashes=pins, labels=rows,
                label_count=73, original_stock_slots=3904, original_etf_slots=2074,
                observation_window='Original calendar sessions t-294 through t-42 inclusive (253 dates); coverage only, no aggregation.',
                selection='All original labels; exact match to frozen input_diagnostics selected route. No alias recovery or alternative-provider selection.',
                output=OUT, pass_rule='Every required semantic/source check must be positively evidenced. Missing evidence blocks; no zero-fill or inferred classification.',
                new_strategy_trials=0, actual_feature_calculations=0, actual_returns=0)


def scalar_kind(value):
    if value is None or value.strip().lower() in ('', 'nan', 'na', 'null', 'none'):
        return 'missing'
    try:
        d = Decimal(value)
    except InvalidOperation:
        return 'invalid'
    if not d.is_finite():
        return 'invalid'
    return 'zero' if d == 0 else ('negative' if d < 0 else 'positive')


def scan_csv(stream, date_key, dividend_key, split_key):
    reader = csv.DictReader(stream)
    fields = reader.fieldnames or []
    need(len(fields) == len(set(fields)), 'Duplicate CSV fields')
    need(all(k in fields for k in (date_key, dividend_key, split_key)), 'Required dividend/split/date field missing')
    kinds = {dividend_key: Counter(), split_key: Counter()}
    dates, valid_actions = [], set()
    metadata = {k: set() for k in fields if any(token in k.lower() for token in
                 ('currency', 'special', 'regular', 'dividend_type', 'declaration', 'payment', 'ex_date'))}
    for row in reader:
        need(None not in row, 'Malformed CSV row')
        d = iso(row[date_key])
        dates.append(d)
        values = {k: scalar_kind(row.get(k)) for k in kinds}
        for k, value in values.items():
            kinds[k][value] += 1
        if all(x in ('zero', 'positive') for x in values.values()):
            valid_actions.add(d)
        for k in metadata:
            # Only semantic metadata is retained, never prices or dividend amounts.
            if row.get(k) not in (None, ''):
                metadata[k].add(row[k])
    need(dates == sorted(set(dates)), 'CSV dates must be unique and sorted')
    return dict(columns=fields, rows=len(dates), first=dates[0] if dates else None, last=dates[-1] if dates else None,
                dividend_cells=dict(kinds[dividend_key]), split_cells=dict(kinds[split_key]),
                semantic_metadata={k: sorted(v) for k, v in metadata.items()}), set(dates), valid_actions


def scan_file(name, original=False):
    opener = gzip.open if name.endswith('.gz') else open
    with opener(path(name), 'rt', newline='') as f:
        return scan_csv(f, 'Date' if original else 'date', 'Dividends' if original else 'dividend',
                        'Stock Splits' if original else 'split')


def check_hashes(refs):
    result = []
    for name, expected in refs.items():
        try:
            actual = sha(name)
            result.append(dict(path=name, expected_sha256=expected, actual_sha256=actual, passed=actual == expected))
        except (OSError, ValueError) as e:
            result.append(dict(path=name, expected_sha256=expected, actual_sha256=None, passed=False, error=str(e)))
    return result


def run(plan_sha):
    need(sha(PLAN) == plan_sha, 'Plan digest changed')
    plan = read(PLAN)
    need(plan['checklist'] == CHECKLIST and plan['output'] == OUT, 'Plan contract changed')
    checks = check_hashes(plan['input_hashes'])
    need(all(c['passed'] for c in checks), 'Frozen implementation/metadata changed')
    meta, sessions = read(META), read(CAL)['sessions']
    need(check_axes(meta, sessions) == [r['symbol'] for r in plan['labels']], 'Label scope changed')
    positions = {d: i for i, d in enumerate(sessions)}
    results, observations, all_refs = [], {}, dict(plan['input_hashes'])
    all_refs[PLAN] = plan_sha
    for row in plan['labels']:
        result = dict(symbol=row['symbol'], passed=False, exact_manifest_match=row['exact_manifest_match'], blockers=[])
        if not row['exact_manifest_match']:
            result['blockers'] = ['NO_EXACT_SELECTED_SOURCE_MANIFEST_ROUTE_NO_ALIAS_INFERENCE']
            results.append(result)
            continue
        files = check_hashes(row['source_hashes'])
        result['file_checks'] = files
        result['all_declared_file_hashes_verified'] = all(x['passed'] for x in files)
        all_refs.update({x['path']: x['actual_sha256'] for x in files if x['passed']})
        if not result['all_declared_file_hashes_verified']:
            result['blockers'].append('SOURCE_BYTES_MISSING_OR_HASH_MISMATCH')
            results.append(result)
            continue
        receipt = read(row['receipt'])
        need(receipt['sha256'] == row['source_hashes'][row['original']], 'Original receipt SHA disagrees')
        need(receipt['symbol'] == ('BRK-B' if row['symbol'] == 'BRK.B' else row['symbol']), 'Receipt provider symbol mismatch')
        result['receipt_metadata'] = receipt
        result['source_files'] = {k: row[k] for k in ('original', 'receipt', 'normalized', 'execution')}
        result['scans'] = {}
        try:
            original, original_dates, original_valid = scan_file(row['original'], True)
            normalized, normalized_dates, normalized_valid = scan_file(row['normalized'])
            execution, execution_dates, execution_valid = scan_file(row['execution'])
        except (ValueError, KeyError, TypeError) as e:
            result['blockers'].append('SOURCE_FIELD_VALIDATION_FAILED: ' + str(e))
            results.append(result)
            continue
        result['scans'] = dict(original=original, normalized=normalized, execution=execution)
        result['date_axes_equal'] = original_dates == normalized_dates == execution_dates
        observations[row['symbol']] = (original_dates, original_valid & normalized_valid & execution_valid)
        result['semantic_checks'] = dict(
            original_is_serialized_vendor_dataframe_not_http_response=True,
            generic_dividend_and_split_columns_present=True,
            ordinary_special_classification_certified=False,
            dividend_currency_and_share_or_ADR_basis_certified=False,
            event_date_vs_ex_date_vs_known_at_certified=False,
            provider_split_adjustment_semantics_independently_certified=False,
            serialized_zero_vs_missing_distinguishable=True,
            zero_means_certified_no_dividend_not_missing_provider_event=False,
            historical_action_completeness_certified=row['inherited_action_completeness_verified'],
            historical_identity_certified=row['inherited_identity_verified'],
            contemporaneous_vintage_certified=False)
        # This fixed source route has generic Date/Dividends/Stock Splits only.
        # Any unexpected semantic field requires a new review rather than inferred admission.
        result['unexpected_semantic_metadata_requires_review'] = original['semantic_metadata']
        result['blockers'] += [
            'ORDINARY_VS_SPECIAL_DIVIDEND_CLASSIFICATION_NOT_EVIDENCED',
            'DIVIDEND_CURRENCY_AND_SHARE_OR_ADR_BASIS_NOT_EVIDENCED',
            'EX_DATE_AND_HISTORICAL_KNOWN_AT_NOT_IN_RECEIPT',
            'SPLIT_ADJUSTMENT_IS_INHERITED_CONVERSION_ASSUMPTION_NOT_SOURCE_CERTIFICATION',
            'SERIALIZED_ZERO_CANNOT_CERTIFY_PROVIDER_EVENT_COMPLETENESS',
            'ORIGINAL_IDENTITY_AND_ACTION_COMPLETENESS_FLAGS_REMAIN_FALSE',
            'CURRENT_DOWNLOAD_NOT_CONTEMPORANEOUS_VINTAGE']
        if not result['date_axes_equal']:
            result['blockers'].append('SOURCE_DATE_AXES_DIFFER')
        results.append(result)
    coverage = []
    for i, (d, labels) in enumerate(zip(meta['dates'], meta['symbols'])):
        end = positions[d] - 42
        start = positions[d] - 294
        need(start >= 0, 'Calendar history insufficient')
        required = set(sessions[start:end + 1])
        need(len(required) == 253, 'Fixed source window length changed')
        for j, symbol in enumerate(labels):
            if symbol in meta['etfs']:
                continue
            observed, action_valid = observations.get(symbol, (set(), set()))
            coverage.append(dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=d,
                                 window_first=sessions[start], window_last=sessions[end], required_sessions=253,
                                 observed_sessions=len(required & observed),
                                 sessions_with_nonmissing_nonnegative_action_fields=len(required & action_valid),
                                 missing_sessions=sorted(required - observed),
                                 source_observation_window_complete=required <= observed,
                                 action_field_window_complete=required <= action_valid,
                                 source_admitted=False, feature_computed=False))
    need(len(coverage) == 3904, 'Original stock slot denominator changed')
    summary = {}
    for name, lo, hi in [('all122', 0, 122), ('first88', 0, 88), ('early53', 0, 53), ('recent35', 53, 88), ('later34', 88, 122)]:
        selected = [r for r in coverage if lo <= r['signal_index'] < hi]
        summary[name] = dict(stock_slots=len(selected), exact_route_slots=sum(r['symbol'] in observations for r in selected),
                             source_observation_complete_slots=sum(r['source_observation_window_complete'] for r in selected),
                             action_field_complete_slots=sum(r['action_field_window_complete'] for r in selected), source_admitted_slots=0)
    # All bytes are checked again before publication; no mutable central state is used.
    need(all(c['passed'] for c in check_hashes(all_refs)), 'Input changed during inspection')
    return dict(schema='after160_dividend_source_feasibility_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                passed=False, audit_execution_completed=True, disposition='CLOSED_FOR_THIS_BOUNDED_SOURCE_SCOPE',
                source_admission_changed=False, new_strategy_trials=0, actual_feature_calculations=0,
                actual_orders=0, actual_returns=0, input_hashes=all_refs, plan_sha256=plan_sha,
                counts=dict(labels=len(results), exact_routes=sum(r['exact_manifest_match'] for r in results),
                            hash_verified_routes=sum(r.get('all_declared_file_hashes_verified', False) for r in results),
                            passed_labels=0, blocked_labels=len(results), original_candidate_cells=5978,
                            original_stock_cells=3904, original_etf_cells_excluded=2074),
                summary=summary, labels=results, source_window_coverage=coverage,
                evidence_scope=[
                    'Coverage statistics count source dates/action-field presence only; never imply factor availability or eligibility.',
                    'No raw dividend magnitudes, cash sums, growth ratios, ranks, prices, orders or returns are emitted.',
                    'Selected source route is fixed by the original diagnostic manifest, not a newly selected best provider.',
                    'normalize_new_prices.py labels Yahoo OHLCV split-adjusted; price_units.py reverses later explicit splits, including dividends, and requires complete split history.',
                    'Installed yfinance history.py maps absent merged dividend events to zero and removes dividend currency after reconciliation. Its exact version at acquisition was not pinned by these receipts.',
                    'Original files are collected DataFrame CSVs. Generic action columns and acquisition timestamps cannot certify regular/special type, ex-date publication timing or complete no-event history.',
                    'No source collection or expansion is requested: this bounded candidate is closed, not a numerical trial.'])


class SyntheticTests(unittest.TestCase):
    def test_scalar_categories(self):
        for value, expected in [('', 'missing'), ('NaN', 'missing'), ('0.0', 'zero'), ('1.2', 'positive'),
                                ('-1', 'negative'), ('abc', 'invalid'), ('Infinity', 'invalid')]:
            self.assertEqual(scalar_kind(value), expected)

    def test_literal_zero_is_distinct_from_missing(self):
        x, dates, good = scan_csv(io.StringIO('Date,Dividends,Stock Splits\n2020-01-02,0,0\n2020-01-03,,0\n'), 'Date', 'Dividends', 'Stock Splits')
        self.assertEqual(x['dividend_cells'], {'zero': 1, 'missing': 1})
        self.assertEqual(good, {'2020-01-02'})
        self.assertEqual(len(dates), 2)

    def test_no_type_is_not_inferred(self):
        x, _, _ = scan_csv(io.StringIO('Date,Dividends,Stock Splits\n2020-01-02,5,0\n'), 'Date', 'Dividends', 'Stock Splits')
        self.assertEqual(x['semantic_metadata'], {})
        self.assertNotIn('5', json.dumps(x))

    def test_metadata_retained(self):
        x, _, _ = scan_csv(io.StringIO('Date,Dividends,Stock Splits,currency,dividend_type\n2020-01-02,1,0,USD,special\n'), 'Date', 'Dividends', 'Stock Splits')
        self.assertEqual(x['semantic_metadata']['dividend_type'], ['special'])

    def test_duplicate_or_reversed_dates_rejected(self):
        for rows in ('2020-01-02,0,0\n2020-01-02,0,0\n', '2020-01-03,0,0\n2020-01-02,0,0\n'):
            with self.assertRaises(ValueError):
                scan_csv(io.StringIO('Date,Dividends,Stock Splits\n' + rows), 'Date', 'Dividends', 'Stock Splits')

    def test_absent_columns_rejected(self):
        with self.assertRaises(ValueError):
            scan_csv(io.StringIO('Date,Dividends\n2020-01-02,0\n'), 'Date', 'Dividends', 'Stock Splits')

    def test_duplicate_columns_rejected(self):
        with self.assertRaises(ValueError):
            scan_csv(io.StringIO('Date,Dividends,Dividends,Stock Splits\n'), 'Date', 'Dividends', 'Stock Splits')

    def test_negative_actions_not_valid(self):
        x, _, good = scan_csv(io.StringIO('Date,Dividends,Stock Splits\n2020-01-02,-1,0\n'), 'Date', 'Dividends', 'Stock Splits')
        self.assertEqual(good, set())
        self.assertEqual(x['dividend_cells'], {'negative': 1})

    def test_path_boundaries(self):
        for name in ('../escape', '/etc/passwd', 'outputs/a_monitor/x', 'CloudStorage/x'):
            with self.assertRaises(ValueError):
                path(name)

    def test_ambiguous_reference_rejected(self):
        with self.assertRaises(ValueError):
            only([{'x': 1}, {'x': 1}], 'x', 1)

    def test_date_is_not_timestamp(self):
        with self.assertRaises(ValueError):
            iso('2020-01-02T00:00:00')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test-only', action='store_true')
    p.add_argument('--freeze-plan', action='store_true')
    p.add_argument('--plan-sha256')
    args = p.parse_args()
    if args.test_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.freeze_plan:
        need(args.plan_sha256 is None, 'Choose exactly one mode')
        exclusive(PLAN, make_plan())
        print(json.dumps(dict(plan=PLAN, sha256=sha(PLAN))))
    else:
        need(args.plan_sha256 is not None, 'Frozen plan SHA required')
        report = run(args.plan_sha256)
        exclusive(OUT, report)
        print(json.dumps(dict(output=OUT, sha256=sha(OUT), passed=report['passed'], counts=report['counts'], summary=report['summary'])))


if __name__ == '__main__':
    main()
