"""Offline source-only audit. No factors, portfolio selection or return calculation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_liquidity162_sources'
PLAN_SHA = 'f77c543bb236b58ae8e20cbc7ca3f1c06f1c895cd1c6a793eab3b1a911ae0dd7'
PROTECTED = ROOT / 'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
PROTECTED_SHA = '89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb'
FIELDS = ('liquid_financial_assets', 'short_term_debt', 'current_portion_long_term_debt',
          'long_term_debt', 'total_assets')
CODE = ('work/liquidity162_audit.py', 'work/test_liquidity162_audit.py',
        'work/liquidity162_msft.py', 'work/test_liquidity162_msft.py',
        'work/liquidity162_nvda.py', 'work/test_liquidity162_nvda.py',
        'work/liquidity162_independent.py', 'work/profitability141_issuer_statements.py',
        'outputs/monthly_earnings_repair/event_schema.py',
        'outputs/monthly_earnings_repair/research_calendar.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, expected):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_relative_to(ROOT) or p.resolve() != p:
        raise ValueError('Local source binding failed: ' + str(p))
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Local source binding failed: ' + str(p))
    return raw


def write_new(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def source_plan():
    plan = json.loads(pinned(OUT / 'SOURCE_PLAN.json', PLAN_SHA))
    for path, sha in plan['source_hashes'].items():
        pinned(path, sha)
    keys = [(r['ticker'], r['fiscal_year']) for r in plan['source_records']]
    if sorted(keys) != sorted((t, y) for t in ('MSFT', 'NVDA') for y in range(2015, 2027)):
        raise ValueError('Unexpected source scope')
    return plan


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def iso_date(text):
    for fmt in ('%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y'):
        try:
            return datetime.strptime(norm(text).replace('Sept ', 'Sep '), fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError('Invalid release date literal')


def release_evidence(html, record):
    """Reparse publication and fiscal identity, independent of the amount parsers."""
    soup = BeautifulSoup(html, 'html.parser')
    text = norm(soup.get_text(' ', strip=True))
    date_pattern = r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})'
    if record['ticker'] == 'MSFT':
        ident = re.search(r'Earnings Release FY(\d{2}) Q([1-4])', text, re.I)
        published = re.search(r'REDMOND,\s+Wash\.(?:\s|—|–|-)*' + date_pattern, text)
        if not ident or not published:
            raise ValueError('Missing MSFT release identity or dateline')
        year, quarter = int(ident[1]) + 2000, int(ident[2])
        opening = text[published.end():published.end()+1800]
        period = re.search(r'(?:quarter|fiscal year) ended\s+' + date_pattern, opening, re.I)
        title = ident[0]
        literal = published[1]
    else:
        title_node = soup.select_one('.article-title')
        date_node = soup.select_one('.article-date')
        body_node = soup.select_one('.article-body')
        if not all((title_node, date_node, body_node)):
            raise ValueError('Missing NVDA original release selectors')
        title = norm(title_node.get_text(' ', strip=True))
        ident = re.search(r'fourth\s+quarter\s+(?:and\s+)?fiscal\s+(\d{4})', title, re.I)
        if not ident:
            raise ValueError('NVDA fiscal identity mismatch')
        year, quarter = int(ident[1]), 4
        literal = norm(date_node.get_text(' ', strip=True))
        period = re.search(r'\bended\s+' + date_pattern,
                           norm(body_node.get_text(' ', strip=True))[:5000], re.I)
    if not period:
        raise ValueError('Missing actual fiscal period in release opening')
    if (year, quarter, iso_date(literal), iso_date(period[1])) != (
            record['fiscal_year'], 4, record['release_date'], record['fiscal_period_end']):
        raise ValueError('Original release facts differ from registered event')
    return {'title': title, 'release_date_literal': literal,
            'period_end_literal': period[1], 'reparsed_from_original_html': True}


def runtime():
    return {'python': sys.version, **{p: importlib.metadata.version(p)
            for p in ('numpy', 'pandas', 'beautifulsoup4', 'exchange-calendars')}}


def tests():
    before = {p: digest(ROOT/p) for p in CODE}
    names = ('test_liquidity162_audit', 'test_liquidity162_msft',
             'test_liquidity162_nvda', 'liquidity162_independent')
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError('Synthetic tests failed; no code registration')
    after = {p: digest(ROOT/p) for p in CODE}
    if before != after:
        raise ValueError('Code changed during synthetic validation')
    write_new('SYNTHETIC_TESTS.json', {'created_utc': now(), 'passed': True,
        'tests_run': result.testsRun, 'synthetic_only': True,
        'code_hashes': after, 'runtime': runtime(),
        'log': stream.getvalue()})


def register():
    source_plan()
    validation = json.loads((OUT/'SYNTHETIC_TESTS.json').read_text())
    hashes = {p: digest(ROOT/p) for p in CODE}
    if (not validation['passed'] or not validation['synthetic_only']
            or validation['code_hashes'] != hashes or validation['runtime'] != runtime()):
        raise ValueError('Current code requires successful synthetic validation')
    write_new('EXTRACTION_CODE.json', {
        'created_utc': now(), 'source_plan_sha256': PLAN_SHA,
        'code_hashes': hashes, 'runtime': runtime(),
        'synthetic_tests_sha256': digest(OUT/'SYNTHETIC_TESTS.json'),
        'financial_amount_extraction_started': False,
        'policy': 'Candidate means source-field proxy only; no ratio, strategy admission or performance claim.',
        'currency_policy': 'Preserve displayed currency label. A dollar symbol alone is not explicit USD evidence. No conversion.',
        'exception_policy': 'Reject ambiguous fields; keep every source. Unexpected parser failure halts instead of omitting a source.',
        'target_unchanged': True})
    print(json.dumps({'registered': len(CODE), 'sha256': digest(OUT/'EXTRACTION_CODE.json')}))


def check_registry(code, validation, hashes, actual_runtime):
    if code['source_plan_sha256'] != PLAN_SHA or code['runtime'] != actual_runtime:
        raise ValueError('Code plan/runtime mismatch')
    if (set(code['code_hashes']) != set(CODE) or code['code_hashes'] != hashes
            or validation.get('code_hashes') != hashes or validation.get('runtime') != actual_runtime
            or validation.get('passed') is not True or validation.get('synthetic_only') is not True):
        raise ValueError('Incomplete or unvalidated code registry')


def bound():
    plan = source_plan()
    code = json.loads((OUT/'EXTRACTION_CODE.json').read_text())
    if code['synthetic_tests_sha256'] != digest(OUT/'SYNTHETIC_TESTS.json'):
        raise ValueError('Synthetic test receipt changed')
    validation = json.loads(pinned(OUT/'SYNTHETIC_TESTS.json', code['synthetic_tests_sha256']))
    check_registry(code, validation, {p:digest(ROOT/p) for p in CODE}, runtime())
    for p, sha in code['code_hashes'].items():
        pinned(p, sha)
    return plan


def verify_schema(row):
    if row['status'] not in ('candidate', 'rejected') or set(row['fields']) != set(FIELDS):
        raise ValueError('Invalid extraction schema')
    for value in row['fields'].values():
        if value is None or value.get('value_reported') is None:
            continue
        if not Decimal(value['value_reported']).is_finite() or value['scale'] not in (1000, 1000000):
            raise ValueError('Invalid amount/scale')
        if not isinstance(value['raw_cells'], list) or not isinstance(value['row_index'], int):
            raise ValueError('Missing raw-cell locator')
    if row['status'] == 'candidate':
        required = set(FIELDS) if row['ticker'] == 'MSFT' else set(FIELDS)-{'current_portion_long_term_debt'}
        if any(row['fields'][k] is None or row['fields'][k].get('value_reported') is None for k in required):
            raise ValueError('Candidate missing a required field')
    if set(row['current_debt_components']) & set(row['noncurrent_debt_components']):
        raise ValueError('Current/noncurrent double count')


def extract():
    plan = bound()
    if (OUT/'EXTRACTED.json').exists():
        raise FileExistsError('Extraction already recorded')
    import liquidity162_msft
    import liquidity162_nvda
    sys.path.insert(0, str(ROOT/'outputs/monthly_earnings_repair'))
    from event_schema import normalize_event
    from research_calendar import market_calendar
    from profitability141_issuer_statements import receipt_check
    calendar_path = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
    calendar_record = json.loads(pinned(calendar_path, plan['source_hashes'][str(calendar_path)]))
    calendar = market_calendar(calendar_record['start'], calendar_record['end'])
    if [str(s.date()) for s in calendar.schedule.index] != calendar_record['sessions']:
        raise ValueError('Pinned calendar sessions differ')
    rows = []
    for record in plan['source_records']:
        raw = pinned(record['raw_path'], record['raw_sha256'])
        if record['ticker'] == 'MSFT':
            receipt_path = Path(record['raw_path']).with_suffix('.meta.json')
            receipt = json.loads(pinned(receipt_path, plan['source_hashes'][str(receipt_path)]))
            module = liquidity162_msft
        else:
            receipt_path = ROOT/'outputs/monthly_earnings_repair/nvidia/manifest.json'
            receipt = json.loads(pinned(receipt_path, plan['source_hashes'][str(receipt_path)]))[record['source_url']]
            module = liquidity162_nvda
        receipt_check(raw, receipt, record)
        html = raw.decode('utf-8')
        publication = release_evidence(html, record)
        row = module.extract(html, record)
        verify_schema(row)
        availability = normalize_event({'symbol': record['ticker'], 'asset_type': 'equity',
            'official': {'verified': True, 'actual_period_end': record['fiscal_period_end'],
            'publication_date': record['release_date'], 'publication_timezone': record['publication_timezone'],
            'source_url': record['source_url'], 'publication_kind': 'release'}}, calendar)
        if availability['status'] != 'ready' or availability['known_by_is_actual_publication']:
            raise ValueError('Unexpected availability admission status: ' + str(availability))
        row.update(source_binding=record, publication_evidence=publication,
                   availability=availability,
                   receipt_binding={'path': str(receipt_path), 'sha256': digest(receipt_path)})
        rows.append(row)
    write_new('EXTRACTED.json', {'created_utc': now(), 'source_plan_sha256': PLAN_SHA,
        'code_binding_sha256': digest(OUT/'EXTRACTION_CODE.json'), 'records': rows,
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'financial_ratios_computed': 0, 'scores_computed': 0, 'holdings_computed': 0, 'returns_computed': 0})
    print(json.dumps({'extracted': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows))}))


def review():
    plan = bound()
    data = json.loads((OUT/'EXTRACTED.json').read_text())
    if data['code_binding_sha256'] != digest(OUT/'EXTRACTION_CODE.json'):
        raise ValueError('Changed extraction code binding')
    from liquidity162_independent import verify
    result = verify(plan, data['records'])
    result.update(created_utc=now(), source_plan_sha256=PLAN_SHA,
                  extraction_sha256=digest(OUT/'EXTRACTED.json'),
                  code_binding_sha256=digest(OUT/'EXTRACTION_CODE.json'))
    write_new('INDEPENDENT_REVIEW.json', result)
    print(json.dumps(result, ensure_ascii=False))


def protect():
    bound()
    expected = json.loads(pinned(PROTECTED, PROTECTED_SHA))
    bad = []
    for path, sha in expected.items():
        try:
            pinned(path, sha)
        except (ValueError, OSError) as exc:
            bad.append({'path': path, 'error': str(exc)})
    write_new('PROTECTED_AFTER.json', {'created_utc': now(), 'manifest_sha256': PROTECTED_SHA,
        'checked_files': len(expected), 'passed': not bad, 'failures': bad,
        'live_observer_root_accessed': False})
    print(json.dumps({'protected_files': len(expected), 'passed': not bad}))
    if bad:
        raise ValueError('Protected archive mismatch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('tests', 'register', 'extract', 'review', 'protect'))
    args = parser.parse_args()
    globals()[args.phase]()
