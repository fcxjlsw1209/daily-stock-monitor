"""Independent, offline audit of the frozen Shell XLS and quote transcription.

No producer imports. Reads the registered XLS using an isolated, pinned xlrd
wheel; decodes modern Excel serial dates independently with datetime arithmetic.
Only the new QUOTES_REVIEW.json may be written, once.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_source168'
FOLLOW = OUT / 'external_followup'
EXPECTED = {
    'QUOTES_PLAN.json': '68dbf606a456cc1fbfb804476cd4d3766a6556759497d971bbcd98d4c146aef1',
    'QUOTES.json': 'f1ba57a3e66f269a39846a0daf699ecdf5b294db639050c4a2434d8b15cd1086',
    'INDEPENDENT_REVIEW.json': 'fc103480d047ae5b8a577a86c7e733164ddd99e68418d1d0df8869d79c3fc2af',
    'FOLLOWUP_REVIEW.json': 'b33d4af2bcc8907d629182482d36b76aa9126499fc895183a5a68bb344b07a7e',
    'TOTAL_OUTSTANDING_REVIEW.json': 'b33db14025ae96114ce71b58bad746a1cbea0b070e928981c3363b57a84c857a',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(test, message):
    if not test:
        raise AssertionError(message)


def same(actual, expected, path=''):
    """Exact structures; floating arithmetic summaries allow roundoff only."""
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), 'keys ' + path)
        for key in expected:
            same(actual[key], expected[key], path + '/' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'length ' + path)
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, path + '/' + str(index))
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-13), 'number ' + path)
    else:
        require(actual == expected, 'value ' + path)


def csv_index(content):
    rows = list(csv.DictReader(io.StringIO(content)))
    keys = [r['Date'][:10] for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate comparator dates')
    return dict(zip(keys, rows))


def summary(source, comparator, security, column, tolerance):
    joined = []
    invalid = []
    for row in source:
        day = row['date']
        if day not in comparator:
            continue
        try:
            q = float(comparator[day][column])
        except (ValueError, KeyError):
            invalid.append(day)
            continue
        if not math.isfinite(q) or q <= 0:
            invalid.append(day)
            continue
        p = row[security]['close']
        joined.append(dict(date=day, source_close=p, comparison_close=q,
                           absolute_difference=abs(p-q), source_to_comparison_ratio=p/q))
    require(not invalid, 'Invalid overlapping comparator cells: ' + repr(invalid))
    ratios = sorted(r['source_to_comparison_ratio'] for r in joined)
    count = len(ratios)
    middle = (ratios[count//2] if count % 2 else (ratios[count//2-1] + ratios[count//2])/2) if count else None
    differences = sorted(joined, key=lambda r: r['absolute_difference'], reverse=True)
    result = dict(overlapping_days=count,
                  within_half_cent_days=sum(r['absolute_difference'] <= tolerance for r in joined),
                  maximum_absolute_difference=differences[0]['absolute_difference'] if count else None,
                  ratio_min=ratios[0] if count else None, ratio_max=ratios[-1] if count else None,
                  ratio_median=middle, first_last=joined[:2]+joined[-2:], largest_differences=differences[:5])
    return result, [r for r in joined if r['absolute_difference'] > tolerance]


def audit():
    for name, expected in EXPECTED.items():
        require(sha(OUT/name) == expected, 'Frozen output changed: ' + name)
    plan, product = read(OUT/'QUOTES_PLAN.json'), read(OUT/'QUOTES.json')
    pins = {}
    for name, spec in plan['source_files'].items():
        path = ROOT/spec['path']
        require(sha(path) == spec['sha256'], 'Input hash ' + name)
        pins[spec['path']] = spec['sha256']
    producer = ROOT/'work/shell_pool_source168_quotes.py'
    require(sha(producer) == plan['script_sha256'], 'Producer source hash')
    pins[str(producer.relative_to(ROOT))] = sha(producer)
    require(product['plan_sha256'] == sha(OUT/'QUOTES_PLAN.json'), 'Plan binding')
    require(plan['registered_utc'] < product['created_utc'], 'Registration precedes extraction')
    paths = {name: ROOT/spec['path'] for name, spec in plan['source_files'].items()}
    require(sha(paths['wheel']) == '6a33ee89877bd9abc1158129f6e94be74e2679636b8a205b43b85206c3f0bbdd', 'Reader pin')
    require(sha(paths['xls']) == 'b8b74d3b8384afb7abd7d1e3886c2d85c8ef63cea2b0cd77bc8f351fde9ca374', 'XLS pin')
    sys.path.insert(0, str(paths['wheel']))
    import xlrd
    require(str(paths['wheel']) in xlrd.__file__ and xlrd.__version__ == '2.0.1', 'Isolated wheel import')
    book = xlrd.open_workbook(file_contents=paths['xls'].read_bytes(), formatting_info=True)
    require(book.datemode == 0 and book.sheet_names() == ['Share Data'], 'Sheet/date system')
    sheet = book.sheet_by_name('Share Data')
    headers = [sheet.row_values(i) for i in range(4)]
    expected_headers = [
        ['Period', '01/04/2016 - 11/30/2020', '', '', ''],
        ['Share', 'Royal Dutch Shell A ADR (NYSE)', '', 'Royal Dutch Shell B ADR (NYSE)', ''],
        ['Currency', 'USD', '', 'USD', ''],
        ['Date', 'Close', 'Volume', 'Close', 'Volume']]
    require(headers == expected_headers == product['header_rows'], 'Header transcription')
    require(sheet.ncols == 5 and sheet.nrows == 1239, 'Raw dimensions')
    source = []
    formats = set()
    for n in range(4, sheet.nrows):
        serial = sheet.cell_value(n, 0)
        require(math.isfinite(serial) and int(serial) == serial and serial > 61, 'Date serial')
        # Modern Excel serials include the fictitious 1900-02-29: base 1899-12-30.
        day = (date(1899, 12, 30) + timedelta(days=int(serial))).isoformat()
        require(plan['period'][0] <= day <= plan['period'][1], 'Date out of scope')
        formats.add(book.format_map[book.xf_list[sheet.cell_xf_index(n, 0)].format_key].format_str)
        row = {'date': day, 'xls_row': n+1}
        for security, start in [('class_A', 1), ('class_B', 3)]:
            price, volume = sheet.cell_value(n, start), sheet.cell_value(n, start+1)
            require(all(sheet.cell_type(n, k) == xlrd.XL_CELL_NUMBER for k in [start, start+1]), 'Quote cell type')
            require(math.isfinite(price) and price > 0, 'Invalid price')
            require(math.isfinite(volume) and volume >= 0 and int(volume) == volume, 'Invalid volume')
            row[security] = {'close': price, 'volume': int(volume)}
        source.append(row)
    require(source == product['rows'], 'Exact all-row/cell transcription')
    days = [row['date'] for row in source]
    require(days == sorted(set(days)), 'Unsorted or duplicate dates')
    require(product['row_count'] == len(source) and product['first_date'] == days[0] and product['last_date'] == days[-1], 'Output boundaries')
    spy = csv_index(paths['spy'].read_text())
    spy_days = {d for d in spy if plan['period'][0] <= d <= plan['period'][1]}
    missing, extra = sorted(spy_days-set(days)), sorted(set(days)-spy_days)
    require(missing == product['missing_vs_spy'] and extra == product['extra_vs_spy'], 'Calendar difference')
    signals = read(paths['signal_dates'])
    signal_days = [row['asof'][:10] for row in signals]
    require(len(signal_days) == len(set(signal_days)), 'Duplicate asof dates')
    require(len(signal_days) == product['early_signal_asof_dates'], 'As-of count')
    require(sorted(set(signal_days)-set(days)) == product['missing_early_signal_asof_dates'], 'As-of coverage')
    comparisons, mismatches = {}, {}
    yahoo = csv_index(paths['yahoo_shel'].read_text())
    tolerance = plan['overlap_check']['close_tolerance_usd']
    for security in ['class_A', 'class_B']:
        for col in ['Close', 'Adj Close']:
            key = 'yahoo_SHEL_' + col + '_' + security
            comparisons[key], mismatches[key] = summary(source, yahoo, security, col, tolerance)
    with zipfile.ZipFile(paths['stooq']) as z:
        for security, member in [('class_A', 'Stocks/rds-a.us.txt'), ('class_B', 'Stocks/rds-b.us.txt')]:
            comparator = csv_index(z.read(member).decode('utf-8-sig'))
            key = 'stooq_' + security
            comparisons[key], mismatches[key] = summary(source, comparator, security, 'Close', tolerance)
    same(comparisons, product['comparisons'], '/comparisons')

    receipt = read(paths['receipt'])
    probe_path = ROOT/receipt['plan']
    probe = read(probe_path)
    pins[str(probe_path.relative_to(ROOT))] = sha(probe_path)
    require(receipt['sha256'] == sha(paths['xls']) and receipt['bytes'] == paths['xls'].stat().st_size, 'XLS receipt content')
    require(ROOT/receipt['path'] == paths['xls'], 'Receipt target')
    require(receipt['origin_page'] == probe['source'], 'Browser origin binding')
    require(probe['registered_utc'] < receipt['retrieved_utc'] < plan['registered_utc'], 'Chronology')
    for field in ['classes', 'from', 'to', 'currency']:
        require(receipt[field] == probe['observed_selection'][field], 'Requested selection')
    chain_receipts = {}
    bodies = {}
    for name in ['historical_page', 'historical_model', 'euroland_lookup', 'euroland_old_structure']:
        rp = FOLLOW/(name+'.receipt.json')
        r = read(rp)
        bp = ROOT/r['path']
        require(r['status'] == 200 and r['sha256'] == sha(bp) and r['bytes'] == bp.stat().st_size, 'Chain receipt ' + name)
        pins[str(rp.relative_to(ROOT))] = sha(rp)
        pins[str(bp.relative_to(ROOT))] = sha(bp)
        bodies[name] = bp.read_text()
        chain_receipts[name] = {'url': r['url'], 'body_sha256': sha(bp), 'receipt_sha256': sha(rp)}
    require(chain_receipts['historical_model']['url'] in bodies['historical_page'], 'Shell HTML to model')
    require(chain_receipts['euroland_lookup']['url'] in bodies['historical_model'], 'Model to embedded tool')
    require('/tools/splookup/?companycode=uk-rdsa&v=old&lang=en-gb' in html.unescape(bodies['euroland_lookup']), 'Tool to old structure')
    old = html.unescape(bodies['euroland_old_structure'])
    require('/tools/splookup/download/?companycode=uk-rdsa&v=old&lang=en-gb' in old, 'Old structure to download')
    securities = []
    for match in re.finditer(r'<tr\b([^>]+)>(.*?)</tr>', old, re.S):
        attrs, inner = match.groups()
        if 'Royal Dutch Shell A ADR (NYSE)' in inner or 'Royal Dutch Shell B ADR (NYSE)' in inner:
            attr = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
            label = re.search(r'Royal Dutch Shell [AB] ADR \(NYSE\)', inner).group()
            securities.append(dict(label=label, instrument=attr['instrument'], isin=attr['isin'], currency=attr['currency']))
    require(securities == [dict(label='Royal Dutch Shell A ADR (NYSE)', instrument='20642', isin='US7802592060', currency='USD'), dict(label='Royal Dutch Shell B ADR (NYSE)', instrument='20643', isin='US7802591070', currency='USD')], 'HTML security identity')
    result = {
        'schema': 'shell168_quotes_independent_review_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_TRANSCRIPTION_AND_LOCAL_COMPARISONS_ONLY',
        'reviewer_script': str(Path(__file__).relative_to(ROOT)), 'reviewer_script_sha256': sha(Path(__file__)),
        'producer_imported': False, 'network_used': False, 'reader': {'version': xlrd.__version__, 'wheel': str(paths['wheel'].relative_to(ROOT))},
        'reviewed_frozen_outputs': EXPECTED, 'source_pins': pins,
        'workbook': {'sheet': sheet.name, 'rows_including_four_headers': sheet.nrows, 'columns': sheet.ncols,
                     'date_system': 'Excel 1900, decoded independently from integral serials using 1899-12-30 base',
                     'date_formats': sorted(formats), 'headers': headers,
                     'all_rows_match_exactly': True, 'quote_and_volume_cells_checked': 4*len(source),
                     'date_cells_checked': len(source), 'first_record': source[0], 'last_record': source[-1]},
        'coverage': {'source_days': len(days), 'local_SPY_days': len(spy_days), 'missing_vs_SPY': missing,
                     'extra_vs_SPY': extra, 'requested_asof_days': len(signal_days), 'missing_asof_days': sorted(set(signal_days)-set(days)),
                     'asof_range': [min(signal_days), max(signal_days)], 'asof_dates_checked': signal_days,
                     'calendar_is_local_SPY_comparison_only': True, 'imputation_performed': False},
        'comparisons': comparisons, 'tolerance_USD': tolerance,
        'yahoo_Close_class_A_discrepancies': mismatches['yahoo_SHEL_Close_class_A'],
        'all_six_comparison_objects_match': True,
        'source_chain': {'links_checked': chain_receipts, 'class_metadata': securities,
                         'receipt_chronology_consistent': True, 'download_receipt_consistent_with_XLS': True,
                         'browser_execution_independently_replayed': False,
                         'provenance_scope': 'Saved Shell-to-Euroland links and byte-matched download receipt; this offline audit cannot independently witness the browser download or certify the full HTTP response.'},
        'limits': [
            'Two dates are absent versus the local SPY file; all 53 requested asof dates are present. No prices are imputed.',
            'Workbook says Close and USD for the two NYSE ADR classes, but provides no explicit raw/split/dividend adjustment declaration.',
            'The almost-complete Yahoo SHEL Close match corroborates the source A price series numerically; the SHEL filename alone does not establish historical class identity or raw-price methodology.',
            'The isolated 2016-03-09 discrepancy remains unresolved; this audit does not choose one source as truth.',
            'The producer field named within_half_cent_days uses the registered 0.0051 USD tolerance, slightly more than exactly half a cent.',
            'Historical rows were downloaded in 2026. Neither release-time availability nor original historical vendor bytes are certified.',
            'Price evidence alone does not certify net shares outstanding, market cap, ranking, pool eligibility or a strategy result.'
        ],
        'market_cap_computed': False, 'rankings_computed': False, 'returns_computed': False,
        'pool_changed': False, 'observer_changed': False, 'existing_files_modified': False,
    }
    for name, expected in EXPECTED.items():
        require(sha(OUT/name) == expected, 'Post-check output changed: ' + name)
    for path, expected in pins.items():
        require(sha(ROOT/path) == expected, 'Post-check source changed: ' + path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    report = audit()
    if args.write:
        with (OUT/'QUOTES_REVIEW.json').open('x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'workbook', 'coverage', 'yahoo_Close_class_A_discrepancies', 'all_six_comparison_objects_match']}, indent=2))
    if args.write:
        print('QUOTES_REVIEW.json SHA256 ' + sha(OUT/'QUOTES_REVIEW.json'))
