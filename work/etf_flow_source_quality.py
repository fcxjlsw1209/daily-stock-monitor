"""Offline calendar and accounting-unit audit. No flow/factor/strategy calculation."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/monthly_etf_primary_flow_sources'
OUT = BASE / 'CALENDAR_ACCOUNTING_AUDIT.json'
NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
PREFIX = '{' + NS['ss'] + '}'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def positive_decimal(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value).replace(',', ''))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result > 0 else None

def parse_ishares(raw):
    text, escapes = re.subn(r'&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)', '&amp;', raw.decode('utf-8'))
    document = ET.fromstring(text)
    sheets = [s for s in document.findall('ss:Worksheet', NS) if s.get(PREFIX + 'Name') == 'Historical']
    if len(sheets) != 1:
        raise ValueError('Expected exactly one Historical sheet')
    table = []
    for row in sheets[0].findall('ss:Table/ss:Row', NS):
        cells = []
        for cell in row.findall('ss:Cell', NS):
            position = int(cell.get(PREFIX + 'Index', str(len(cells) + 1)))
            if position < len(cells) + 1:
                raise ValueError('Nonmonotone sparse cell index')
            cells.extend([''] * (position - 1 - len(cells)))
            cells.append(cell.findtext('ss:Data', default='', namespaces=NS))
        table.append(cells)
    if table[0] != ['As Of', 'NAV per Share', 'Ex-Dividends', 'Shares Outstanding']:
        raise ValueError('Unexpected source columns')
    result = []
    for n, row in enumerate(table[1:], 2):
        if len(row) != 4:
            raise ValueError('Unexpected field count')
        result.append({'date': datetime.strptime(row[0], '%b %d, %Y').date().isoformat(),
                       'nav': row[1], 'shares_outstanding': row[3], 'source_row': n})
    return result, escapes

def coverage(rows, sessions, start, end):
    dates = [r['date'] for r in rows]
    if len(set(dates)) != len(dates):
        raise ValueError('Duplicate date cannot be resolved implicitly')
    mapping = {r['date']: r for r in rows}
    valid_sessions = set(sessions)
    absent = [day for day in sessions if day not in mapping]
    invalid = {key: [day for day in sessions if day in mapping and positive_decimal(mapping[day].get(key)) is None]
               for key in ('nav', 'shares_outstanding')}
    extra = [day for day in dates if start <= day <= end and day not in valid_sessions]
    return {'required_sessions': len(sessions), 'absent_sessions': absent,
            'invalid_session_fields': invalid, 'source_non_session_dates': sorted(extra),
            'complete_nav_and_shares': not absent and not any(invalid.values())}

def accounting_checks(rows, sessions):
    """Reconcile source values, not investment returns; never infer missing AUM."""
    session_set = set(sessions)
    checks, failures, unavailable = [], [], []
    for row in rows:
        if row['date'] not in session_set:
            continue
        nav, shares, assets = (positive_decimal(row.get(k)) for k in ('nav', 'shares_outstanding', 'net_assets'))
        if nav is None or shares is None or assets is None:
            unavailable.append(row['date'])
            continue
        error = abs(nav * shares - assets)
        # A deliberately coarse source-unit screen: half a cent per share plus
        # one cent of assets. It catches split/scale mismatches, not precision.
        tolerance = Decimal('0.005') * shares + Decimal('0.01')
        record = {'date': row['date'], 'absolute_difference': str(error),
                  'relative_difference': str(error / assets), 'coarse_rounding_tolerance': str(tolerance)}
        checks.append(record)
        if error > tolerance:
            failures.append(record)
    return {'checked_rows': len(checks), 'unavailable_dates': sorted(unavailable),
            'coarse_unit_screen_failures': sorted(failures, key=lambda r: r['date']),
            'largest_relative_difference': max(checks, key=lambda r: Decimal(r['relative_difference'])) if checks else None,
            'threshold': 'abs(NAV*shares - net_assets) <= 0.005*shares + 0.01; source-unit screen only, not admission',
            'derived_aum_or_share_proxy_used': False}

def main():
    if OUT.exists():
        raise RuntimeError('Write-once audit already exists; do not overwrite')
    cal_path = ROOT / 'outputs/monthly_earnings_repair/research_calendar.py'
    spec = importlib.util.spec_from_file_location('etf_source_existing_calendar', cal_path)
    calendar_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calendar_module)
    calendar = calendar_module.market_calendar()
    all_sessions = [str(d.date()) for d in calendar.sessions_in_range('2016-01-01', '2026-09-24')]
    start_index = all_sessions.index('2016-08-17') - 63
    sessions = all_sessions[start_index:]
    start, end = sessions[0], sessions[-1]
    input_hashes = {str(cal_path.relative_to(ROOT)): sha(cal_path),
                    'work/etf_flow_source_quality.py': sha(Path(__file__))}
    prices_path = ROOT / 'outputs/monthly_2016_extension/PRICES.npz'
    if sha(prices_path) != 'c877cbe443964d53972114bc654c473608362a6bb31577fb1de5e00cbd3c1485':
        raise ValueError('Registered original price archive mismatch')
    input_hashes[str(prices_path.relative_to(ROOT))] = sha(prices_path)
    with np.load(prices_path, allow_pickle=False) as archive:
        original_dates = [str(d) for d in archive['dates']]
    if original_dates != sorted(set(original_dates)):
        raise ValueError('Original price dates are not unique and increasing')
    original_window = [d for d in original_dates if start <= d <= end]
    original_axis_check = {
        'source': str(prices_path.relative_to(ROOT)), 'only_member_read': 'dates',
        'price_value_arrays_read': False,
        'calendar_dates_absent_from_original_price_axis': sorted(set(sessions) - set(original_window)),
        'original_price_dates_not_in_calendar': sorted(set(original_window) - set(sessions)),
        'same_date_axis': original_window == sessions,
    }
    ssga_path = BASE / 'ssga/FEASIBILITY.json'
    ish_path = BASE / 'ishares/FEASIBILITY.json'
    for path in (ssga_path, ish_path):
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
    funds = []
    for item in json.loads(ssga_path.read_text())['sources']:
        extract = ROOT / item['extraction_file']
        raw = ROOT / item['source_file']
        if sha(extract) != item['extraction_sha256'] or sha(raw) != item['source_sha256']:
            raise ValueError('SSGA immutable source mismatch')
        payload = json.loads(extract.read_text())
        input_hashes[str(extract.relative_to(ROOT))] = sha(extract)
        input_hashes[str(raw.relative_to(ROOT))] = sha(raw)
        rows = payload['rows']
        funds.append({'symbol': item['symbol'], 'provider': 'SSGA',
                      'coverage': coverage(rows, sessions, start, end),
                      'accounting': accounting_checks(rows, sessions)})
    for item in json.loads(ish_path.read_text())['funds']:
        raw = BASE / 'ishares' / item['source_file']
        if sha(raw) != item['source_sha256']:
            raise ValueError('iShares immutable source mismatch')
        input_hashes[str(raw.relative_to(ROOT))] = sha(raw)
        rows, escapes = parse_ishares(raw.read_bytes())
        funds.append({'symbol': item['ticker'], 'provider': 'iShares',
                      'coverage': coverage(rows, sessions, start, end),
                      'in_memory_xml_ampersand_repairs': escapes,
                      'accounting': {'status': 'NO_INDEPENDENT_HISTORICAL_NET_ASSETS_COLUMN',
                                     'derived_aum_or_share_proxy_used': False}})
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Offline source calendar and accounting-unit checks, not flows or strategy evaluation.',
              'status': 'SOURCE_AUDIT_ONLY_NOT_STRICT_PIT_ADMISSION',
              'research_start': '2016-08-17', 'research_end': end,
              'warmup_sessions': 63, 'required_first_date': start,
              'required_sessions': len(sessions), 'funds': funds,
              'calendar_provenance': calendar_module.provenance(), 'input_sha256': input_hashes,
              'original_registered_price_axis_check': original_axis_check,
              'new_flow_factor_model_order_return_calculations': 0,
              'missing_filled': False, 'sources_or_observers_modified': False,
              'unresolved': ['historical publication/revision clock', 'split/restatement conventions',
                             'trade/order/settlement balance semantics', 'four other ETF histories incomplete']}
    with OUT.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'audit': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
                      'required_start': start, 'sessions': len(sessions),
                      'complete_funds': [r['symbol'] for r in funds if r['coverage']['complete_nav_and_shares']],
                      'unit_failures': {r['symbol']: len(r['accounting'].get('coarse_unit_screen_failures', [])) for r in funds}}))

if __name__ == '__main__':
    main()
