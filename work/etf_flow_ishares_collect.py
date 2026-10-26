"""Collect official iShares raw sources separately from offline research.

No credentials, derived flow calculations, signals, forecasts, or orders.
One request per invocation; no retries or automatic redirect following.
403/429 blocks this host for the rest of the America/Los_Angeles day.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib import error, parse, request
from zoneinfo import ZoneInfo

ROOT = Path('/Users/shiweili/code/monthly-stock-research')
OUT = ROOT / 'outputs/monthly_etf_primary_flow_sources/ishares'
ALLOWED_HOSTS = ('www.ishares.com', 'www.blackrock.com')


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def collect(label, url):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Unsafe source label')
    parsed = parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
        raise ValueError('Only unauthenticated official iShares/BlackRock HTTPS sources are allowed')
    OUT.mkdir(parents=True, exist_ok=True)
    day = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    receipt = OUT / (label + '_RETRIEVAL.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        raw = OUT / prior['raw_file']
        if prior['requested_url'] != url or hashlib.sha256(raw.read_bytes()).hexdigest() != prior['sha256']:
            raise ValueError('Existing source does not match request or original SHA')
        return {'reused': True, 'receipt': str(receipt), 'status': prior['status']}
    stop = OUT / ('STOP_' + parsed.hostname + '_' + day + '.json')
    if stop.exists():
        raise RuntimeError('Host stopped for this local day after 403/429; do not retry or bypass')
    record = {'requested_url': url, 'requested_utc': datetime.now(timezone.utc).isoformat(),
              'local_date': day, 'local_timezone': 'America/Los_Angeles',
              'source_role': 'raw issuer source only; no factor or backtest admission',
              'automatic_retries': 0, 'automatic_redirects': False}
    opener = request.build_opener(NoRedirect())
    req = request.Request(url, headers={'User-Agent': 'LocalResearch/1.0', 'Accept': '*/*'})
    try:
        response = opener.open(req, timeout=40)
    except error.HTTPError as exc:
        response = exc
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        with (OUT / (label + '_TRANSPORT_ERROR.json')).open('x') as handle:
            json.dump(record, handle, indent=2)
        raise
    with response:
        body = response.read(40_000_001)
        if len(body) > 40_000_000:
            raise ValueError('Unexpected source size')
        record.update(status=response.code, final_url=response.url,
                      response_headers={key: response.headers.get(key) for key in
                                        ('Content-Type', 'Content-Disposition', 'Last-Modified', 'ETag',
                                         'Date', 'Retry-After', 'Location')},
                      retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    content = (record['response_headers']['Content-Type'] or '').lower()
    suffix = '.xlsx' if body[:2] == b'PK' else ('.xls' if 'excel' in content else '.html')
    raw = OUT / (label + suffix)
    with raw.open('xb') as handle:
        handle.write(body)
    record['raw_file'] = raw.name
    with receipt.open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    if record['status'] in (403, 429):
        with stop.open('x') as handle:
            json.dump({'status': record['status'], 'host': parsed.hostname, 'local_date': day,
                       'receipt': receipt.name, 'retry_after': record['response_headers']['Retry-After'],
                       'policy': 'Stop requests to this host for the local day; no bypass.'}, handle, indent=2)
    return {'reused': False, 'status': record['status'], 'bytes': len(body), 'raw_file': str(raw),
            'receipt': str(receipt), 'sha256': record['sha256']}


def inspect_saved(ticker):
    """Offline metadata inspection only; original workbook bytes remain unchanged."""
    import math
    import xml.etree.ElementTree as ET
    from html.parser import HTMLParser
    from html import unescape

    label = ticker + '_fund_download_20261023'
    receipt_path = OUT / (label + '_RETRIEVAL.json')
    receipt = json.loads(receipt_path.read_text())
    raw_path = OUT / receipt['raw_file']
    raw = raw_path.read_bytes()
    if receipt['status'] != 200 or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('Only original successful, digest-verified source may be inspected')
    text, escapes = re.subn(r'&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)', '&amp;', raw.decode('utf-8'))
    workbook = ET.fromstring(text)
    ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
    prefix = '{' + ns['ss'] + '}'
    tables = {}
    for sheet in workbook.findall('ss:Worksheet', ns):
        rows = []
        for row in sheet.findall('ss:Table/ss:Row', ns):
            values = []
            for cell in row.findall('ss:Cell', ns):
                index = int(cell.get(prefix + 'Index', str(len(values) + 1)))
                while len(values) < index - 1:
                    values.append('')
                values.append(cell.findtext('ss:Data', default='', namespaces=ns))
            rows.append(values)
        tables[sheet.get(prefix + 'Name')] = rows
    historical = tables['Historical']
    headers = historical[0]
    if headers[:4] != ['As Of', 'NAV per Share', 'Ex-Dividends', 'Shares Outstanding']:
        raise ValueError('Unexpected history headers')
    dates = []
    field_valid_dates = {key: [] for key in ('NAV per Share', 'Shares Outstanding')}
    missing = {key: [] for key in field_valid_dates}
    nonpositive = {key: [] for key in field_valid_dates}
    for row in historical[1:]:
        day = datetime.strptime(row[0], '%b %d, %Y').date().isoformat()
        dates.append(day)
        for key in field_valid_dates:
            cell = row[headers.index(key)]
            try:
                value = float(cell.replace(',', ''))
                if not math.isfinite(value):
                    raise ValueError('nonfinite')
            except (ValueError, AttributeError):
                missing[key].append(day)
                continue
            if value <= 0:
                nonpositive[key].append(day)
            else:
                field_valid_dates[key].append(day)
    page_receipt_path = OUT / (ticker + '_product_20261023_RETRIEVAL.json')
    page_receipt = json.loads(page_receipt_path.read_text())
    page_raw = (OUT / page_receipt['raw_file']).read_bytes()
    if hashlib.sha256(page_raw).hexdigest() != page_receipt['sha256']:
        raise ValueError('Product page SHA mismatch')
    page = page_raw.decode('utf-8')
    schema = []
    for script in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', page, re.S):
        schema.extend(json.loads(script).get('@graph', []))
    fund = next(item for item in schema if 'InvestmentFund' in item.get('@type', []))
    if fund.get('alternateName') != ticker:
        raise ValueError('Product ticker identity mismatch')
    identifiers = re.findall(r'productIdentifier=([^"<>\s]+)', unescape(page))
    keyfacts = [item.get('additionalProperty', []) for item in schema if item.get('@id', '').endswith('#key-facts')]
    direct_links = re.findall(r'href="([^"<>]*component=fundDownload[^"<>]*)"', page)
    direct_links = sorted({unescape(link) for link in direct_links})
    if receipt['requested_url'] not in direct_links:
        raise ValueError('Saved download was not the actual product page link')
    corpus = ' '.join(' '.join(row) for row in tables.get('Disclaimers', []))
    result = {
        'ticker': ticker, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'source_receipt': receipt_path.name, 'source_sha256': receipt['sha256'],
        'product_page_receipt': page_receipt_path.name, 'product_page_sha256': page_receipt['sha256'],
        'download_url_observed_in_product_page': receipt['requested_url'],
        'identity': {'fund_name': fund['name'], 'ticker': fund['alternateName'],
                     'page_identifiers_raw': sorted(set(identifiers)),
                     'holdings_header': tables['Holdings'][:6], 'key_facts': keyfacts},
        'format': 'Excel 2003 SpreadsheetML XML served as .xls',
        'parsing': {'raw_workbook_unmodified': True, 'bare_ampersands_escaped_in_memory_only': escapes,
                    'parser': 'Python standard-library ElementTree; no workbook export'},
        'sheets': {key: {'physical_rows': len(rows)} for key, rows in tables.items()},
        'historical_sheet': 'Historical', 'headers': headers,
        'source_data_rows': len(dates), 'date_start': min(dates), 'date_end': max(dates),
        'unique_dates': len(set(dates)), 'duplicate_dates': len(dates) - len(set(dates)),
        'strict_descending_dates': all(a > b for a, b in zip(dates, dates[1:])),
        'field_coverage': {key: {'positive_finite_rows': len(valid),
                               'date_start': min(valid) if valid else None,
                               'date_end': max(valid) if valid else None,
                               'missing_count': len(missing[key]), 'missing_dates': missing[key],
                               'nonpositive_count': len(nonpositive[key]), 'nonpositive_dates': nonpositive[key]}
                           for key, valid in field_valid_dates.items()},
        'source_grain': 'dated daily observations; trading-calendar completeness not yet audited',
        'daily_aum_column_present': any('asset' in key.lower() or 'aum' in key.lower() for key in headers),
        'historical_publication_timestamp': 'UNKNOWN: observation dates and retrieval UTC are present, historical release timestamps are not',
        'split_adjustment_status': 'UNKNOWN: separate stock-split history and historical shares/NAV adjustment convention not established',
        'current_vintage_not_point_in_time_archive': True,
        'terms_note': 'Raw workbook includes original BlackRock personal, non-commercial-use notice; retained in original bytes',
        'disclaimer_mentions_split': 'split' in corpus.lower(),
        'no_flows_features_returns_or_backtests_computed': True,
        'admission': 'SOURCE_FEASIBLE_PENDING_TIMING_SPLITS_AND_CALENDAR_AUDIT'
    }
    target = OUT / (ticker + '_INSPECTION.json')
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Raw workbook changed during inspection')
    return {key: result[key] for key in ('ticker', 'source_data_rows', 'date_start', 'date_end', 'field_coverage', 'daily_aum_column_present')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label', nargs='?')
    parser.add_argument('url', nargs='?')
    parser.add_argument('--inspect', choices=('IGV', 'IWM', 'MTUM', 'SOXX', 'TLT'))
    args = parser.parse_args()
    if args.inspect:
        if args.label or args.url:
            parser.error('Offline inspection cannot be combined with collection')
        result = inspect_saved(args.inspect)
    else:
        if not args.label or not args.url:
            parser.error('Collection requires a source label and URL')
        result = collect(args.label, args.url)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
