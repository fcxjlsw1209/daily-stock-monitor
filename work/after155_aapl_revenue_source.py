"""Preregistered offline Apple annual net-sales cells; source evidence only.

The only imported research helper is the pure Apple PDF geometry/decimal module.
No source extraction occurs on import. Existing files and central state are never
written. Run fabricated tests, freeze, and then extract with the printed digest.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import after141_aapl_source_extract as pdf_helpers

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_aapl_revenue156'
SELF = 'work/after155_aapl_revenue_source.py'
TEST = 'work/after155_aapl_revenue_tests.py'
OLD_PLAN = 'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
OLD_PLAN_SHA = 'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
CALENDAR = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SOURCE_MAP = 'work/after155_multi_issuer_source_map.json'
PROFILE = {
    'issuer': 'Apple Inc.', 'symbol': 'AAPL', 'fiscal_years': list(range(2016, 2026)),
    'operations_page_1based': 1, 'balance_sheet_page_1based': 2,
    'exact_total_labels': ['Net sales', 'Total net sales'],
    'row_policy': 'Exactly one complete four-cell row across both total labels; section-only labels excluded. Duplicate/ambiguous totals held. Never sum products/services.',
    'columns': {'current_quarter': 0, 'prior_quarter': 1, 'current_annual': 2, 'comparative_annual': 3},
    'duration_literal': 'Twelve Months Ended', 'allowed_days': [364, 371],
    'annual_end': 'Final Saturday of September, matching declared original fiscal year.',
    'current_start': 'Day after prior fiscal end explicitly printed in the same original balance sheet, checked against operations comparative end.',
    'comparative_start': 'Earlier scoped ORIGINAL supplies its current-period boundary metadata only, and must precede current publication; otherwise explicit unknown. No value substitution.',
    'currency': 'Preserve literal dollar evidence; dollar_literal_only, ISO unset. No USD certification from the symbol or issuer nationality.',
    'scale': 'Explicit operations heading In millions; retain multiplier separately from exact displayed decimal.',
    'availability': 'First pinned calendar session strictly after current parent release date for BOTH cells and all contextual notes; date-only precision. Historical first-public bytes unverified.',
    'notes': 'Bounded keyword context from all PDF pages and parent article body. Keywords are review leads, not confirmed restatements. Absence does not establish accounting continuity.',
    'ambiguity': 'Preserve every planned year, explicit HELD with reason; never choose replacement sources or later comparative values.',
    'admission': 'No financial, identity, currency, accounting-chain, factor or strategy admission.',
}
NOTE_PATTERN = re.compile(r'restat\w*|recast\w*|retrospect\w*|reclassif\w*|revenue recognition|accounting (?:standard|principle|change)|(?:Topic|ASC)\s*606|adopt(?:ed|ion|ing)|(?:52|53)[ -]week', re.I)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def path(rel):
    p = Path(rel)
    require(not p.is_absolute() and '..' not in p.parts, 'Local relative path required')
    p = ROOT / p
    require(p.resolve() == p, 'Symlink/path redirection forbidden: ' + rel)
    return p


def binding(rel, expected=None):
    p = path(rel)
    require(p.is_file(), 'Missing local input: ' + rel)
    raw = p.read_bytes()
    if expected is not None:
        require(sha(raw) == expected, 'Expected digest mismatch: ' + rel)
    return {'path': rel, 'sha256': sha(raw), 'bytes': len(raw)}


def read_bound(b):
    raw = path(b['path']).read_bytes()
    require(sha(raw) == b['sha256'] and len(raw) == b['bytes'], 'Frozen input changed: ' + b['path'])
    return raw


def write_new(name, value):
    p = path(OUT + '/' + name)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with p.open('xb') as f:
        f.write(raw)
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(raw), 'bytes': len(raw)}


def first_session(release, sessions):
    date.fromisoformat(release)
    require(bool(sessions) and sessions == sorted(set(sessions)), 'Calendar must be sorted and unique')
    for day in sessions:
        date.fromisoformat(day)
    require(sessions[0] <= release < sessions[-1], 'Release outside pinned calendar')
    return sessions[bisect_right(sessions, release)]


def fiscal_endpoint(value):
    day = date.fromisoformat(value)
    require(day.month == 9 and 24 <= day.day <= 30 and day.weekday() == 5, 'Expected final September Saturday')
    return day


def period_from_boundaries(prior_end, end):
    prior, finish = fiscal_endpoint(prior_end), fiscal_endpoint(end)
    require(prior.year + 1 == finish.year, 'Nonconsecutive annual endpoints')
    days = (finish - prior).days
    require(days in PROFILE['allowed_days'], 'Annual period is not 52 or 53 weeks')
    return {'start': (prior + timedelta(days=1)).isoformat(), 'end': end,
            'days_inclusive': days, 'weeks': days // 7, 'prior_fiscal_end_explicit': prior_end,
            'start_is_direct_quote': False,
            'derivation': 'Day after prior fiscal end printed in same original balance sheet',
            'duration_literal': PROFILE['duration_literal']}


def statement_metadata(texts, spec):
    require(len(texts) >= 2, 'Operations and balance-sheet pages required')
    op, bs = (t.splitlines() for t in texts[:2])
    for text, title in zip(texts[:2], ['CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS', 'CONDENSED CONSOLIDATED BALANCE SHEETS']):
        require('Apple Inc.' in text and title in text and 'unaudited' in text.lower(), 'Original unaudited Apple statement identity missing')
    require(not re.search(r'NON[ -]?GAAP|RECONCILIATION', texts[0], re.I), 'Adjusted operations page excluded')
    head = re.sub(r'\s+', '', ''.join(op[:9]))
    require(head.count('ThreeMonthsEnded') == head.count('TwelveMonthsEnded') == 1 and head.index('ThreeMonthsEnded') < head.index('TwelveMonthsEnded'), 'Mixed annual/quarter duration header ambiguous')
    dates, line, date_text, years_text = pdf_helpers.headers(op, 4)
    bs_dates, bs_line, bs_date_text, bs_years_text = pdf_helpers.headers(bs, 2)
    require(dates[0] == dates[2] == bs_dates[0] == spec['fiscal_year_end'], 'Current period end mismatch')
    require(dates[1] == dates[3] == bs_dates[1], 'Prior operations/balance endpoint mismatch')
    require(date.fromisoformat(dates[2]).year == spec['fiscal_year'], 'Declared fiscal year mismatch')
    require(dates[2] < spec['release_date'], 'Release does not follow period end')
    current = period_from_boundaries(bs_dates[1], bs_dates[0])
    current['boundary_locator'] = {'page_1based': 2, 'line_1based': bs_line,
        'date_header_lines': [bs_date_text, bs_years_text], 'header_dates': bs_dates}
    return {'current_period': current, 'comparative_period': {'start': None, 'end': dates[3],
        'days_inclusive': None, 'weeks': None, 'duration_literal': PROFILE['duration_literal'],
        'reason': 'Earlier scoped original required for comparative start; never assume 365 days'},
        'header_dates': dates, 'header_line_1based': line,
        'date_header_lines': [date_text, years_text]}


def revenue_row(lines):
    matches = []
    for label in PROFILE['exact_total_labels']:
        pattern = re.compile(r'\s*' + r'\s*'.join(re.escape(c) for c in label if not c.isspace()) + r'\s+(.+)', re.I)
        for i, line in enumerate(lines):
            hit = pattern.fullmatch(line)
            if hit:
                cells = hit[1].replace('/dollarsign', '$').replace('$', '').split()
                require(len(cells) == 4, 'Total-sales row does not contain four financial cells')
                for cell in cells:
                    pdf_helpers.decimal_cell(cell)
                matches.append((label, cells, i + 1, line))
    require(len(matches) == 1, 'Unique exact total-sales row required')
    return matches[0]


def parse_revenue(texts, spec):
    meta = statement_metadata(texts, spec)
    lines = texts[0].splitlines()
    unit_lines = [(i + 1, line) for i, line in enumerate(lines[:9]) if re.search(r'\(In millions,', line)]
    require(len(unit_lines) == 1, 'Explicit unique operations In millions heading required')
    label, cells, rowline, rowtext = revenue_row(lines)
    dollar = '$' if '$' in rowtext else '/dollarsign' if '/dollarsign' in rowtext else None
    require(dollar is not None, 'Total-sales row has no explicit dollar glyph')
    revenues = {}
    for role, column in [('current', 2), ('comparative', 3)]:
        period = meta[role + '_period']
        revenues[role] = {'measure': 'total_net_sales', 'measure_label': label,
            'displayed_amount': cells[column], 'amount_decimal': pdf_helpers.decimal_cell(cells[column]),
            'scale': 1000000, 'unit_literal': unit_lines[0][1],
            'unit_locator': {'page_1based': 1, 'line_1based': unit_lines[0][0]},
            'currency_literal': dollar, 'currency_evidence_class': 'dollar_literal_only',
            'iso_currency': None, 'ISO_currency_verified': False, 'period': period,
            'date': period['end'], 'header_dates': meta['header_dates'],
            'date_header_lines': meta['date_header_lines'],
            'all_row_cells': cells, 'row_text': rowtext, 'row_text_sha256': sha(rowtext.encode()),
            'locator': {'page_1based': 1, 'line_1based': rowline, 'financial_column_index_0based': column},
            'vintage_role': 'own_year_original' if role == 'current' else 'same_current_release_prior_year_comparative',
            'historical_original_event_created': role == 'current',
            'fact_admitted': False, 'strategy_admitted': False}
    return {'metadata': meta, 'revenue': revenues}


def receipt_check(meta, raw, url, kind):
    require(meta.get('status', meta.get('http_status')) == 200, 'Receipt is not HTTP200')
    require(urlparse(url).scheme == 'https' and urlparse(url).hostname == 'www.apple.com', 'Original Apple HTTPS URL required')
    require(meta.get('url', meta.get('source_url')) == url and meta.get('final_url') == url, 'Receipt URL mismatch')
    require(meta.get('public_unauthenticated') is True and not meta.get('error'), 'Public source receipt required')
    require(meta.get('response_sha256', meta.get('sha256')) == sha(raw), 'Receipt digest mismatch')
    require(meta.get('response_bytes', meta.get('size')) == len(raw), 'Receipt byte count mismatch')
    stamp = datetime.fromisoformat(meta['retrieved_at'].replace('Z', '+00:00'))
    require(stamp.utcoffset() is not None, 'Retrieval timezone missing')
    require(meta.get('content_type', '').split(';')[0].lower() == ('application/pdf' if kind == 'pdf' else 'text/html'), 'Receipt MIME mismatch')
    if kind == 'pdf':
        require(raw.startswith(b'%PDF-') and meta.get('decoder') == 'identity' and meta.get('decoded_sha256') == sha(raw), 'Identity original PDF bytes required')
    return {'verified': True, 'url': url, 'retrieved_at': meta['retrieved_at'], 'sha256': sha(raw)}


def jsonld_dates(value):
    if isinstance(value, dict):
        dates = [value['datePublished'][:10]] if isinstance(value.get('datePublished'), str) else []
        return dates + [d for v in value.values() for d in jsonld_dates(v)]
    if isinstance(value, list):
        return [d for v in value for d in jsonld_dates(v)]
    return []


def publication(raw, spec, parent_url):
    soup = BeautifulSoup(raw, 'html.parser')
    visible = [pdf_helpers.norm(x.get_text(' ', strip=True)) for x in soup.select('.category-eyebrow__date')]
    require(len(visible) == 1 and datetime.strptime(visible[0], '%B %d, %Y').date().isoformat() == spec['release_date'], 'Visible release date mismatch')
    dates = [d for script in soup.select('script[type="application/ld+json"]') for d in jsonld_dates(json.loads(script.get_text()))]
    require(bool(dates) and set(dates) == {spec['release_date']}, 'Structured release date mismatch')
    headlines = [pdf_helpers.norm(x.get_text(' ', strip=True)) for x in soup.find_all('h1')]
    require(len(headlines) == 1 and 'fourth quarter' in headlines[0].lower(), 'Original fourth-quarter release title required')
    require(spec['source_url'] in [urljoin(parent_url, x['href']) for x in soup.find_all('a', href=True)], 'Direct original PDF link missing')
    return {'release_date': spec['release_date'], 'precision': 'date_only',
        'visible_date': visible[0], 'headline': headlines[0], 'structured_dates': dates,
        'direct_pdf_link_verified': True, 'publication_timezone': 'America/Los_Angeles',
        'first_public_version_verified': False, 'historical_web_snapshot_verified': False}


def note_contexts(text, source, availability, page=None):
    text = pdf_helpers.norm(text)
    out = []
    for match in NOTE_PATTERN.finditer(text):
        start, stop = max(0, match.start() - 180), min(len(text), match.end() + 220)
        excerpt = text[start:stop]
        out.append({'keyword': match.group(), 'excerpt': excerpt, 'excerpt_sha256': sha(excerpt.encode()),
                    'normalized_character_start': start, 'normalized_character_end': stop,
                    'page_1based': page, 'source': deepcopy(source),
                    'known_at_session': availability, 'time_precision': 'current_release_date_only_next_session',
                    'confirmed_accounting_change': False, 'requires_review': True})
    return out


def attach_prior_duration(row, earlier):
    prior = row['revenue']['comparative']['period']
    if earlier is None:
        return
    require(earlier['fiscal_year'] + 1 == row['fiscal_year'], 'Earlier original year mismatch')
    require(earlier['publication']['release_date'] < row['publication']['release_date'], 'Future original metadata forbidden')
    source_period = earlier['revenue']['current']['period']
    require(source_period['end'] == prior['end'], 'Comparative end does not match earlier original')
    for key in ['start', 'days_inclusive', 'weeks', 'prior_fiscal_end_explicit', 'start_is_direct_quote', 'boundary_locator']:
        prior[key] = deepcopy(source_period[key])
    prior['derivation'] = 'Earlier scoped ORIGINAL supplies current-period boundary metadata only; comparative monetary value remains in current release'
    prior['duration_source'] = deepcopy(earlier['source'])
    prior['duration_source_release_date'] = earlier['publication']['release_date']
    prior.pop('reason', None)


def freeze():
    old_binding = binding(OLD_PLAN, OLD_PLAN_SHA)
    old = json.loads(read_bound(old_binding))
    test_binding = binding(OUT + '/TESTS.json')
    tests = json.loads(read_bound(test_binding))
    require(tests['passed'] and tests['synthetic_only'], 'Passing fabricated tests required before freeze')
    inputs = {rel: binding(rel) for rel in [SELF, TEST, SOURCE_MAP, CALENDAR,
        'work/after141_aapl_source_extract.py', 'run_local.sh', 'local_run.py', 'local.sandbox.sb']}
    inputs[OLD_PLAN] = old_binding
    inputs[test_binding['path']] = test_binding
    require(tests['source_sha256'] == inputs[SELF]['sha256'] and tests['tests_sha256'] == inputs[TEST]['sha256'], 'Tests are stale')
    sources = []
    for item in old['sources']:
        if item['symbol'] != 'AAPL':
            continue
        spec = {k: item[k] for k in ['symbol', 'fiscal_year', 'fiscal_year_end', 'release_date', 'source_url']}
        spec['raw'] = binding(item['raw_path'], item['raw_sha256'])
        for key in ['receipt', 'parent_release', 'parent_receipt']:
            spec[key] = binding(item[key]['path'], item[key]['sha256'])
        sources.append(spec)
    require([x['fiscal_year'] for x in sources] == PROFILE['fiscal_years'], 'Planned original-year scope mismatch')
    for p, digest in old['pdf_runtime_hashes'].items():
        require(sha(Path(p).read_bytes()) == digest, 'PDF runtime digest mismatch')
    plan = {'created_utc': now(), 'status': 'FROZEN_SOURCE_ONLY_PROFILE', 'profile': PROFILE,
        'inputs': inputs, 'sources': sources, 'pdf_runtime': old['pdf_runtime'],
        'pdf_runtime_hashes': old['pdf_runtime_hashes'],
        'freeze_reads': 'Metadata and raw-byte hashes only; no PDF or financial body parsed in this phase.',
        'known_exclusions': [{'fiscal_year': 2015, 'status': 'NO_SCOPED_OWN_YEAR_ORIGINAL',
            'reason': 'FY2016 comparative is not an FY2015 original. Earlier start metadata remains unknown.'}],
        'new_trials': 0, 'observer_writes': 0, 'financial_computations': 0}
    print(json.dumps(write_new('PLAN.json', plan)))


def verify_plan(plan):
    require(plan['profile'] == PROFILE, 'Profile changed after freeze')
    for b in plan['inputs'].values():
        read_bound(b)
    for spec in plan['sources']:
        for key in ['raw', 'receipt', 'parent_release', 'parent_receipt']:
            read_bound(spec[key])
    for p, digest in plan['pdf_runtime_hashes'].items():
        require(sha(Path(p).read_bytes()) == digest, 'Frozen PDF runtime changed')


def extract(plan_sha):
    raw_plan = path(OUT + '/PLAN.json').read_bytes()
    require(sha(raw_plan) == plan_sha, 'PLAN digest mismatch')
    plan = json.loads(raw_plan)
    verify_plan(plan)
    for name in ['DATASET.json', 'SOURCE_CHECKS.json', 'RECEIPT.json']:
        require(not path(OUT + '/' + name).exists(), 'Write-once extraction already exists')
    sys.path.append(plan['pdf_runtime']['bundle_site_packages'])
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    sessions = json.loads(read_bound(plan['inputs'][CALENDAR]))['sessions']
    rows, checks = [], []
    for spec in plan['sources']:
        try:
            raw = read_bound(spec['raw'])
            parent = read_bound(spec['parent_release'])
            pm = json.loads(read_bound(spec['parent_receipt']))
            pdf_http = receipt_check(json.loads(read_bound(spec['receipt'])), raw, spec['source_url'], 'pdf')
            parent_http = receipt_check(pm, parent, pm['source_url'], 'html')
            pub = publication(parent, spec, pm['source_url'])
            available = first_session(spec['release_date'], sessions)
            reader = PdfReader(io.BytesIO(raw))
            texts = [page.extract_text(extraction_mode='layout') for page in reader.pages]
            row = parse_revenue(texts, spec)
            doc = pdfium.PdfDocument(raw)
            try:
                page = doc[0]
                try:
                    tp = page.get_textpage()
                    try:
                        for cell in row['revenue'].values():
                            pdf_helpers.attach_geometry(cell, tp)
                    finally:
                        tp.close()
                finally:
                    page.close()
            finally:
                doc.close()
            row.update(symbol='AAPL', fiscal_year=spec['fiscal_year'], status='EXTRACTED_SOURCE_CANDIDATE',
                publication=pub, source=spec['raw'], source_url=spec['source_url'],
                receipt=spec['receipt'], parent_release=spec['parent_release'], parent_receipt=spec['parent_receipt'],
                retrieved_at=pdf_http['retrieved_at'], page_count=len(texts),
                page_text_sha256=[sha(t.encode()) for t in texts],
                availability={'candidate_first_usable_session': available, 'calendar': plan['inputs'][CALENDAR],
                    'rule': PROFILE['availability'], 'daily_official_calendar_certification': False},
                fact_admitted=False, strategy_admitted=False, accounting_chain_admitted=False,
                identity_status='issuer_name_and_original_host_verified_only')
            for role, cell in row['revenue'].items():
                cell['known_at_session'] = available
                cell['release_date'] = spec['release_date']
                cell['source'] = deepcopy(spec['raw'])
            notes = [n for i, text in enumerate(texts, 1) for n in note_contexts(text, spec['raw'], available, i)]
            soup = BeautifulSoup(parent, 'html.parser')
            bodies = soup.select('.pagebody')
            article = bodies[0] if len(bodies) == 1 else soup.find('article')
            require(article is not None, 'Unambiguous parent article body unavailable for note review')
            notes += note_contexts(article.get_text(' ', strip=True), spec['parent_release'], available)
            row['accounting_note_leads'] = notes
            row['accounting_note_review_complete'] = False
            row['holds'] = ['currency_iso_unverified', 'first_public_version_unverified', 'accounting_continuity_review_pending', 'independent_revenue_source_review_pending']
            rows.append(row)
            checks.append({'fiscal_year': spec['fiscal_year'], 'passed': True, 'pdf_http': pdf_http,
                'parent_http': parent_http, 'both_cells_cross_engine_match': True})
        except (ValueError, KeyError, TypeError, IndexError) as e:
            rows.append({'symbol': 'AAPL', 'fiscal_year': spec['fiscal_year'], 'status': 'HELD',
                'source': spec['raw'], 'publication': {'release_date': spec['release_date']},
                'revenue': None, 'holds': [str(e)], 'fact_admitted': False, 'strategy_admitted': False})
            checks.append({'fiscal_year': spec['fiscal_year'], 'passed': False, 'error': str(e)})
    originals = {r['fiscal_year']: r for r in rows if r['status'] == 'EXTRACTED_SOURCE_CANDIDATE'}
    for row in originals.values():
        attach_prior_duration(row, originals.get(row['fiscal_year'] - 1))
        if row['revenue']['comparative']['period']['start'] is None:
            row['holds'].append('comparative_period_start_unknown')
    verify_plan(plan)
    data = {'status': 'SOURCE_CANDIDATES_ONLY_NOT_ADMITTED', 'plan_sha256': plan_sha,
        'rows': rows, 'known_exclusions': plan['known_exclusions'],
        'limits': list(PROFILE[k] for k in ['currency', 'availability', 'notes', 'admission']),
        'financial_ratios_factors_orders_returns': 0, 'new_strategy_trials': 0, 'observer_writes': 0}
    dataset = write_new('DATASET.json', data)
    checked = write_new('SOURCE_CHECKS.json', {'plan_sha256': plan_sha, 'rows': checks})
    receipt = {'created_utc': now(), 'plan_sha256': plan_sha, 'dataset': dataset, 'source_checks': checked,
        'original_reports_planned': len(rows), 'original_reports_extracted': len(originals),
        'monetary_cells_extracted': 2 * len(originals), 'held_reports': len(rows) - len(originals),
        'comparative_period_starts_unknown': [r['fiscal_year'] for r in originals.values() if r['revenue']['comparative']['period']['start'] is None],
        'current_53_week_fiscal_years': [r['fiscal_year'] for r in originals.values() if r['revenue']['current']['period']['weeks'] == 53],
        'currency_iso_certifications': 0, 'fact_admissions': 0, 'new_strategy_trials': 0,
        'network_requests': 0, 'observer_writes': 0, 'all_bound_inputs_unchanged': True,
        'tests': plan['inputs'][OUT + '/TESTS.json']}
    result = write_new('RECEIPT.json', receipt)
    print(json.dumps({'receipt': result, 'counts': {k: v for k, v in receipt.items() if isinstance(v, (int, bool, list))}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--freeze', action='store_true')
    actions.add_argument('--extract', metavar='PLAN_SHA256')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        extract(args.extract)
