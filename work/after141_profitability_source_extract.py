"""Offline current-year statement extraction. It does not admit financial factors."""
import argparse
from bisect import bisect_right
from datetime import date, datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import after141_aapl_source_extract as apple
import after141_amd_source_extract as amd

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / 'outputs/monthly_profitability_extension_sources_v1'
PLAN_SHA = 'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'
INVENTORY = 'work/after141_profitability_coverage_options.json'
CALENDAR = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
ARTIFACTS = ['RAW_FACTS.json', 'SOURCE_CHECKS.json', 'VISUAL_REVIEW.json', 'MANIFEST.json']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(value):
    return ' '.join(value.split())


def dateline_norm(value):
    return re.sub(r'[\u2010-\u2015]', '-', norm(value))


def timestamp(value):
    require(isinstance(value, str), 'Missing retrieval timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.utcoffset() is not None, 'Retrieval timestamp must have timezone')
    return parsed


def local_path(relative):
    p = ROOT / relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Relative local source required')
    require(p.resolve().is_relative_to(ROOT) and not p.is_symlink(), 'Source escaped local root or is symlink')
    return p


def bound_read(relative, expected, bindings):
    raw = local_path(relative).read_bytes()
    require(sha(raw) == expected, 'SHA mismatch: ' + relative)
    bindings[relative] = expected
    return raw


def receipt_check(meta, raw, expected_url, kind):
    status = meta.get('status', meta.get('http_status'))
    require(type(status) is int and status == 200, 'Original receipt is not HTTP 200')
    require(meta.get('url', meta.get('source_url')) == expected_url and meta.get('final_url') == expected_url, 'Receipt URL mismatch')
    require(urlparse(expected_url).scheme == 'https' and urlparse(expected_url).hostname in ('www.apple.com', 'ir.amd.com'), 'Original issuer URL required')
    require(meta.get('public_unauthenticated') is True and not meta.get('error'), 'Original public receipt required')
    require(meta.get('response_sha256', meta.get('sha256')) == sha(raw), 'Receipt raw digest mismatch')
    size = meta.get('response_bytes', meta.get('size', meta.get('bytes')))
    require(type(size) is int and size == len(raw), 'Receipt byte count mismatch')
    timestamp(meta.get('retrieved_at'))
    content_type = meta.get('content_type', '').split(';')[0].lower()
    require(content_type == ('application/pdf' if kind == 'pdf' else 'text/html'), 'Receipt content type mismatch')
    if kind == 'pdf':
        require(raw.startswith(b'%PDF-') and meta.get('decoder') == 'identity', 'Identity PDF response required')
        require(meta.get('decoded_sha256') == sha(raw), 'Decoded identity bytes mismatch')
    return dict(http_status=status, url=expected_url, retrieved_at=meta['retrieved_at'], content_type=content_type,
                raw_sha256=sha(raw), original_receipt_verified=True)


def next_session(release, sessions):
    date.fromisoformat(release)
    require(bool(sessions) and sessions == sorted(set(sessions)), 'Calendar sessions must be sorted and unique')
    require(sessions[0] <= release < sessions[-1], 'Release outside pinned calendar coverage')
    return sessions[bisect_right(sessions, release)]


def jsonld_dates(value):
    if isinstance(value, dict):
        result = [value['datePublished'][:10]] if isinstance(value.get('datePublished'), str) else []
        return result + [d for child in value.values() for d in jsonld_dates(child)]
    if isinstance(value, list):
        return [d for child in value for d in jsonld_dates(child)]
    return []


def apple_publication(raw, spec, parent_url, indexed_releases):
    soup = BeautifulSoup(raw, 'html.parser')
    visible = [norm(x.get_text(' ', strip=True)) for x in soup.select('.category-eyebrow__date')]
    require(len(visible) == 1 and datetime.strptime(visible[0], '%B %d, %Y').date().isoformat() == spec['release_date'], 'Apple visible release date mismatch')
    published = [d for script in soup.select('script[type="application/ld+json"]') for d in jsonld_dates(json.loads(script.get_text()))]
    require(bool(published) and set(published) == {spec['release_date']}, 'Apple structured release date mismatch')
    headlines = [norm(x.get_text(' ', strip=True)) for x in soup.find_all('h1')]
    require(len(headlines) == 1 and 'fourth quarter' in headlines[0].lower(), 'Apple current Q4 release heading required')
    links = [urljoin(parent_url, x.get('href', '')) for x in soup.find_all('a', href=True)]
    require(spec['source_url'] in links, 'Apple release does not link this PDF')
    matches = [(acc, event) for acc, event in indexed_releases.items()
               if event.get('source_sha256') == spec['raw_sha256']
               and event.get('publication', {}).get('fiscal_quarter') == 4]
    require(len(matches) == 1, 'Unique Apple original-release index match required')
    acc, event = matches[0]
    require(event['publication']['release_date'] == spec['release_date'] and event['parsed']['statement']['current_end'] == spec['fiscal_year_end'], 'Apple indexed current period/date mismatch')
    require(event['parent_release_sha256'] == sha(raw), 'Apple parent/index digest mismatch')
    require(any(y['end'] == spec['fiscal_year_end'] and y['fiscal_year'] == spec['fiscal_year'] for y in event['publication']['year_ends']), 'Apple current fiscal identity mismatch')
    return dict(release_date=spec['release_date'], precision='date_only', publication_timezone='America/Los_Angeles',
                verification='visible issuer date plus NewsArticle datePublished; direct PDF outlink; indexed current annual end',
                visible_date=visible[0], headline=headlines[0], old_index_accession_reference_only=acc,
                old_SEC_revenue_admission_not_inherited=True)


def amd_publication(raw, spec, events):
    matches = [e for e in events if e.get('evidence_sha256') == spec['raw_sha256'] and e.get('fiscal_quarter') == 4]
    require(len(matches) == 1, 'Unique AMD annual release event required')
    event = matches[0]
    for key, value in [('ticker', 'AMD'), ('release_date', spec['release_date']), ('fiscal_year', spec['fiscal_year']),
                       ('fiscal_period_end_actual', spec['fiscal_year_end']), ('source_url', spec['source_url']), ('evidence_path', spec['raw_path'])]:
        require(event.get(key) == value, 'AMD indexed publication mismatch: ' + key)
    text = norm(BeautifulSoup(raw, 'html.parser').get_text(' ', strip=True))
    evidence = event.get('release_date_evidence', [])
    require(bool(evidence), 'AMD original dateline evidence missing')
    for item in evidence:
        require(item.get('sha256') == sha(raw) and item.get('evidence_path') == spec['raw_path'] and item.get('source_url') == spec['source_url'], 'AMD dateline binding mismatch')
        require(dateline_norm(item['excerpt']) in dateline_norm(text) and spec['release_date'] in amd.dates([item['excerpt']]), 'AMD original dateline absent or inconsistent')
    require(event.get('precision') == 'date_only' and event.get('publication_timezone') == 'America/Los_Angeles', 'AMD date convention mismatch')
    require('Advanced Micro Devices' in text or 'ADVANCED MICRO DEVICES' in text, 'AMD issuer name absent')
    return dict(release_date=spec['release_date'], precision='date_only', publication_timezone='America/Los_Angeles',
                verification='original issuer dateline and unique existing annual release event',
                dateline_evidence=[x['excerpt'] for x in evidence], dateline_match_normalization='whitespace and Unicode dash-to-hyphen only', original_event_first_public_time_verified=event.get('first_public_time_verified'))


def write_new(name, data):
    path = DIRECTORY / name
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return sha(path.read_bytes())


def run(write=False):
    if write:
        require(not any((DIRECTORY / n).exists() for n in ARTIFACTS), 'Final source artifacts already exist; do not overwrite')
    plan_path = DIRECTORY / 'EXTRACTION_PLAN.json'
    require(sha(plan_path.read_bytes()) == PLAN_SHA, 'Frozen extraction plan mismatch')
    plan = json.loads(plan_path.read_text())
    bindings = {'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json': PLAN_SHA}
    for path, expected in plan['input_hashes'].items():
        bound_read(path, expected, bindings)
    for path, expected in plan['pdf_runtime_hashes'].items():
        require(sha(Path(path).read_bytes()) == expected, 'PDF runtime changed: ' + path)
    inventory = json.loads(local_path(INVENTORY).read_text())
    indices = {}
    # These index hashes are transitively frozen by the inventory input. Verify
    # them explicitly; never derive a fresh identity or publication date silently.
    for option in inventory['options']:
        if option['symbol'] in ('AAPL', 'AMD'):
            item = option['index']
            indices[option['symbol']] = json.loads(bound_read(item['path'], item['sha256'], bindings))
    sessions = json.loads(local_path(CALENDAR).read_text())['sessions']
    for day in sessions:
        date.fromisoformat(day)
    rows, checks, visual_pages = [], [], []
    for spec in plan['sources']:
        raw = bound_read(spec['raw_path'], spec['raw_sha256'], bindings)
        meta = json.loads(local_path(spec['receipt']['path']).read_text())
        http = receipt_check(meta, raw, spec['source_url'], spec['format'])
        warnings = io.StringIO()
        handler = logging.StreamHandler(warnings)
        logging.getLogger('pypdf').addHandler(handler)
        try:
            if spec['symbol'] == 'AAPL':
                parent = local_path(spec['parent_release']['path']).read_bytes()
                pm = json.loads(local_path(spec['parent_receipt']['path']).read_text())
                parent_http = receipt_check(pm, parent, pm['source_url'], 'html')
                publication = apple_publication(parent, spec, pm['source_url'], indices['AAPL']['releases'])
                fact = apple.extract(raw, spec, plan['pdf_runtime']['bundle_site_packages'])
                artifacts = []
                for page in (1, 2):
                    for suffix in ('txt', 'png'):
                        rel = 'outputs/monthly_profitability_extension_sources_v1/pdf_artifacts/{}/page{}.{}'.format(spec['fiscal_year'], page, suffix)
                        digest = sha(local_path(rel).read_bytes())
                        artifacts.append({'path': rel, 'sha256': digest})
                        if suffix == 'txt':
                            require(digest == fact['reconstructed_page_text_sha256'][page-1], 'Rendered page text changed')
                    visual_pages.append(dict(fiscal_year=spec['fiscal_year'], page_1based=page, rendered_image=artifacts[-1],
                                             visual_check='Agent visually inspected current annual/date column, target row, scale, and consolidated original-statement heading; no image edits',
                                             fact_semantic_admission=False))
                fact['rendered_statement_artifacts'] = artifacts
            else:
                parent_http = None
                publication = amd_publication(raw, spec, indices['AMD'])
                fact = amd.extract(raw, spec)
        finally:
            logging.getLogger('pypdf').removeHandler(handler)
        fact['publication'] = publication
        fact['availability'] = dict(release_date=spec['release_date'], precision='date_only',
            publication_timezone='America/Los_Angeles', candidate_first_usable_session=next_session(spec['release_date'], sessions),
            rule='first pinned research-calendar session strictly after issuer-local date; no intraday timestamp inferred',
            first_public_version_verified=False, publication_original_text_checked=True, historical_web_snapshot_verified=False,
            calendar=CALENDAR, daily_official_calendar_certification=False)
        fact['provenance'] = dict(raw_path=spec['raw_path'], raw_sha256=spec['raw_sha256'], receipt=spec['receipt'],
            source_url=spec['source_url'], retrieved_at=http['retrieved_at'], parent_release=spec['parent_release'], parent_receipt=spec['parent_receipt'])
        fact['fact_admitted'] = False
        fact['strategy_admitted'] = False
        fact['identity_policy'] = 'Issuer-name/source event only. No continuous security identity or trading eligibility admitted.'
        rows.append(fact)
        checks.append(dict(symbol=spec['symbol'], fiscal_year=spec['fiscal_year'], http=http, parent_http=parent_http,
            current_annual_and_same_end_cells_extracted=True, publication_checked=True,
            parser_warnings=warnings.getvalue().splitlines(), fact_admitted=False, independent_review_required=True))
    require(len(rows) == len(plan['sources']) == 20, 'Every planned original release must be preserved')
    require(len({(x['symbol'], x['fiscal_year']) for x in rows}) == 20, 'Duplicate issuer-year candidate')
    summary = dict(planned_sources=20, extracted_current_annual_pairs=len(rows), source_errors=0,
        AAPL_years=sorted(x['fiscal_year'] for x in rows if x['symbol']=='AAPL'),
        AMD_years=sorted(x['fiscal_year'] for x in rows if x['symbol']=='AMD'), fact_admitted=0,
        factor_rank_order_return_computations=0, SEC_requests=0, network_requests=0, changes_to_141_or_central_state=False)
    if not write:
        return summary
    test_path = DIRECTORY / 'TESTS.json'
    tests = json.loads(test_path.read_text())
    require(tests.get('passed') is True and tests.get('synthetic_only') is True and tests.get('tests_run') == 70, 'Synthetic verification receipt required')
    for relative, expected in tests['source_hashes'].items():
        bound_read(relative, expected, bindings)
    bindings[str(test_path.relative_to(ROOT))] = sha(test_path.read_bytes())
    for path in sorted((ROOT/'work').glob('after141_*source*.py')):
        bindings[str(path.relative_to(ROOT))] = sha(path.read_bytes())
    for name in ['after141_aapl_pdf_probe.py', 'after141_profitability_coverage_options_build.py']:
        path = ROOT/'work'/name
        bindings[str(path.relative_to(ROOT))] = sha(path.read_bytes())
    raw_artifact = dict(schema_version=1, status='SOURCE_ONLY_RAW_CURRENT_ANNUAL_PAIRS_PENDING_INDEPENDENT_REVIEW',
        extraction_plan_sha256=PLAN_SHA, rows=rows, missing_events=[dict(symbol='AMD', fiscal_year=2019, reason='Original current-annual release not in reviewed local catalog; never filled from later comparative column')],
        history_completeness='not_certified; scoped existing cached releases only; unknown corrections and missing annual events remain possible',
        comparative_values_are_source_context_only=True, earlier_year_facts_never_created_from_comparatives=True, **summary)
    hashes = {}
    hashes['RAW_FACTS.json'] = write_new('RAW_FACTS.json', raw_artifact)
    hashes['SOURCE_CHECKS.json'] = write_new('SOURCE_CHECKS.json', dict(checks=checks, source_hashes=bindings,
        pdf_runtime_hashes=plan['pdf_runtime_hashes'], historical_first_version_certified=False, summary=summary))
    hashes['VISUAL_REVIEW.json'] = write_new('VISUAL_REVIEW.json', dict(pages=visual_pages, page_count=20,
        inspection_method='Agent viewed all 20 unmodified PDFium page images via view_image; pypdf numeric text independently matched PDFium same-row char geometry.',
        scope='Source cell readability/column placement only; no factor or identity admission', inspected_utc=datetime.now(timezone.utc).isoformat()))
    manifest = dict(schema_version=1, status='COMPLETE_SOURCE_PREPARATION_NOT_ADMISSION', created_utc=datetime.now(timezone.utc).isoformat(),
        extraction_plan_sha256=PLAN_SHA, output_hashes=hashes, source_hashes=bindings, pdf_runtime_hashes=plan['pdf_runtime_hashes'], summary=summary,
        limitations=['All current annual candidates require independent semantic, currency, period, publication and identity review.',
          'AAPL original monetary Gross margin and AMD FY2015-18 monetary Gross margin are not percentage summaries; financial semantic mapping still requires review.',
          'AMD FY2019 is missing and must remain missing. AMD FY2025 discontinued-operations comparability must be explicitly decided.',
          'Annual starts are derived only from prior fiscal end printed in the SAME original statement; no earlier-year monetary fact is backfilled.',
          'Dollar symbols and statement scales are preserved; no independent ISO-currency certification is inferred.',
          'Current 2026 retrieval vintage; historical first public versions and complete correction/event histories are not certified.',
          'Pinned calendar gives date-only next-session candidates, not a claim of daily official calendar certification.',
          'No factor, score, stock ranking, order, return, experiment registration or change to round141/central/observer state occurred.'])
    hashes['MANIFEST.json'] = write_new('MANIFEST.json', manifest)
    return dict(summary=summary, output_hashes=hashes)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write once only after source validation and independent review readiness')
    print(json.dumps(run(parser.parse_args().write), indent=2))
