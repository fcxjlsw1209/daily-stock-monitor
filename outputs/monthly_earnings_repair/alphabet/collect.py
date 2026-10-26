"""Verify quarterly Alphabet/Google release dates from saved official PDFs.

The Google Cloud sample corpus is an explicitly labelled official archive copy,
not an assertion that the current IR URL is accessible or files never changed.
"""
import argparse
import calendar
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
import sys

from fetch import HERE, ROOT, RAW, dump, fetch, sha

AS_OF = '2026-09-24'
INDEX = RAW / 'd49aa72278fb0faf05a4.html'
ARCHIVE_DOCUMENTATION = 'https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/web-app/README.md'
ARCHIVE_DOC_PATH = RAW / '9f5808d44c12ebd604eb.html'
PDF_PYTHON = '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
QUARTERS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DATE = MONTH + r'\s+\d{1,2},?\s+\d{4}'


def date_value(raw):
    return dt.datetime.strptime(re.sub(r'\s+', ' ', raw).replace(',', ''), '%B %d %Y').date().isoformat()


def extract_pdf(path):
    text_path = path.with_suffix('.txt')
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = '\n\f\n'.join(p.extract_text() or '' for p in reader.pages)
    except ImportError:
        code = "from pypdf import PdfReader;import sys;print('\\n\\f\\n'.join(p.extract_text() or '' for p in PdfReader(sys.argv[1]).pages))"
        text = subprocess.run([PDF_PYTHON, '-c', code, str(path)], check=True, capture_output=True, text=True).stdout
    text_path.write_text(text)
    return text


def parse_text(text, target, source_url, evidence_path, evidence_sha256):
    s = re.sub(r'\s+', ' ', text).strip()
    headline = re.search(r'(?:Alphabet|Google(?:\s+Inc\.?)?)\s+Announces\s+(First|Second|Third|Fourth)\s+Quarter(?:\s+and\s+(?:Fiscal\s+)?(?:Year\s+End|Fiscal\s+Year))?\s+(\d{4})\s+(?:Financial\s+)?Results', s, re.IGNORECASE)
    if not headline:
        # Older releases say "Fourth Quarter and Fiscal Year 2011 Results".
        headline = re.search(r'(?:Alphabet|Google(?:\s+Inc\.?)?)\s+Announces\s+(First|Second|Third|Fourth)\s+Quarter(?:\s+and\s+Fiscal\s+Year)?\s+(\d{4})\s+(?:Financial\s+)?Results', s, re.IGNORECASE)
    if not headline:
        raise ValueError('Earnings headline and fiscal quarter/year not verified')
    q, y = QUARTERS[headline.group(1).lower()], int(headline.group(2))
    if (y, q) != (target['fiscal_year'], target['fiscal_quarter']):
        raise ValueError('PDF headline differs from requested fiscal quarter')
    # The archived FY2011Q4 PDF maps its dateline dash glyph to +/- in extracted
    # text; the date itself remains legible and is checked against the PDF page.
    d = re.search(r'MOUNTAIN\s+VIEW\s*,\s*(?:Calif\.|California|CA)\s*[–—\-:±]?\s*(' + DATE + ')', s, re.IGNORECASE)
    if not d:
        raise ValueError('Official Mountain View dateline with actual release date absent')
    released = date_value(d.group(1))
    if released > AS_OF:
        raise ValueError('Release after fixed research cutoff')
    first_paragraph = s[d.start():d.start() + 900]
    period = re.search(r'(?:quarter(?:\s+and\s+(?:the\s+)?(?:fiscal\s+)?(?:year|full\s+year))?|fourth\s+quarter)\s+ended\s+(' + DATE + ')', first_paragraph, re.IGNORECASE)
    if not period:
        raise ValueError('Actual quarter end missing from first release paragraph')
    actual = date_value(period.group(1))
    expected = '%04d-%02d-%02d' % (y, q * 3, calendar.monthrange(y, q * 3)[1])
    if actual != expected or actual >= released:
        raise ValueError('Actual fiscal period does not match headline quarter or release ordering')
    # Identify the original announcing entity from the opening release sentence,
    # not from the hosting URL, headline capitalization or later subsidiary text.
    # Historical PDF extraction also produces soft hyphens and +/- dash glyphs.
    issuer_match = re.match(r'\s*[–—\-:±\u00ad]*\s*(Google|Alphabet)\s+Inc\.?', s[d.end():], re.IGNORECASE)
    if not issuer_match:
        raise ValueError('Original announcing issuer absent immediately after dateline')
    issuer = {'google': 'Google Inc.', 'alphabet': 'Alphabet Inc.'}[issuer_match.group(1).lower()]
    headline_issuer = re.match(r'(Google|Alphabet)\b', headline.group(0), re.IGNORECASE)
    if headline_issuer.group(1).lower() != issuer_match.group(1).lower():
        raise ValueError('Original announcing issuer disagrees with release headline')
    archive = 'storage.googleapis.com/cloud-samples-data/' in source_url
    return {
        'ticker': 'GOOGL', 'issuer_name_at_release': issuer,
        'issuer_identity_note': 'GOOGL is the current research issuer key; Google predecessor and Alphabet successor are distinguished by the announcing entity immediately after the original release dateline, with a case-insensitive headline cross-check. This is not a claim that this ticker existed throughout the historical period.',
        'issuer_identity_evidence': s[d.start():d.end() + issuer_match.end()],
        'release_date': released, 'publication_timestamp': None, 'precision': 'date_only',
        'publication_timezone': 'America/Los_Angeles',
        'publication_timezone_basis': 'Explicit conservative issuer-local calendar-date research policy from the original Mountain View, California dateline, not an observed CMS timezone or exact publication time.',
        'publication_timezone_evidence': {'kind': 'issuer_local_calendar_date_policy', 'observed_dateline': d.group(0),
            'evidence_path': evidence_path, 'cms_timezone_observed': False, 'exact_publication_time_inferred': False},
        'first_public_time_verified': False, 'fiscal_period_end_actual': actual, 'fiscal_year': y, 'fiscal_quarter': q,
        'source_url': source_url, 'evidence_path': evidence_path, 'evidence_sha256': evidence_sha256,
        'evidence_excerpt': first_paragraph[:period.end()], 'headline': headline.group(0),
        'source_class': 'issuer_official_google_cloud_archive_copy' if archive else 'issuer_official_ir_cdn_release',
        'archive_provenance_url': ARCHIVE_DOCUMENTATION if archive else None,
        'archive_provenance_evidence_path': str(ARCHIVE_DOC_PATH.relative_to(ROOT)) if archive else None,
        'archive_provenance_sha256': sha(ARCHIVE_DOC_PATH.read_bytes()) if archive else None,
        'vendor_fiscal_period_end_label': None, 'vendor_reported_date': None, 'vendor_report_time_raw': None,
        'timestamp_note': 'Only the dated earnings-release dateline is verified. PDF creation/modification metadata and conference-call times are not publication timestamps.',
        'in_study_window': released >= '2016-01-01', 'research_cutoff': AS_OF,
    }


def targets():
    items = json.loads(INDEX.read_text())['items']
    result = {}
    for item in items:
        name = item['name']; tail = name.rsplit('/', 1)[-1]
        m = re.match(r'(20\d\d)Q([1-4])_.*earnings.*\.pdf$', tail)
        if m and 2011 <= int(m.group(1)) <= 2022 and 'Transcript' not in tail:
            y, q = int(m.group(1)), int(m.group(2))
            result[(y, q)] = dict(fiscal_year=y, fiscal_quarter=q, source_url='https://storage.googleapis.com/cloud-samples-data/' + name)
    for y in range(2023, 2027):
        for q in range(1, 5 if y < 2026 else 3):
            result[(y, q)] = dict(fiscal_year=y, fiscal_quarter=q,
                source_url='https://s206.q4cdn.com/479360582/files/doc_financials/%d/q%d/%dq%d-alphabet-earnings-release.pdf' % (y, q, y, q))
    result[(2024, 1)]['source_url'] = 'https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/2024q1-alphabet-earnings-release-pdf.pdf'
    return [result[k] for k in sorted(result)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only'); args = ap.parse_args()
    events_path = HERE / 'events.json'; manifest_path = HERE / 'manifest.json'
    existing = json.loads(events_path.read_text()) if events_path.exists() else []
    events = {(x['fiscal_year'], x['fiscal_quarter']): x for x in existing}
    old_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = {x['source_url']: x for x in old_manifest}
    all_targets = targets(); dump(HERE / 'TARGETS.json', all_targets)
    for target in all_targets:
        y, q = target['fiscal_year'], target['fiscal_quarter']
        if args.only and '%dQ%d' % (y, q) not in args.only.split(','):
            continue
        path, meta = fetch(target['source_url'])
        row = dict(meta, fiscal_year=y, fiscal_quarter=q, evidence_path=str(path.relative_to(ROOT)))
        try:
            if meta['http_status'] != 200 or not path.read_bytes().startswith(b'%PDF-'):
                raise ValueError('Source not a successful original PDF response')
            text = extract_pdf(path)
            e = parse_text(text, target, meta['final_url'], row['evidence_path'], meta['sha256'])
            events[(y, q)] = e; row['verified'] = True
        except (ValueError, subprocess.CalledProcessError) as err:
            row['verified'] = False; row['error'] = str(err)
        manifest[target['source_url']] = row
        dump(events_path, [events[k] for k in sorted(events)])
        dump(manifest_path, list(manifest.values()))
        dump(HERE / 'PENDING.json', [t for t in all_targets if (t['fiscal_year'], t['fiscal_quarter']) not in events])
        print(y, q, 'verified' if row['verified'] else row.get('error'), flush=True)
    print('VERIFIED', len(events), 'EXPECTED', len(all_targets), flush=True)


if __name__ == '__main__':
    main()
