"""Freeze a bounded issuer-source collection; never reads prices or model data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'work/after157_annual_body_collect.py'
OUT = 'outputs/after157_annual_period_source_v1'
SCOPE = 'work/after157_annual_body_scope.json'
PINS = {
    'work/after157_annual_source_routes.json': '4c9c08eeed509336b6961f013e5b0ed231aeef17785c2ac2f8aecb2ac9ed98ca',
    'work/after157_annual_pair_source_feasibility.json': '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242',
    'work/after157_annual_original_body_inventory.json': '43c83311b2328e535303faed748ab7e385f962e61c7fb340216cea4d29056f44',
    'work/after157_annual_pair_source_feasibility_plan.json': '5750f95e17d8891f7781962efed7bd841408385cce1abfeee1a345827d3fb57d',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--collector-sha', required=True)
    p.add_argument('--review-sha', required=True)
    args = p.parse_args()
    pins = dict(PINS)
    pins[HELPER] = args.collector_sha
    pins['work/after157_annual_body_admission_review.json'] = args.review_sha
    for path, expected in pins.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Changed collection input: ' + path)
    metadata = json.loads((ROOT / 'work/after157_annual_pair_source_feasibility.json').read_text())
    events = list(metadata['selected_original_and_amendment_metadata'].values())
    targets = []

    def add(key, provider, url, filename, role, accession=None):
        item = dict(id=key, provider=provider, url=url, body_file=filename,
                    receipt_file=key + '.receipt.json', role=role)
        if accession:
            matches = [e for e in events if e['accessionNumber'] == accession]
            if len(matches) != 1:
                raise ValueError('Expected unique selected accession ' + accession)
            event = matches[0]
            item['expected_original'] = {k: event[k] for k in (
                'event_id', 'cik', 'accessionNumber', 'form', 'reportDate', 'filingDate', 'primaryDocument')}
        targets.append(item)

    add('msft2015_page', 'microsoft', 'https://www.microsoft.com/investor/reports/ar15/download-center/index.html',
        'msft2015_page.html', 'publisher_page')
    add('msft2016_page', 'microsoft', 'https://www.microsoft.com/investor/reports/ar16/download-center/',
        'msft2016_page.html', 'publisher_page')
    add('msft2015_10k', 'microsoft', 'https://www.microsoft.com/investor/reports/ar15/docs/MSFT_FY15Q4_10K.docx',
        'msft2015_10k.docx', 'original_annual_body_candidate', '0001193125-15-272806')
    add('msft2016_10k', 'microsoft', 'https://www.microsoft.com/investor/reports/ar16/docs/MSFT_FY16Q4_10K.docx',
        'msft2016_10k.docx', 'original_annual_body_candidate', '0001193125-16-662209')
    add('ko_annual_publications', 'coca_cola', 'https://investors.coca-colacompany.com/filings-reports/annual-publications',
        'ko_annual_publications.html', 'publisher_page')
    add('ko_annual_filings', 'coca_cola', 'https://investors.coca-colacompany.com/filings-reports/annual-filings-10-k',
        'ko_annual_filings.html', 'publisher_page')
    add('ko2016_10k', 'coca_cola', 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-17-000009/a2016123110-k.htm',
        'ko2016_10k.html', 'original_annual_body_candidate', '0000021344-17-000009')
    add('ko2017_10k', 'coca_cola', 'https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0000021344-18-000008/a2017123110-k.htm',
        'ko2017_10k.html', 'original_annual_body_candidate', '0000021344-18-000008')
    scope = dict(
        schema='after157_annual_body_collection_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        output_dir=OUT, helper_sha256=args.collector_sha, input_hashes=pins,
        max_bytes=30*1024*1024, timeout_seconds=35, targets=targets,
        source_scope='Four issuer original-annual body candidates and four official publisher pages. No SEC requests.',
        route_disclosures=[
            'Microsoft DOCX targets are the HTTPS form of exact HTTP paths exposed in official 10-K viewer links. No host or path substitution.',
            'KO2016 primary route is a candidate assembled from the official-host template and frozen accession/primaryDocument, not a previously verified direct page link.',
            'KO2017 primary route was independently found; web-reader size failure did not supply complete original bytes.',
        ],
        selection_basis='Source-route availability and pre-existing source-only pair coverage, never returns or new factor values.',
        stop_policy='No retries. Stop provider after authentication/rate/server/network failure; retain Retry-After and exact response evidence. Reject unapproved or SEC redirect before sending it.',
        admission_policy='Collection is not admission. Require original cover, issuer, original-accession/version binding, ordinary annual type and consecutive unchanged fiscal-period convention, under the pinned source-only gates and independent review.',
        unchanged='Original5978 slots, selected events, source chronology, conditional identity, information cutoffs and original evidence remain unchanged.',
        activity=dict(strategy_trials=0, feature_values=0, ranks=0, orders=0, returns=0, observer_writes=0),
    )
    path = ROOT / SCOPE
    with path.open('x') as f:
        json.dump(scope, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(scope=SCOPE, sha256=digest(path), targets=len(targets))))


if __name__ == '__main__':
    main()
