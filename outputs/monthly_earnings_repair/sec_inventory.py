"""Offline SEC 8-K 2.02 leads; never relabel filings as earnings release times.

This inventory makes the next original-exhibit review targeted. 2.02 also
includes updates/preliminary results, so none are admitted as quarterly releases
by this module. Current ticker metadata locates an issuer, not historical prices.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re

P = Path(__file__).resolve().parent
O = P.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    p = path.with_suffix('.provenance.json')
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    if not str(m.get('source', '')).startswith('https://data.sec.gov/submissions/'):
        return None
    if m.get('sha256') != sha(path):
        raise ValueError('SEC source checksum mismatch: ' + str(path))
    return dict(path=str(path), sha256=m['sha256'], source_url=m['source'],
                retrieved_at=m.get('retrieved_at'))


def main():
    monthly = json.loads((O / 'monthly_architecture_restart/META.json').read_text())['monthly']
    wanted = sorted(set(sum(monthly['symbols'], [])) - set(monthly['etfs']))
    # Explicit provider naming aliases only; no merger/successor stitching.
    aliases = {'research_equity_FB': 'META', 'BRK.B': 'BRK-B'}
    lookup = {aliases.get(s, s): s for s in wanted}
    all_files = list((O / 'dynamic_top100').rglob('*submissions.json'))
    continuations = defaultdict(list)
    for path in (O / 'dynamic_top100').rglob('CIK*-submissions-*.json'):
        if not path.name.endswith('.provenance.json'):
            continuations[path.name].append(path)
    entries, events, sources = [], {}, {}
    matched = defaultdict(list)
    for path in all_files:
        try:
            obj = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        for ticker in obj.get('tickers', []):
            if ticker in lookup:
                matched[lookup[ticker]].append((path, obj))
    for ticker in wanted:
        matches = matched[ticker]
        issuer_records = []; missing_pages = set(); issuer_ids = set()
        for path, obj in matches:
            provenance = source(path)
            if not provenance:
                continue
            cik = str(obj['cik']).zfill(10); issuer_ids.add(cik)
            pages = [(path, obj.get('filings', {}).get('recent', {}), provenance)]
            for page in obj.get('filings', {}).get('files', []):
                if page.get('filingTo', '') < '2015-01-01':
                    continue
                found = False
                for f in continuations.get(page['name'], []):
                    prov = source(f)
                    if prov:
                        pages.append((f, json.loads(f.read_text()), prov)); found = True
                if not found:
                    missing_pages.add(page['name'])
            for f, rows, prov in pages:
                sources[str(f)] = prov
                lengths = {len(rows[k]) for k in ['form', 'items', 'filingDate', 'accessionNumber'] if k in rows}
                if len(lengths) != 1 or not all(k in rows for k in ['form', 'items', 'filingDate', 'accessionNumber']):
                    raise ValueError('SEC column length/schema mismatch: ' + str(f))
                if any(k in rows and len(rows[k]) != len(rows['form'])
                       for k in ['primaryDocument', 'reportDate', 'acceptanceDateTime']):
                    raise ValueError('SEC optional column length/schema mismatch: ' + str(f))
                for i, form in enumerate(rows['form']):
                    if form not in ('8-K', '8-K/A') or '2.02' not in re.split(r'[,;\s]+', rows['items'][i]):
                        continue
                    day = rows['filingDate'][i]
                    if not '2015-01-01' <= day <= '2026-10-12':
                        continue
                    accession = rows['accessionNumber'][i]
                    doc = rows.get('primaryDocument', [''] * len(rows['form']))[i]
                    rec = dict(symbol=ticker, cik=cik, accession=accession, form=form,
                        filing_date=day, report_date_raw=rows.get('reportDate', [None] * len(rows['form']))[i],
                        acceptance_timestamp_raw=rows.get('acceptanceDateTime', [None] * len(rows['form']))[i],
                        release_date=None, exact_release_timestamp=None, eligible_event=False,
                        reason='Review original 2.02/exhibit to distinguish actual quarterly release, preliminary results, amendments and event date.',
                        timestamp_note='SEC acceptance is not original release; raw API clock not independently verified.',
                        document_url='https://www.sec.gov/Archives/edgar/data/%d/%s/%s' % (int(cik), accession.replace('-', ''), doc),
                        evidence_path=str(f), evidence_sha256=prov['sha256'])
                    key = (cik, accession)
                    associated = set(events.get(key, {}).get('associated_symbols', [])) | {ticker}
                    rec['associated_symbols'] = sorted(associated)
                    events[key] = rec
                    issuer_records.append(rec)
        unique = {r['accession']: r for r in issuer_records}
        entries.append(dict(symbol=ticker, issuer_ciks=sorted(issuer_ids),
            cached_2_02_candidates=len(unique), missing_continuations=sorted(missing_pages),
            earliest_filing=min((r['filing_date'] for r in unique.values()), default=None),
            latest_filing=max((r['filing_date'] for r in unique.values()), default=None),
            quarterly_release_history_complete=False,
            note='No automatic inclusion from filing index alone; foreign 6-K earnings need text review.' if matches else 'No matching cached issuer metadata.'))
    output = dict(status='offline_review_inventory_not_admitted_events', network_requests=0,
        symbols=len(wanted), symbols_with_candidates=sum(x['cached_2_02_candidates'] > 0 for x in entries),
        issuers_with_candidates=len(set(e['cik'] for e in events.values())),
        candidate_filings=len(events), coverage=entries,
        candidates=sorted(events.values(), key=lambda r: (r['symbol'], r['filing_date'], r['accession'])),
        source_checksums=list(sources.values()))
    (P / 'SEC_REVIEW_QUEUE.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ['coverage', 'candidates', 'source_checksums']}))


if __name__ == '__main__':
    main()
