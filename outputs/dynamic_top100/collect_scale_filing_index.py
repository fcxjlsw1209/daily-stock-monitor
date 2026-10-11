"""Locate original filings for suspicious share-unit observations."""
from pathlib import Path
import datetime, hashlib, json, time
import requests

P = Path(__file__).resolve().parent
D = P / 'pilot_identity/filing_indexes'


def main():
    queue = json.loads((P / 'pilot_identity/share_scale_review_queue.json').read_text())
    s = requests.Session()
    s.headers['User-Agent'] = 'Personal investment research data-quality audit'
    out = []

    def cached(name):
        path = D / name
        if not path.exists():
            url = 'https://data.sec.gov/submissions/' + name
            r = s.get(url, timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
            path.with_suffix('.provenance.json').write_text(json.dumps(dict(source=url, sha256=hashlib.sha256(r.content).hexdigest(), retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
            time.sleep(.2)
        return json.loads(path.read_text())

    for cik in sorted(set(r['cik'] for r in queue)):
        sub = cached('CIK' + cik + '.json')
        targets = {r['accessions'] for r in queue if r['cik'] == cik}
        days = [r['filed'] for r in queue if r['cik'] == cik]
        chunks = [sub['filings']['recent']]
        for item in sub['filings'].get('files', []):
            if item['filingFrom'] <= max(days) and item['filingTo'] >= min(days):
                chunks.append(cached(item['name']))
        for c in chunks:
            for i, acc in enumerate(c.get('accessionNumber', [])):
                if acc in targets:
                    url = 'https://www.sec.gov/Archives/edgar/data/{}/{}/{}'.format(int(cik), acc.replace('-', ''), c['primaryDocument'][i])
                    out.append(dict(cik=cik, accession=acc, filed=c['filingDate'][i], source=url))
                    targets.remove(acc)
        print(cik, 'unresolved', len(targets), flush=True)
        (P / 'pilot_identity/share_scale_filing_urls.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
