"""Find pre-pilot original reports with cached historical filing indexes."""
from pathlib import Path
import requests, json, hashlib, datetime, time
P = Path(__file__).resolve().parent
D = P / 'pilot_identity/filing_indexes'
D.mkdir(exist_ok=True)
s = requests.Session()
s.headers['User-Agent'] = 'Personal investment research data-quality audit'
results, errors = [], []
for folder in ['foreign_sec', 'priority_sec']:
 for f in sorted((P / folder).glob('*_submissions.json')):
  symbol = f.name.split('_')[0]
  if symbol in ['ARM', 'BABA', 'SHOP', 'BUD']:
   continue  # Do not retrospectively substitute later issuer entities.
  sub = json.loads(f.read_text())
  cik = str(sub['cik']).zfill(10)
  chunks = [sub['filings']['recent']]
  for a in sub['filings'].get('files', []):
   if a['filingFrom'] > '2010-03-31' or a['filingTo'] < '2008-01-01':
    continue
   dest = D / a['name']
   url = 'https://data.sec.gov/submissions/' + a['name']
   try:
    if not dest.exists():
     r = s.get(url, timeout=40)
     r.raise_for_status()
     dest.write_bytes(r.content)
     dest.with_suffix('.provenance.json').write_text(json.dumps(dict(url=url, sha256=hashlib.sha256(r.content).hexdigest(), retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
     time.sleep(.15)
    chunks.append(json.loads(dest.read_text()))
   except Exception as e:
    errors.append(dict(symbol=symbol, source=url, error=type(e).__name__))
  reports = []
  for chunk in chunks:
   for i, form in enumerate(chunk.get('form', [])):
    day = chunk['filingDate'][i]
    if form not in ['20-F', '40-F', '10-K'] or not '2008-01-01' <= day < '2010-03-01':
     continue
    acc, doc = chunk['accessionNumber'][i], chunk['primaryDocument'][i]
    reports.append(dict(filed=day, form=form, accession=acc, url=f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace("-", "")}/{doc}'))
  reports = sorted({r['accession']: r for r in reports}.values(), key=lambda r: r['filed'])
  results.append(dict(symbol=symbol, cik=cik, sec_name=sub['name'], latest_report=reports[-1] if reports else None, reports=reports))
  print(symbol, reports[-1]['filed'] if reports else 'NONE', flush=True)
  (D / 'PRE_PILOT_REPORTS.json').write_text(json.dumps(dict(issuers=results, errors=errors), indent=2))
