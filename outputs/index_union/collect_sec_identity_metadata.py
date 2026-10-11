"""Reuse SEC caches; fetch only missing candidate CIK metadata at <=2 req/s."""
from pathlib import Path
import json,hashlib,time,requests,datetime
P=Path(__file__).resolve().parent
def main():
 audit=json.loads((P/'SEC_IDENTITY_REVIEW.json').read_text());wanted=set(audit['missing_current_ticker'])
 candidates={}
 for r in json.loads((P.parent/'dynamic_top100/registry_identity_hints.json').read_text()):
  if r['symbol'] in wanted:
   for c in r['candidate_ciks']:candidates.setdefault(c,set()).add(r['symbol'])
 for r in json.loads((P/'FREE_IDENTITY_SOURCE_AUDIT.json').read_text())['records']:
  if r['symbol'] in wanted and r['candidate_cik']:candidates.setdefault(r['candidate_cik'],set()).add(r['symbol'])
 import pandas as pd,ast
 for r in pd.read_csv(P.parent/'dynamic_top100/additional_sources/sec_filer_registry/listed_filer_metadata.csv.gz').to_dict('records'):
  try:tickers=ast.literal_eval(r['tickers'])
  except (ValueError,SyntaxError,TypeError):continue
  for symbol in tickers:
   symbol=symbol.replace('-','.')
   if symbol in wanted:candidates.setdefault(str(int(r['cik'])).zfill(10),set()).add(symbol)
 dest=P/'sources/sec_identity';dest.mkdir(exist_ok=True)
 cached={}
 for f in (P.parent/'dynamic_top100').rglob('*submissions*.json'):
  try:d=json.loads(f.read_text())
  except (ValueError,OSError):continue
  if isinstance(d,dict) and d.get('cik'):cached[str(d['cik']).zfill(10)]=str(f)
 progress=[]
 for cik,symbols in sorted(candidates.items()):
  f=dest/(cik+'_submissions.json')
  if cik in cached or f.exists():continue
  url='https://data.sec.gov/submissions/CIK'+cik+'.json'
  try:
   r=requests.get(url,headers={'User-Agent':'HistoricalFinancialResearch/1.0','Accept':'application/json'},timeout=25)
   if r.status_code in (403,429):print('stop source',r.status_code,flush=True);break
   r.raise_for_status();d=r.json()
   if str(d.get('cik','')).zfill(10)!=cik:raise ValueError('CIK mismatch')
   f.write_bytes(r.content);f.with_suffix('.provenance.json').write_text(json.dumps(dict(url=url,sha256=hashlib.sha256(r.content).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
   progress.append(dict(cik=cik,symbols=sorted(symbols),status='downloaded',issuer=d['name'],tickers=d.get('tickers',[])))
  except (requests.RequestException,ValueError) as e:progress.append(dict(cik=cik,status='unavailable',error=type(e).__name__))
  (dest/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(progress[-1],flush=True);time.sleep(.6)
 print('finished',len(progress),flush=True)
if __name__=='__main__':main()
