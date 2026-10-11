"""Download finite annual primary filings from public mirror; stop on rate limits."""
import requests,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent/'sources/spdr_sectors'
def main():
 rows=json.loads((P/'report_queue.json').read_text());progress=[]
 session=requests.Session();session.headers['User-Agent']='HistoricalFinancialResearch/1.0'
 for row in reversed(rows):
  dest=P/(row['year']+'_report.'+row['document'].rsplit('.',1)[-1]);meta=dest.with_suffix('.provenance.json')
  if dest.exists() and meta.exists():
   if hashlib.sha256(dest.read_bytes()).hexdigest()!=json.loads(meta.read_text())['sha256']:raise ValueError('Cache changed')
   progress.append(dict(year=row['year'],status='cached'));continue
  url='https://edgar-mirror.sec-api.io/1064641/'+row['accession'].replace('-','')+'/'+row['document']
  try:
   r=session.get(url,timeout=45)
   record=dict(year=row['year'],status=r.status_code,bytes=len(r.content))
   if r.status_code in (403,429):
    record['retry_after']=r.headers.get('Retry-After');progress.append(record);break
   if r.status_code==200:
    # Save exact mirror body with SEC manifest identity and provenance.
    dest.write_bytes(r.content);meta.write_text(json.dumps(dict(**row,mirror_url=url,sha256=hashlib.sha256(r.content).hexdigest(),retrieved_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
   progress.append(record);print(record,flush=True)
  except requests.RequestException as e:progress.append(dict(year=row['year'],error=type(e).__name__))
  finally:(P/'download_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
  time.sleep(1)
if __name__=='__main__':main()
