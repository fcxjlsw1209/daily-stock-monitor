"""Keep the API key in macOS Keychain; save only public response data."""
import argparse,csv,datetime,getpass,hashlib,io,json,os,subprocess,time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
SERVICE='daily-stock-monitor.alpha-vantage'
ACCOUNT=getpass.getuser()
P=Path(__file__).resolve().parent/'additional_sources'/'alpha_vantage'
def keychain_read():
 r=subprocess.run(['/usr/bin/security','find-generic-password','-a',ACCOUNT,'-s',SERVICE,'-w'],capture_output=True,text=True)
 if r.returncode:raise RuntimeError('Keychain item unavailable. Run setup first or allow Keychain access.')
 return r.stdout.strip()
def setup():
 key=getpass.getpass('Paste Alpha Vantage API key (hidden): ').strip()
 if not key or not key.isalnum():raise RuntimeError('Expected a nonempty alphanumeric API key; nothing saved.')
 r=subprocess.run(['/usr/bin/security','add-generic-password','-U','-a',ACCOUNT,'-s',SERVICE,'-w',key],capture_output=True,text=True)
 if r.returncode:raise RuntimeError('Keychain save failed; no credential details printed.')
 print('Saved to macOS Keychain. No key written to project files.')
def probe(requests=None):
 key=keychain_read();P.mkdir(parents=True,exist_ok=True)
 for day,state in (requests or [('2010-01-04','active'),('2023-10-16','delisted')]):
  stem=day+'_'+state;meta=P/(stem+'.json');out=P/(stem+'.csv')
  if meta.exists() and out.exists():
   old=json.loads(meta.read_text())
   if old.get('validated') and hashlib.sha256(out.read_bytes()).hexdigest()==old.get('sha256'):
    print(stem,'cached',old['rows'],'rows');continue
  params=dict(function='LISTING_STATUS',date=day,state=state,apikey=key)
  # Never log request URL or exception text: query contains credential.
  try:
   with urlopen('https://www.alphavantage.co/query?'+urlencode(params),timeout=40) as response:raw=response.read()
  except Exception:raise RuntimeError('Alpha Vantage request failed; no credential-bearing URL logged.') from None
  body=raw.decode('utf-8-sig').replace(key,'[REDACTED]')
  if not body.startswith('symbol,'):
   raise RuntimeError('API did not return listing CSV (possible quota or entitlement issue). Response not saved; stopped to preserve quota.')
  rows=list(csv.DictReader(io.StringIO(body)))
  if not rows or not {'symbol','name','exchange','assetType','ipoDate','delistingDate','status'}.issubset(rows[0]):raise RuntimeError('Unexpected listing schema; stopped.')
  if any(r['status'].lower()!=state for r in rows):raise RuntimeError('Unexpected listing status in response.')
  if state=='active' and any(r['ipoDate'] not in ('','null','None') and r['ipoDate']>day for r in rows):raise RuntimeError('Response contains future IPO dates; stopped.')
  if state=='delisted' and any(r['delistingDate'] not in ('','null','None') and r['delistingDate']>day for r in rows):raise RuntimeError('Response contains future delisting dates; stopped.')
  clean=body.encode();out.write_bytes(clean)
  record=dict(date=day,state=state,rows=len(rows),validated=True,sha256=hashlib.sha256(clean).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='https://www.alphavantage.co/query',function='LISTING_STATUS',historical_identity_and_completeness_verified=False)
  meta.write_text(json.dumps(record,indent=2));print(stem,len(rows),'rows saved');time.sleep(1)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['setup','probe','history']);args=parser.parse_args()
 try:
  if args.mode=='setup':setup()
  elif args.mode=='history':probe([(str(year)+'-01-04','active') for year in range(2011,2027)]+[('2026-10-05','active'),('2026-10-05','delisted')])
  else:probe()
 except RuntimeError as e:print(str(e));raise SystemExit(1)
