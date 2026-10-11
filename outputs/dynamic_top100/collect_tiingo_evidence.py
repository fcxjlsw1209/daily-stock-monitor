"""Cache public Tiingo evidence. Credentials remain in Keychain; no identity inferred."""
import argparse,datetime,getpass,hashlib,json,subprocess
from pathlib import Path
import requests
P=Path(__file__).resolve().parent/'additional_sources'/'tiingo'
def collect(symbols):
 r=subprocess.run(['/usr/bin/security','find-generic-password','-a',getpass.getuser(),'-s','daily-stock-monitor.tiingo','-w'],capture_output=True,text=True)
 if r.returncode or not r.stdout.strip():raise RuntimeError('Keychain credential unavailable')
 key=r.stdout.strip();session=requests.Session();session.headers['Authorization']='Token '+key
 P.mkdir(exist_ok=True)
 for symbol in symbols:
  if not symbol.replace('-','').replace('.','').isalnum():raise ValueError('Invalid symbol')
  for part in ['metadata','prices']:
   dest=P/(symbol+'_'+part+'.json');provenance=P/(symbol+'_'+part+'.provenance.json')
   if dest.exists() and provenance.exists():
    meta=json.loads(provenance.read_text())
    if hashlib.sha256(dest.read_bytes()).hexdigest()==meta['sha256']:
     print(symbol,part,'cached',flush=True);continue
    raise RuntimeError('Cache checksum mismatch')
   url='https://api.tiingo.com/tiingo/daily/'+symbol+('/prices' if part=='prices' else '')
   params={'startDate':'2008-01-01','endDate':'2026-10-05'} if part=='prices' else {}
   try:response=session.get(url,params=params,timeout=45)
   except requests.RequestException:raise RuntimeError('Network request failed; credentials not logged') from None
   if response.status_code!=200:
    print(symbol,part,'HTTP',response.status_code,flush=True)
    if response.status_code in (401,403,429):raise RuntimeError('Authentication, entitlement or quota blocked; stopped')
    continue
   try:data=response.json()
   except ValueError:raise RuntimeError('Unexpected non-JSON response') from None
   if part=='prices' and (not isinstance(data,list) or any(not {'date','open','high','low','close','volume'}.issubset(x) for x in data)):
    raise RuntimeError('Unexpected price schema')
   if part=='metadata' and (not isinstance(data,dict) or 'name' not in data):raise RuntimeError('Unexpected metadata schema')
   body=json.dumps(data).replace(key,'[REDACTED]').encode();dest.write_bytes(body)
   provenance.write_text(json.dumps({'source':url,'parameters':params,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(body).hexdigest(),'identity_and_adjustments_verified':False},indent=2))
   print(symbol,part,{'name':data.get('name'),'exchange':data.get('exchangeCode')} if isinstance(data,dict) else {'rows':len(data),'first':data[0]['date'] if data else None,'last':data[-1]['date'] if data else None},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('symbols',nargs='+');args=p.parse_args()
 try:collect(args.symbols)
 except (RuntimeError,ValueError) as e:print(str(e));raise SystemExit(1)
