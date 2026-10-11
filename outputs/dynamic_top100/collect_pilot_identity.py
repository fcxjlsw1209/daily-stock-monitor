"""Retrieve historical issuer evidence; do not equate successor companies."""
from pathlib import Path
import requests,json,hashlib,datetime
P=Path(__file__).resolve().parent/'pilot_identity';P.mkdir(exist_ok=True)
# Discovery CIKs verified against returned SEC records/filing history before use.
targets={'Google':1288776,'Hewlett-Packard':47217,'Kraft Foods':1103982,'News Corp':1308161,'Exxon Mobil':34088,'Medtronic':64670}
targets.update({'CVS Caremark':64803,'Alcon':1167379,'Eli Lilly':59478,'Blackrock':1364742,'Bank of New York Mellon':1390777,'Walgreen':104207,'Freeport-McMoran Copper & Gold':831259,"Lowe's Companies":60667,'E I Du Pont de Nemours':30554,'Wellpoint':1156039,'Travelers Cos.':86312,'Mosaic':1285785,'Costco Wholesale':909832,'Newmont Mining':1164727,'Johnson Controls':53669,'FPL Group':753308,'Allergan':850693,'Time Warner Cable':1377013,'TJX Cos':109198,'Chubb':20171,'National Oilwell Varco':1021860,'Motorola':68505})
targets.update({'Express Scripts':885721,'Covidien':1385187,'Tyco International':833444,'Ace':896159,'DirecTV':1465112})
s=requests.Session();s.headers['User-Agent']='Personal investment research data-quality audit'
for name,cik in targets.items():
 for kind,url in [('submissions',f'https://data.sec.gov/submissions/CIK{cik:010}.json'),('facts',f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010}.json')]:
  f=P/f'{cik:010}_{kind}.json'
  if f.exists():j=json.loads(f.read_text())
  else:
   r=s.get(url,timeout=40)
   if r.status_code!=200:print(name,kind,r.status_code,flush=True);continue
   j=r.json();f.write_bytes(r.content);f.with_suffix('.provenance.json').write_text(json.dumps({'source':url,'sha256':hashlib.sha256(r.content).hexdigest(),'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
  print(name,kind,j.get('name',j.get('entityName')),str(j.get('formerNames',[]))[:180],flush=True)
