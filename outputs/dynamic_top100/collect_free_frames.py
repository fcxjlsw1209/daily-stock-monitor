"""Public SEC frames as discovery evidence only, NEVER point-in-time values."""
from pathlib import Path
import requests,json,time,datetime
P=Path(__file__).resolve().parent/'additional_sources';D=P/'sec_frames';D.mkdir(exist_ok=True)
records=[];issuers={}
for y in range(2010,2027):
 for q in range(1,5):
  if y==2026 and q>2:continue
  name=f'CY{y}Q{q}I';url=f'https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/{name}.json';path=D/(name+'.json')
  rec=dict(frame=name,url=url)
  try:
   if not path.exists():
    r=requests.get(url,headers={'User-Agent':'Personal investment research data-quality audit'},timeout=30);r.raise_for_status();path.write_bytes(r.content);time.sleep(.3)
   data=json.loads(path.read_text());rec['rows']=len(data.get('data',[]))
   for row in data.get('data',[]):
    cik=str(row['cik']).zfill(10);entry=issuers.setdefault(cik,dict(cik=cik,names=[],first_frame=name,last_frame=name,frames=0))
    if row['entityName'] not in entry['names']:entry['names'].append(row['entityName'])
    entry['last_frame']=name;entry['frames']+=1
  except Exception as e:rec['error']=str(e)
  records.append(rec);(D/'registry.json').write_text(json.dumps(records,indent=2));(D/'issuer_discovery.json').write_text(json.dumps(list(issuers.values()),indent=2))
  print(name,rec.get('rows',rec.get('error')),len(issuers),'unique CIK',flush=True)
