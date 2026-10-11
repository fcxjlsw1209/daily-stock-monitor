"""Cache public primary releases; a keyword hit is not a membership event."""
import datetime,hashlib,json,time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent;D=P/'sources/sp_releases'
def main():
 D.mkdir(exist_ok=True);html=P/'sources/sp_search_exact.html';s=BeautifulSoup(html.read_text(),'html.parser');rows=[]
 for item in s.select('.wd_item'):
  a=item.select_one('.wd_title a');date=item.select_one('.wd_date')
  if a:rows.append(dict(url=a['href'],title=a.get_text(' ',strip=True),date=date.get_text(' ',strip=True) if date else None))
 (P/'sp_release_queue.json').write_text(json.dumps(rows,indent=2)+'\n')
 session=requests.Session();records=[]
 for i,row in enumerate(rows,1):
  key=hashlib.sha256(row['url'].encode()).hexdigest()[:16];f=D/(key+'.html');prov=f.with_suffix('.provenance.json')
  if f.exists() and prov.exists():
   m=json.loads(prov.read_text())
   if hashlib.sha256(f.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Cache changed')
  else:
   try:r=session.get(row['url'],timeout=40)
   except requests.RequestException:print('network stopped',i,flush=True);break
   if r.status_code!=200:
    print('HTTP stopped',r.status_code,'Retry-After',r.headers.get('Retry-After'),flush=True);break
   f.write_bytes(r.content);prov.write_text(json.dumps(dict(url=row['url'],sha256=hashlib.sha256(r.content).hexdigest(),retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2));time.sleep(.4)
  body=BeautifulSoup(f.read_text(),'html.parser');tables=[t.get_text(' ',strip=True) for t in body.find_all('table')]
  records.append(dict(**row,file=str(f.relative_to(P)),tables=tables,content_reviewed=False))
  (P/'sp_release_inventory.json').write_text(json.dumps(records,indent=2)+'\n')
  if i%10==0:print('cached',i,'/',len(rows),flush=True)
 print('finished',len(records),'of',len(rows),flush=True)
if __name__=='__main__':main()
