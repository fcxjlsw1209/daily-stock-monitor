"""Cache statement files linked by the issuer's existing quarterly archives."""
import json,re,urllib.parse
from bs4 import BeautifulSoup
import core as c

def run():
 qp=c.HERE/'T/supplement_queue.json';queue=json.loads(qp.read_text());seen={a['url']for a in queue}
 for p in sorted((c.HERE/'T/raw').glob('archive_*.html')):
  m=json.loads(p.with_name(p.name+'.meta.json').read_text())
  if m['status']!=200:continue
  for link in BeautifulSoup(p.read_bytes(),'html.parser').find_all('a',href=True):
   href=link['href']
   if not ('.xlsx'in href.lower() or '.pdf'in href.lower()) or not re.search(r'\.xlsx|master|(?:ATT|atnt)[ _-]*[1-4]Q\d{2,4}[ _-]+Earnings\.pdf|financial.statements',href,re.I):continue
   u=urllib.parse.urljoin(m['url'],href).replace(' ','%20')
   if u in seen:continue
   a=dict(url=u,archive=c.proof(p,m['url'],href),discovery_basis='Explicit issuer quarterly-archive attachment link')
   try:
    b,r,meta=c.fetch(u,'T');a.update(evidence_path=c.rel(r),evidence_sha256=c.sha(b),meta_path=c.rel(r.with_name(r.name+'.meta.json')))
    if b.startswith(b'%PDF'):c.pdf_text(r)
   except Exception as e:a['error']=str(e)
   queue.append(a);seen.add(u);c.dump(qp,queue);print(u,a.get('error','OK'),flush=True)
if __name__=='__main__':run()
