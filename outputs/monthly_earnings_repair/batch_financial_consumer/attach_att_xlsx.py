"""Link each original workbook to a release within the same issuer quarterly package."""
import json,re,urllib.parse
from bs4 import BeautifulSoup
import core as c
qp=c.HERE/'T/supplement_queue.json';queue=json.loads(qp.read_text());byurl={a['url']:a for a in queue}
for p in (c.HERE/'T/raw').glob('archive_*.html'):
 m=json.loads(p.with_name(p.name+'.meta.json').read_text())
 if m['status']!=200:continue
 for box in BeautifulSoup(p.read_bytes(),'html.parser').select('div.newsWrapper'):
  title=box.get_text(' ',strip=True);qm=re.match(r'(First|Second|Third|Fourth)[ -]Quarter Earnings Package',title,re.I)
  if not qm:continue
  releases=[];schedules=[]
  for row in box.select('tr'):
   rt=row.get_text(' ',strip=True)
   urls=[urllib.parse.urljoin(m['url'],a['href']).replace(' ','%20')for a in row.find_all('a',href=True)]
   if rt.startswith('News Release'):releases+=urls
   if rt.startswith('Financial and Operational Schedules'):schedules+=urls
  for u in schedules:
   if u not in byurl or 'evidence_path'not in byurl[u]:continue
   a=byurl[u];a.update(related_release_urls=releases,archive_quarter=['first','second','third','fourth'].index(qm[1].lower())+1,quarter_package_evidence=c.proof(p,m['url'],qm.group()))
   if u.lower().endswith('.xlsx'):c.workbook_text(c.ROOT/a['evidence_path'])
c.dump(qp,queue)
