import json,re,urllib.parse
from bs4 import BeautifulSoup
import core as c
qp=c.HERE/'BAC/release_queue.json';q=json.loads(qp.read_text())
for a in q:
 if 'newsroom' not in a['url'] or 'evidence_path'not in a:continue
 bs=BeautifulSoup((c.ROOT/a['evidence_path']).read_bytes(),'html.parser');link=next((x for x in bs.find_all('a',href=True) if re.search('earnings release',x.get_text(' ',strip=True),re.I)and'.pdf'in x['href'].lower()),None)
 if link:
  u=urllib.parse.urljoin(a['url'],link['href'])
  try:
   b,p,m=c.fetch(u,'BAC');c.pdf_text(p);a['period_source']=dict(url=u,evidence_path=c.rel(p),evidence_sha256=c.sha(b),meta_path=c.rel(p.with_name(p.name+'.meta.json')),link_proof=c.proof(c.ROOT/a['evidence_path'],a['url'],link.get_text(' ',strip=True)))
   print(a['text'],'OK',flush=True)
  except Exception as e:a['period_source_error']=str(e);print(str(e),flush=True)
 c.dump(qp,q)
