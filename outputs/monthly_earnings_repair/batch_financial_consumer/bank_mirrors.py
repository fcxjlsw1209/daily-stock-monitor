"""Public issuer-hosted filing HTML plus issuer-linked full release exhibits."""
import json,re,urllib.parse
from bs4 import BeautifulSoup
import core as c

def select_press(links):
 """Prefer the financial release, excluding a separate repurchase announcement."""
 for pattern in [r'^(?:THE )?PRESS RELEASE$',r'^EX-99\.1$',r'^EXHIBIT 99\.1$']:
  found=next((x for x in links if re.match(pattern,x.get_text(' ',strip=True).rstrip(' »'),re.I)),None)
  if found is not None:return found
 return next((x for x in links if re.match(r"^PRESS RELEASE DATED",x.get_text(" ",strip=True),re.I) and re.search(r"(?:ex|dex)99[._]?1\.",x["href"],re.I)),None)

def run():
 grouped={};qp=c.HERE/'BAC/release_queue.json';queue=json.loads(qp.read_text())
 for p in sorted((c.HERE/'BAC').glob('indexed_filing_*.json')):
  if p.name.endswith('.meta.json'):continue
  for title,url in re.findall(r'^\s*(.*?) \((https://[^\s]+)\)',json.loads(p.read_text())['response'],re.M):
   if not re.match(r'https://investor.bankofamerica.com/regulatory-and-other-filings/(?:select-sec-filings|current-reports)/content/',url):continue
   am=re.search(r'/content/(\d{10}-\d{2}-\d{6})/',url)
   if not am:continue
   a=dict(url=url.split('?')[0],text=title,discovery_index=c.proof(p,url))
   if am[1]not in grouped or 'PRESS RELEASE'in title:grouped[am[1]]=a
 for accession,a in sorted(grouped.items()):
  try:
   b,p,m=c.fetch(a['url'],'BAC')
   if b.startswith(b'%PDF'):
    # A complete original filing may include both the announcement and full exhibit.
    # The separate parser must locate the earnings-release heading after its cover.
    mu=pu=a['url'];mb=pb=b;mp=pp=p;link=c.proof(p,mu)
   else:
    bs=BeautifulSoup(b,'html.parser');links=bs.find_all('a',href=True)
    main=next((x for x in links if re.match(r'^(?:FORM )?8-K\b',x.get_text(' ',strip=True))),None);press=select_press(links)
    if main is None or press is None:raise ValueError('Issuer document links do not identify both 8-K and earnings-release exhibit')
    mu=urllib.parse.urljoin(a['url'],main['href']);pu=urllib.parse.urljoin(a['url'],press['href'])
    if not all(urllib.parse.urlsplit(u).netloc=='investor.bankofamerica.com'for u in [mu,pu]):raise ValueError('Link is not on the issuer host')
    mb,mp,mm=c.fetch(mu,'BAC');pb,pp,pm=c.fetch(pu,'BAC');link=c.proof(p,a['url'],press['href'])
   item=dict(url=mu,evidence_path=c.rel(mp),evidence_sha256=c.sha(mb),meta_path=c.rel(mp.with_name(mp.name+'.meta.json')),issuer_filing_mirror=True,release_exhibit=dict(url=pu,evidence_path=c.rel(pp),evidence_sha256=c.sha(pb)),attachment_link_evidence=link,discovery_index=a['discovery_index'],accession=accession)
   queue=[r for r in queue if not (r.get('issuer_filing_mirror') and r.get('accession')==accession)]
   queue.append(item);c.dump(qp,queue);print(accession,'PAIR OK',len(mb),len(pb),flush=True)
  except Exception as e:
   a.update(accession=accession,error=str(e));failed=c.HERE/'BAC/mirror_failures.json';v=json.loads(failed.read_text())if failed.exists()else[];v.append(a);c.dump(failed,v);print(accession,str(e),flush=True)
if __name__=='__main__':run()
