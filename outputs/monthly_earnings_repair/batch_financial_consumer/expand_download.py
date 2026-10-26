import sys,json,re,urllib.parse
import core as c
from bs4 import BeautifulSoup
from archive_download import links

def saveq(s,items):
 qp=c.HERE/s/'release_queue.json';old=json.loads(qp.read_text()) if qp.exists() else [];merged={a['url']:a for a in old}
 for a in items:merged.setdefault(a['url'],a)
 q=list(merged.values());c.dump(qp,q);print(s,'QUEUE',len(q),flush=True)
 for i,a in enumerate(q):
  if a.get('evidence_path'):continue
  try:
   a['url']=a['url'].replace(' ','%20');b,p,m=c.fetch(a['url'],s);a.update(evidence_path=c.rel(p),evidence_sha256=c.sha(b),meta_path=c.rel(p.with_name(p.name+'.meta.json')));a.pop('error',None)
   if b.startswith(b'%PDF'):c.pdf_text(p)
   print(s,i+1,'OK',p.name,flush=True)
  except Exception as e:a['error']=str(e);print(s,i+1,'ERROR',str(e),flush=True)
  c.dump(qp,q)
 return q

def run(s):
 items=[]
 if s in ['HD','T']:return saveq(s,[])
 if s=='C':
  p=c.HERE/s/'raw'/'expanded_archive.html';url='https://www.citigroup.com/global/investors/quarterly-earnings';bs=BeautifulSoup(p.read_bytes(),'html.parser')
  for img in bs.find_all('img',alt=True):
   alt=img['alt'];a=img.find_parent('a',href=True)
   if a and 'Press Release PDF' in alt and any(str(y) in alt for y in range(2010,2027)):items.append(dict(url=urllib.parse.urljoin(url,a['href']),text=alt,archive=c.proof(p,url)))
 if s=='WFC':
  for y in range(2020,2010,-1):
   for q in ['fourth','third','second','first']:
    items.append(dict(url=f'https://www.wellsfargo.com/assets/pdf/about/investor-relations/earnings/{q}-quarter-{y}-earnings.pdf',discovery_note='Candidate based on observed official historical archive naming; all facts require original body evidence'))
 if s=='MA':
  for y in range(2026,2017,-1):
   for q in range(4,0,-1):
    if y==2026 and q>2:continue
    items.append(dict(url=f'https://s25.q4cdn.com/479285134/files/doc_financials/{y}/q{q}/{q}Q{y%100:02}-Mastercard-Earnings-Release.pdf',discovery_note='Independent public original-release CDN; path candidates based on indexed 2025Q4 source, issuer verified from body'))
 if s=='PYPL':
  ls=json.loads((c.HERE/s/'expanded_links.json').read_text());items=[a for a in ls if re.search(r'PayPal Reports .*Quarter',a['text'])]
 if s=='HSBC':
  u='https://www.hsbc.com/investors/results-and-announcements/all-reporting/group';ls=links(u,s,'full_archive');items=[a for a in ls if a['url'].endswith('.pdf') and re.search(r'(Earnings Release|media release)',a['text']) and not re.search(r'Chinese|Form 6-K|Transcript',a['text'],re.I)]
 saveq(s,items)
if __name__=='__main__':
 for s in sys.argv[1:]:
  try:run(s)
  except Exception as e:print('FAILED',s,str(e),flush=True)
