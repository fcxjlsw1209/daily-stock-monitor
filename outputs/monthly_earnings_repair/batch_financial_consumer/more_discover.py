import sys,json,urllib.parse,re
import core as c
from archive_download import links
urls={
'C':'https://www.citigroup.com/global/investors/quarterly-earnings',
'PM':'https://www.pmi.com/investor-relations/reports-filings',
'BUD':'https://www.ab-inbev.com/investors/document-search',
'BAC':'https://investor.bankofamerica.com/regulatory-and-other-filings/select-sec-filings',
'PYPL':'https://newsroom.paypal-corp.com/news-quarterly-earnings?l=100&o=0',
'MA':'https://s25.q4cdn.com/479285134/files/doc_financials/2025/q4/4Q25-Mastercard-Earnings-Release.pdf'}
for s,u in urls.items():
 try:
  b,p,m=c.fetch(u,s,'expanded_archive');
  if b.startswith(b'%PDF'):
   c.pdf_text(p); c.dump(c.HERE/s/'release_queue.json',[dict(url=u,evidence_path=c.rel(p),evidence_sha256=c.sha(b),meta_path=c.rel(p.with_name(p.name+'.meta.json')),discovery_note='Public indexed original issuer-branded earnings-release PDF, identity verified inside source')])
  else:
   ls=links(u,s,'expanded_archive');c.dump(c.HERE/s/'expanded_links.json',ls);print(s,len(ls),[(a['text'],a['url']) for a in ls if re.search('release|earnings|result',a['text'],re.I)][:8],flush=True)
 except Exception as e:print(s,str(e),flush=True)
