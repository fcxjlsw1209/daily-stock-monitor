import json
import core as c
h=json.loads((c.HERE.parent/'SEC_REVIEW_QUEUE.json').read_text())['candidates'];out=[]
for s,year,base in [('NKE',2021,'https://investors.nike.com/financial-information/sec-filings/content')]:
 a=next(x for x in h if x['symbol']==s and x['filing_date'].startswith(str(year)) and (s!='NKE'or x['filing_date'].startswith('2021-06')))
 u=base+'/'+a['accession']+'/'+a['accession']+'.pdf';row=dict(ticker=s,url=u,discovery_hint=a,discovery_basis='Cached filing accession maps to issuer-linked original document route; filing and acceptance dates are discovery hints only')
 try:
  b,p,m=c.fetch(u,s);row.update(evidence_path=c.rel(p),evidence_sha256=c.sha(b));print(s,m['status'],len(b),b[:10]);print(c.pdf_text(p)[:1700]if b.startswith(b'%PDF')else c.text(b)[:200])
 except Exception as e:row['error']=str(e);print(s,e,flush=True)
 out.append(row);c.dump(c.HERE/'MIRROR_PROBE.json',out)
