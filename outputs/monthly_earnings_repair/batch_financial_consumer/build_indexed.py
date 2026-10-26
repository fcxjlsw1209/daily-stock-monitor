"""Explicitly indexed issuer excerpts; never mislabeled as original raw bytes."""
import json,re,collections
import core as c
from build_events import extract
ALLOWED={'NKE':['investors.nike.com'],'MA':['investor.mastercard.com','s25.q4cdn.com'],'PM':['www.pmi.com'],'TMO':['ir.thermofisher.com','india.newsroom.thermofisher.com'],'WFC':['newsroom.wf.com','www.wellsfargo.com'],'MCD':['corporate.mcdonalds.com'],'DIS':['thewaltdisneycompany.com','investors.thewaltdisneycompany.com'],'PEP':['www.pepsico.com','investor.pepsico.com'],'CMCSA':['www.cmcsa.com','cmcsa.gcs-web.com'],'COST':['investor.costco.com','s201.q4cdn.com']}
def main():
 events=[];pending=[]
 for s,hosts in ALLOWED.items():
  for p in (c.HERE/s).glob('indexed*.json'):
   if p.name.endswith('.meta.json'):continue
   v=json.loads(p.read_text());r=v['response']
   for block in r.split('--------------------------------------------------------------------------------'):
    m=re.match(r'^\s*(.*?) \((https://[^\s]+)\)',block)
    if not m:continue
    url=m[2]
    if not any('https://'+h+'/' in url for h in hosts):continue
    a=dict(url=url,evidence_path=c.rel(p),evidence_sha256=c.sha(p.read_bytes()),index_block=block,index_query=v.get('query',v.get('queries')))
    try:events.append(extract(a,s))
    except Exception as e:pending.append(dict(ticker=s,source_url=url,evidence_path=c.rel(p),reason=str(e)))
 unique={}
 for x in events:unique.setdefault((x['ticker'],x['fiscal_year'],x['fiscal_quarter']),x)
 out=sorted(unique.values(),key=lambda x:(x['ticker'],x['release_date']));c.dump(c.HERE/'indexed_events.json',out);c.dump(c.HERE/'INDEXED_PENDING.json',pending)
 print('INDEXED',len(out),dict(collections.Counter(x['ticker']for x in out)));print('PENDING',dict(collections.Counter(x['ticker']+': '+x['reason']for x in pending)))
if __name__=='__main__':main()
