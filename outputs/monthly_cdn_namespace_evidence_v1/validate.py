"""Current publisher namespace evidence only; no historical availability claim."""
import hashlib,json
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from html.parser import HTMLParser
class Links(HTMLParser):
 def __init__(self):super().__init__();self.hrefs=[];self.text=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':self.hrefs.extend(v for k,v in attrs if k=='href')
 def handle_data(self,data):self.text.append(data)
from pypdf import PdfReader
R=Path(__file__).resolve().parents[2]
def digest(b):return hashlib.sha256(b).hexdigest()
def check(ok,why):
 if not ok:raise ValueError(why)
def receipt(base,expected_host):
 base=R/base;mfiles=list((base/'raw').glob('*.meta.json'));check(len(mfiles)==1,'one_receipt')
 m=json.loads(mfiles[0].read_bytes()); raw=base/m['decoded_path']; response=base/m['response_path'];b=raw.read_bytes(); rb=response.read_bytes()
 check(m['status']==200 and m['public_unauthenticated'] is True,'http200')
 check(m['url']==m['final_url'] and urlparse(m['url']).scheme=='https' and urlparse(m['url']).hostname==expected_host,'issuer_host')
 check(m['decoded_sha256']==digest(b) and m['response_sha256']==digest(rb),'receipt_hash')
 check(m['decoded_bytes']==len(b) and m['response_bytes']==len(rb),'receipt_size')
 check(datetime.fromisoformat(m['retrieved_at']).utcoffset() is not None,'retrieval_time')
 if m['decoder']=='identity':check(rb==b,'identity_decode')
 else:raise ValueError('unsupported_decoder')
 return m,b,raw,mfiles[0],response

def namespace(url,host,tenant):
 u=urlparse(url)
 return u.scheme=='https' and u.hostname==host and u.path.startswith('/'+tenant+'/files/doc_financials/')

def main():
 reports=[];pins={}
 def pin(p):pins[str(p.relative_to(R))]=digest(p.read_bytes())
 for issuer,base,host,cdn,tenant in [('PFE','outputs/monthly_pfizer_cdn_attribution_v1','cdn.pfizer.com','s206.q4cdn.com','795948973'),('V','outputs/monthly_visa_cdn_attribution_v1','www.visa.ca','s1.q4cdn.com','050606653')]:
  m,b,raw,mp,rp=receipt(base,host)
  for p in [raw,mp,rp]:pin(p)
  if issuer=='PFE':
   r=PdfReader(raw); t=r.pages[87].extract_text();url='https://s206.q4cdn.com/795948973/files/doc_financials/2024/ar/10K_Final.pdf'
   check("Pfizer's 2024 10-K Report is available online at:" in t and url in t,'explicit_issuer_attribution')
   anchor={'page_1based':88,'printed_page':87,'url':url,'description':"Pfizer's 2024 10-K Report"}
   scopepath=R/'outputs/monthly_pfizer_cached_scope_v2/SCOPE.json';scope=json.loads(scopepath.read_bytes());urls=[x['source']['url']for x in scope['documents']];pin(scopepath)
  else:
   s=Links();s.feed(b.decode('utf-8'));url='https://s1.q4cdn.com/050606653/files/doc_financials/2019/q4/CORRECTED-TRANSCRIPT-Visa,-Inc.(V-US),-Q4-2019-Earnings-Call,-24-October-2019-5-00-PM-ET.pdf';anchors=[v for v in s.hrefs if v==url]
   check(len(anchors)==1 and 'Visa' in ' '.join(s.text),'explicit_issuer_attribution')
   fulltext=' '.join(s.text);pos=fulltext.index(url);context=fulltext[pos:pos+len(url)+200];check('Visa' in context and 'Fiscal Year 2019' in context,'issuer_financial_context')
   anchor={'href':url,'context':context}
   urls=[]
   for path in ['outputs/monthly_visa_release_parser_v2/PLAN.json','outputs/monthly_visa_release_parser_v3/PLAN.json']:
    pp=R/path;pin(pp);scope=json.loads(pp.read_bytes());urls.extend(x['url']for x in scope['documents'])
  check(namespace(url,cdn,tenant),'linked_namespace');check(all(namespace(u,cdn,tenant)for u in urls),'candidate_namespace')
  reports.append({'issuer':issuer,'issuer_host':host,'source_url':m['url'],'anchor':anchor,'cdn_host':cdn,'tenant':tenant,'candidate_documents':len(urls),'current_issuer_namespace_reference':True,'exact_candidate_files_linked':False,'historical_namespace_ownership_verified':False,'historical_pdf_publication_verified':False,'semantic_admissions':0})
 pin(Path(__file__).resolve())
 out={'status':'CURRENT_NAMESPACE_REFERENCE_ONLY','reports':reports,'source_hashes':pins,'semantic_admissions':0,'network_requests':0,'independent_audit':False}
 p=Path(__file__).with_name('RESULT.json');check(not p.exists(),'result_already_exists');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'reports':reports,'pins':len(pins)},indent=2))
if __name__=='__main__':main()
