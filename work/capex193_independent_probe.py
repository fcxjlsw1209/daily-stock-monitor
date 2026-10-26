"""Independent direct-source inventory, no producer values or financial ratios."""
from pathlib import Path
from html.parser import HTMLParser
import json, hashlib, re, sys, io
ROOT=Path(__file__).resolve().parents[1]
SCOPE=ROOT/'work/capex193_source_scope.json'
def norm(s):return ' '.join(s.split())
class Tables(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.tables=[];self.stack=[];self.text=[]
 def handle_starttag(self,tag,attrs):
  if tag=='table':
   t={'index':len(self.tables),'context':norm(' '.join(self.text))[-600:],'rows':[],'row':None,'cell':None};self.tables.append(t);self.stack.append(t)
  if not self.stack:return
  t=self.stack[-1]
  if tag=='tr':t['row']=[];t['rows'].append(t['row'])
  if tag in ('td','th') and t['row'] is not None:
   t['cell']={'text':'','colspan':int(dict(attrs).get('colspan','1'))};t['row'].append(t['cell'])
 def handle_endtag(self,tag):
  if not self.stack:return
  t=self.stack[-1]
  if tag in ('td','th'):t['cell']=None
  if tag=='tr':t['row']=None
  if tag=='table':self.stack.pop()
 def handle_data(self,data):
  self.text.append(data)
  if self.stack and self.stack[-1]['cell'] is not None:self.stack[-1]['cell']['text']+=data

def main():
 scope=json.loads(SCOPE.read_text());rows=[]
 for e in scope['prospective_targets']:
  p=ROOT/e['source']['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==e['source']['sha256']
  item={'event_id':e['event_id'],'candidates':[]}
  if raw.startswith(b'%PDF') and '--html-only' in sys.argv:
   item['blocked']='PDF review deferred: inherited PDFium runtime SHA mismatch; no PDF parsing attempted'
  elif raw.startswith(b'%PDF'):
   pdfplan=json.loads((ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json').read_text())
   
   for q,d in pdfplan['pdf_runtime_hashes'].items():assert hashlib.sha256(Path(q).read_bytes()).hexdigest()==d,q
   sys.path.append(pdfplan['pdf_runtime']['bundle_site_packages'])
   from pypdf import PdfReader
   doc=PdfReader(io.BytesIO(raw))
   for i,page in enumerate(doc.pages):
    txt=page.extract_text(extraction_mode='layout')
    if 'cash flows' in txt.lower():
     item['candidates'].append({'page':i+1,'text':txt})
  else:
   h=Tables();h.feed(raw.decode('utf-8',errors='replace'))
   for t in h.tables:
    rr=[[norm(c['text']) for c in r] for r in t['rows']]
    hits=[i for i,r in enumerate(rr) if re.search(r'(propert\w*.*(?:equipment|plant)|capital expend)', ' '.join(r),re.I)]
    for i in hits:
     item['candidates'].append({'table':t['index'],'row':i,'cells':rr[i],'spans':[c['colspan'] for c in t['rows'][i]],'context':t['context'],'headers':rr[:10]})
  rows.append(item)
 print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
