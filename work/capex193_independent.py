"""CAPEX193 direct original-source audit. No producer financial parser or strategy imports."""
from pathlib import Path
from html.parser import HTMLParser
from decimal import Decimal
from datetime import date, datetime, timezone
from bisect import bisect_right
import argparse, base64, csv, hashlib, io, json, re, sys, zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_capex193_source/independent'
SCOPE='work/capex193_source_scope.json'
SCOPE_SHA='71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08'
RUNTIME='work/capex193_pdf_runtime/RUNTIME.json'
RUNTIME_SHA='ad437674d06761ef278da476110c1d3aedf386cd5a0276161304627d2140c98a'
DATA='outputs/monthly_cash_quality_source_admission_v1/DATASET.json'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SESSIONS_SHA='c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def norm(s):return ' '.join(s.split())
def compact(s):return re.sub(r'\s+','',s).lower()
def sha(b):return hashlib.sha256(b).hexdigest()
def path(s):
 p=Path(s);p=p if p.is_absolute() else ROOT/p
 need(p.resolve().is_relative_to(ROOT),'nonlocal path');return p

def pinned(s,d):
 b=path(s).read_bytes();need(sha(b)==d,'SHA mismatch: '+str(s));return b

def load(s):return json.loads(path(s).read_text())
def publish(name,value):
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def runtime():
 r=json.loads(pinned(RUNTIME,RUNTIME_SHA));a=json.loads(pinned(r['acquisition']['path'],r['acquisition']['sha256']))
 for p,h in r['pdf_runtime_hashes'].items():pinned(p,h)
 for p,h in a['members'].items():pinned(p,h)
 base=path(RUNTIME).parent;site=path(r['pdf_runtime']['bundle_site_packages'])
 meta=json.loads(pinned(base/'pypi_release.json',a['metadata_sha256']))
 matches=[x for x in meta['urls'] if x['filename']==a['wheel_filename']]
 need(len(matches)==1 and matches[0]['digests']['sha256']==a['wheel_sha256'] and matches[0]['url']==a['wheel_url'],'official wheel metadata')
 wheel=pinned(base/a['wheel_filename'],a['wheel_sha256']);z=zipfile.ZipFile(io.BytesIO(wheel));record=[n for n in z.namelist() if n.endswith('.dist-info/RECORD')]
 need(len(record)==1,'wheel RECORD')
 count=0
 for n,d,size in csv.reader(io.StringIO(z.read(record[0]).decode())):
  b=z.read(n);need((site/n).read_bytes()==b,'wheel installed bytes: '+n)
  if d:
   alg,h=d.split('=',1);need(alg=='sha256' and base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('=')==h and len(b)==int(size),'RECORD '+n);count+=1
 need(count==a['RECORD_verified_members']==70,'RECORD count')
 old=json.loads(pinned(r['old_plan']['path'],r['old_plan']['sha256']));n=0
 for p,h in old['pdf_runtime_hashes'].items():
  if '/pypdf/' in p:
   suffix=p[p.index('/pypdf/')+1:];need(sha((site/suffix).read_bytes())==h,'original pypdf '+suffix);n+=1
 need(n==r['pypdf_original_files_verified_and_copied']==55,'pypdf count')
 need(not any(k in sys.modules for k in ('pypdf','pypdfium2','pypdfium2_raw')),'PDF runtime already imported')
 sys.path.insert(0,str(site))
 import pypdf,pypdfium2
 need(Path(pypdf.__file__).resolve().is_relative_to(site) and Path(pypdfium2.__file__).resolve().is_relative_to(site),'PDF module origin')
 return pypdf,pypdfium2,dict(manifest_sha256=RUNTIME_SHA,verified_runtime_files=len(r['pdf_runtime_hashes']),official_wheel_sha256=a['wheel_sha256'],record_verified_members=count,pypdf_original_files=n,old_runtime_mismatches=r['old_runtime_mismatches'],historical_runtime_currently_passes=False)

class Tables(HTMLParser):
 """Independent stdlib parser; keeps physical cells and explicit spans."""
 def __init__(self):super().__init__(convert_charrefs=True);self.tables=[];self.stack=[];self.text=[]
 def handle_starttag(self,tag,attrs):
  if tag=='table':
   t={'index':len(self.tables),'context':norm(' '.join(self.text))[-1200:],'rows':[],'row':None,'cell':None,'preamble':[]};self.tables.append(t);self.stack.append(t)
  if not self.stack:return
  t=self.stack[-1]
  if tag=='tr':t['row']=[];t['rows'].append(t['row'])
  if tag in ('td','th') and t['row'] is not None:
   a=dict(attrs);t['cell']={'text':'','colspan':int(a.get('colspan','1')),'rowspan':int(a.get('rowspan','1'))};t['row'].append(t['cell'])
 def handle_endtag(self,tag):
  if not self.stack:return
  t=self.stack[-1]
  if tag in ('td','th'):t['cell']=None
  if tag=='tr':t['row']=None
  if tag=='table':self.stack.pop()
 def handle_data(self,data):
  self.text.append(data)
  if self.stack and not self.stack[-1]['rows']:self.stack[-1]['preamble'].append(data)
  if self.stack and self.stack[-1]['cell'] is not None:self.stack[-1]['cell']['text']+=data

def grid(table):
 g={}
 for r,row in enumerate(table['rows']):
  col=0
  for p,c in enumerate(row):
   while (r,col) in g:col+=1
   need(c['colspan']>0 and c['rowspan']>0,'bad spans')
   item=dict(c,physical=p,row=r,columns=list(range(col,col+c['colspan'])))
   for dr in range(c['rowspan']):
    for dc in range(c['colspan']):need((r+dr,col+dc) not in g,'overlap');g[r+dr,col+dc]=item
   col+=c['colspan']
 return g

def rowtext(t,r):return norm(' '.join(c['text'] for c in t['rows'][r]))
def number(literal):
 s=re.sub(r'\s+','',literal).replace('$','').replace('−','-')
 need(bool(re.fullmatch(r'(?:\([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\)|-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',s)),'ambiguous monetary literal: '+repr(literal))
 neg=s.startswith('(');s=s.strip('()').replace(',','');v=Decimal(s);return -v if neg else v

def selected(t,r,cols):
 g=grid(t);items={}
 for c in cols:
  need((r,c) in g,'missing annual cell');x=g[r,c];need(x['row']==r and x['rowspan']==1 and set(x['columns']).issubset(cols),'mixed financial columns');items[x['physical']]=x
 return norm(' '.join(items[p]['text'] for p in sorted(items))),[{'physical':p,'columns':items[p]['columns'],'text':norm(items[p]['text'])} for p in sorted(items)]

def header_evidence(t,loc):
 g=grid(t);start=loc.get('statement_segment',{}).get('start_row',0);stop=loc['row_index_zero_based'];cols=loc['logical_columns']
 # Stop at the first net-income data row, never harvest later monetary rows as headers.
 ni=[i for i in range(start,stop) if re.match(r'\s*Net (?:income|loss|\(loss\))',rowtext(t,i),re.I)]
 if ni:stop=min(ni)
 rows=[]
 for r in range(start,stop):
  ss=[]
  for c in cols:
   x=g.get((r,c));
   if x and norm(x['text']) and norm(x['text']) not in ss:ss.append(norm(x['text']))
  if ss:rows.append({'row':r,'text':' | '.join(ss)})
 text=' '.join(x['text'] for x in rows)
 for lit in loc['header_literals']:
  need(compact(lit) in compact(text) or compact(lit) in compact(t['context']),'annual/header literal missing '+lit)
 need(str(date.fromisoformat(loc['period_end']).year) in text,'annual year missing')
 if 'cash' in loc['statement_title'].lower():
  need(re.search(r'(?:twelve\s+months|years?|fiscal\s+year)\s*(?:ended)?',text,re.I) and not re.search(r'three\s+months|quarter',text,re.I),'annual column not isolated')
 context=t['context']+' '+norm(' '.join(t.get('preamble',[])))+' '+' '.join(rowtext(t,r) for r in range(start,min(stop,start+5)))
 unit_word={'1000':'thousands','1000000':'millions'}.get(loc['scale']);need(unit_word and re.search(r'\b'+unit_word+r'\b',context,re.I),'units '+repr(context))
 title=norm(loc['statement_title']).lower();need(compact(title) in compact(context),'statement title absent')
 return rows

def html_pair(h,field):
 if field is None:return {'checked':False,'reason':'inherited_field_missing'}
 loc=field['locator'];need(loc['format']=='html','HTML locator')
 t=h.tables[loc['table_index_zero_based']];r=loc['row_index_zero_based'];text=rowtext(t,r)
 need(norm(loc['row_label']).lower() in text.lower(),'original row label')
 lit,cells=selected(t,r,loc['logical_columns']);v=number(lit)
 need(v==Decimal(field['value_decimal']),'inherited amount does not match raw HTML')
 heads=header_evidence(t,loc)
 return {'checked':True,'value_decimal':str(v),'scale':field['scale'],'annual_period_end':loc['period_end'],'row':r,'table':t['index'],'literal':lit,'header_evidence':heads,'cells':cells,'row_text':text,'row_text_sha256':sha(text.encode())}

def html_extract(raw,event):
 h=Tables();h.feed(raw.decode('utf-8',errors='strict'));cfo=event['fields']['current'].get('operating_cash_flow')
 pairs={}
 for period,name in [('current','operating_cash_flow'),('current','total_assets'),('comparison','total_assets')]:
  f=event['fields'][period].get(name)
  if f is not None and f['source']['sha256']==sha(raw):pairs[period+'_'+name]=html_pair(h,f)
  else:pairs[period+'_'+name]={'checked':False,'reason':'field_missing_or_other_pinned_source'}
 allhits=[]
 for t in h.tables:
  for r in range(len(t['rows'])):
   text=rowtext(t,r)
   if re.search(r'(?:purchase|additions?|payments?).*propert.*(?:equipment|plant)',text,re.I):allhits.append({'table':t['index'],'row':r,'text':text,'context':t['context']})
 if cfo is None:return None,['missing_formal_cashflow_current_annual_column'],allhits,pairs
 loc=cfo['locator'];t=h.tables[loc['table_index_zero_based']];start=loc.get('statement_segment',{}).get('start_row',0);end=loc.get('statement_segment',{}).get('end_row_exclusive',len(t['rows']))
 hits=[x for x in allhits if x['table']==t['index'] and start<=x['row']<end]
 pure=[x for x in hits if not re.search(r'intangible|principal|finance\s+lease|non.cash',x['text'],re.I)]
 if len(pure)!=1:
  reason='combined_ppe_and_intangible_not_separable' if hits and not pure else 'multiple_scope_pp_and_e_rows' if len(pure)>1 else 'cash_pp_and_e_row_missing'
  return None,[reason],allhits,pairs
 x=pure[0];r=x['row'];before=' '.join(rowtext(t,i) for i in range(start,r))
 need(re.search(r'investing(?:\s+activities)?\s*:?',before,re.I),'investing section missing')
 lit,cells=selected(t,r,loc['logical_columns']);v=number(lit);need(v<=0,'outflow displayed as inflow')
 heads=header_evidence(t,loc)
 return {'signed_literal':lit,'signed_value_decimal':str(v),'positive_outflow_decimal':str(-v),'scale':loc['scale'],'annual_period_end':loc['period_end'],'row_text':x['text'],'row_text_sha256':sha(x['text'].encode()),'statement_title':loc['statement_title'],'locator':{'format':'html','table':t['index'],'row':r,'logical_columns':loc['logical_columns'],'cells':cells,'header_evidence':heads}},[],allhits,pairs


def pdf_row(doc, page_index, label, index, header_dates):
 pg=doc[page_index];tp=pg.get_textpage()
 try:
  text=tp.get_text_range();lines=[norm(x) for x in text.splitlines()]
  found=[x for x in lines if x.lower().startswith(label.lower())]
  need(len(found)==1,'unique PDF row: '+label)
  tail=found[0][len(label):];tokens=re.findall(r'\([\d,]+\)|-?[\d,]+',tail)
  need(len(tokens)==len(header_dates),'PDF annual column count '+repr(found))
  values=[number(x) for x in tokens]
  # Geometry independent of producer: search exact source label, select right-hand
  # band at its vertical center, then require the complete annual numeric row.
  search=tp.search(label,match_case=False);candidates=[]
  while True:
   hit=search.get_next()
   if hit is None:break
   start,count=hit
   boxes=[tp.get_charbox(i) for i in range(start,start+count) if tp.get_text_range(i,1).strip()]
   box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
   cy=(box[1]+box[3])/2
   band=tp.get_text_bounded(box[2]+1,cy-5,pg.get_width(),cy+5)
   nums=re.findall(r'\([\d,]+\)|-?[\d,]+',band)
   if len(nums)==len(header_dates):candidates.append((box,cy,band,nums))
  need(len(candidates)==1,'unique numeric PDF geometry '+label)
  box,cy,band,nums=candidates[0]
  need([number(x) for x in nums]==values,'PDF geometry full row differs '+label+repr(band))
  geom=[]
  for i in range(tp.count_chars()):
   ch=tp.get_text_range(i,1)
   if not ch.strip():continue
   b=tp.get_charbox(i);mid=(b[1]+b[3])/2
   if b[0]>box[2] and abs(mid-cy)<5:geom.append({'index':i,'text':ch,'bbox':list(b)})
  need(geom,'PDF numeric geometry empty')
  # The years and date labels precede the financial rows in source reading order.
  head=text[:text.lower().index(label.lower())]
  compact=re.sub(r'\s+',' ',head)
  for d in header_dates:
   dt=date.fromisoformat(d);need(str(dt.year) in head and dt.strftime('%B')+' '+str(dt.day)+',' in compact,'PDF year/date header')
  return dict(value_decimal=str(values[index]),signed_literal=tokens[index],page=page_index+1,row_text=found[0],row_text_sha256=sha(found[0].encode()),label_bbox=box,numeric_band=norm(band),numeric_char_geometry=geom,financial_column_index=index,header_dates=header_dates)
 finally:tp.close();pg.close()

def pdf_pair(doc,field):
 if field is None:return {'checked':False,'reason':'inherited_field_missing'}
 loc=field['locator'];need(loc['format']=='pdf','PDF locator')
 ans=pdf_row(doc,loc['page_1based']-1,loc['row_label'],loc['financial_column_index_0based'],loc['header_dates'])
 need(Decimal(ans['value_decimal'])==Decimal(field['value_decimal']),'inherited PDF amount mismatch')
 pg=doc[loc['page_1based']-1];tp=pg.get_textpage();txt=norm(tp.get_text_range());tp.close();pg.close()
 need(loc['statement_title'].lower() in txt.lower() and re.search(r'\(In millions',txt,re.I),'PDF title/unit')
 if 'cash' in loc['statement_title']:need('Twelve Months Ended' in txt,'PDF annual header')
 return dict(ans,checked=True,scale=field['scale'],annual_period_end=loc['period_end'])

def pdf_extract(raw,event,pdfium):
 doc=pdfium.PdfDocument(raw)
 try:
  cfo=event['fields']['current']['operating_cash_flow'];loc=cfo['locator']
  ans=pdf_row(doc,loc['page_1based']-1,'Payments for acquisition of property, plant and equipment',0,loc['header_dates'])
  v=Decimal(ans['value_decimal']);need(v<=0,'PDF positive cash sign')
  pg=doc[loc['page_1based']-1];tp=pg.get_textpage();txt=norm(tp.get_text_range());tp.close();pg.close()
  need('Twelve Months Ended' in txt and '(In millions)' in txt and 'CONSOLIDATED STATEMENTS OF CASH FLOWS' in txt.upper(),'PDF annual statement')
  a=txt.index('Investing activities:');b=txt.index('Payments for acquisition of property, plant and equipment');c=txt.index('Financing activities:');need(a<b<c,'PDF investing placement')
  ans.update(signed_value_decimal=str(v),positive_outflow_decimal=str(-v),scale='1000000',annual_period_end=event['periods']['current']['period_end'],statement_title='CONSOLIDATED STATEMENTS OF CASH FLOWS')
  pairs={}
  for period,name in [('current','operating_cash_flow'),('current','total_assets'),('comparison','total_assets')]:
   field=event['fields'][period].get(name)
   need(field is not None and field['source']['sha256']==sha(raw),'PDF paired source identity')
   pairs[period+'_'+name]=pdf_pair(doc,field)
  return ans,[],[],pairs
 finally:doc.close()

def next_session(release,sessions):
 i=bisect_right(sessions,release);need(i<len(sessions),'calendar horizon');return sessions[i]

def expected():
 scope=json.loads(pinned(SCOPE,SCOPE_SHA))
 for p,h in scope['input_sha256'].items():pinned(p,h)
 events={e['event_id']:e for e in load(DATA)['events']};need(len(events)==44,'inherited44')
 sessions=json.loads(pinned(SESSIONS,SESSIONS_SHA))['sessions'];need(sessions==sorted(set(sessions)),'session axis')
 pypdf,pdfium,rr=runtime();out=[];sources={}
 for target in scope['prospective_targets']:
  e=events[target['event_id']];src=target['source'];raw=pinned(src['path'],src['sha256']);sources[src['path']]=src['sha256'];receipt=src['receipt'];pinned(receipt['path'],receipt['sha256']);sources[receipt['path']]=receipt['sha256']
  need(e['cash_quality_admission']['admitted']==target['inherited_admitted'],'inherited admission')
  available=next_session(e['release_date'],sessions);need(available==e['available_from']==target['inherited_available_from'],'strict next session')
  if raw.startswith(b'%PDF'):cap,holds,evidence,pairs=pdf_extract(raw,e,pdfium)
  else:cap,holds,evidence,pairs=html_extract(raw,e)
  # Original assets occasionally have a distinct pinned document: audit that raw source.
  for period,name in [('current','operating_cash_flow'),('current','total_assets'),('comparison','total_assets')]:
   k=period+'_'+name;f=e['fields'][period].get(name)
   if f and not pairs[k]['checked']:
    fs=f['source'];fb=pinned(fs['path'],fs['sha256']);sources[fs['path']]=fs['sha256'];pinned(fs['receipt']['path'],fs['receipt']['sha256']);sources[fs['receipt']['path']]=fs['receipt']['sha256']
    if fb.startswith(b'%PDF'):
     doc=pdfium.PdfDocument(fb);pairs[k]=pdf_pair(doc,f);doc.close()
    else:hh=Tables();hh.feed(fb.decode('utf-8'));pairs[k]=html_pair(hh,f)
  if cap:need(330<=e['periods']['current']['days_inclusive']<=400,'current annual duration')
  inherited=e['cash_quality_admission']['reasons'];verified=cap is not None
  out.append(dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],release_date=e['release_date'],available_from=available,inherited_admitted=target['inherited_admitted'],inherited_hold_reasons=inherited,review_passed=True,cash_ppe_verified=verified,extraction_verified=verified,current_ppE_verified=verified,source_sha256=src['sha256'],source=src,positive_outflow_decimal=cap['positive_outflow_decimal'] if cap else None,signed_value_decimal=cap['signed_value_decimal'] if cap else None,scale=cap['scale'] if cap else None,annual_period_end=cap['annual_period_end'] if cap else None,hold_reasons=inherited+holds,capex_evidence=cap,held_candidate_evidence=evidence,paired_cfo_verified=pairs['current_operating_cash_flow']['checked'],paired_assets_verified=pairs['current_total_assets']['checked'] and pairs['comparison_total_assets']['checked'],paired_original_fields=pairs,fact_admitted=False))
  print(e['event_id'],verified,cap['positive_outflow_decimal'] if cap else holds,flush=True)
 need(len(out)==44 and sum(e['inherited_admitted'] for e in out)==31,'scope census')
 publish('EXPECTED.json',dict(schema='capex193_independent_original_source_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_only=True,ratios_factors_returns_calculated=False,scope_sha256=SCOPE_SHA,calendar_sha256=SESSIONS_SHA,code_sha256=sha(Path(__file__).read_bytes()),runtime=rr,source_pins=sources,events=out))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pdf-scan','expected','compare']);args=ap.parse_args()
 scope=json.loads(pinned(SCOPE,SCOPE_SHA))
 for p,h in scope['input_sha256'].items():pinned(p,h)
 if args.action=='pdf-scan':
  pypdf,pdfium,rr=runtime();out=[]
  for t in scope['prospective_targets']:
   raw=pinned(t['source']['path'],t['source']['sha256'])
   if not raw.startswith(b'%PDF'):continue
   rd=pypdf.PdfReader(io.BytesIO(raw));doc=pdfium.PdfDocument(raw);pages=[]
   for i,pp in enumerate(rd.pages):
    pptext=pp.extract_text(extraction_mode='layout');pg=doc[i];tp=pg.get_textpage();ft=tp.get_text_range()
    pages.append({'page':i+1,'pypdf_layout':pptext,'pdfium_text':ft});tp.close();pg.close()
   doc.close();out.append({'event_id':t['event_id'],'source_sha256':t['source']['sha256'],'pages':pages})
  publish('PDF_DIRECT_SCAN.json',{'runtime':rr,'events':out});print('scanned',len(out))
 elif args.action=='expected':expected()
 else:raise ValueError('not implemented yet')
if __name__=='__main__':main()
