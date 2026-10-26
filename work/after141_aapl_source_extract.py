"""Current annual Apple financial-statement cells; source extraction only."""
import hashlib,io,re,sys
from datetime import date,timedelta
from decimal import Decimal

MONTHS={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
DATE=re.compile(r'('+'|'.join(MONTHS)+r')\s+(\d{1,2}),')
AMOUNT=re.compile(r'\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?')

def require(ok,msg):
 if not ok:raise ValueError(msg)

def norm(text):return ' '.join(text.split())
def digest(raw):return hashlib.sha256(raw).hexdigest()

def decimal_cell(cell):
 require(isinstance(cell,str) and AMOUNT.fullmatch(cell) is not None,'Invalid exact monetary cell')
 require(not (cell.startswith('(') and '-' in cell),'Double negative cell')
 neg=cell.startswith('(')
 require(neg==cell.endswith(')'),'Unbalanced monetary parenthesis')
 number=Decimal(cell.strip('()').replace(',',''))
 require(number.is_finite(),'Nonfinite monetary cell')
 return format(-number if neg else number,'f')

def headers(lines,count):
 found=[]
 for i,line in enumerate(lines[:12]):
  parts=DATE.findall(line)
  if not parts:continue
  years=[]
  for following in lines[i+1:i+4]:
   years=re.findall(r'\b(?:19|20)\d{2}\b',following)
   if years:break
  require(len(parts)==len(years)==count,'Ambiguous year-end headers')
  dates=[date(int(y),MONTHS[m],int(d)).isoformat() for (m,d),y in zip(parts,years)]
  found.append((dates,i+1,norm(line),norm(following)))
 require(len(found)==1,'Unique financial date-header line required')
 return found[0]

def amount_row(lines,label,count):
 found=[]
 for index,line in enumerate(lines):
  m=re.fullmatch(r'\s*'+r'\s*'.join(re.escape(c) for c in label if not c.isspace())+r'\s+(.+)',line,re.I)
  if m:found.append((index,line,m[1]))
 require(len(found)==1,'Unique exact row required: '+label)
 index,line,tail=found[0]
 require('%' not in tail,'Percentage row not an amount')
 tail=tail.replace('/dollarsign','$')
 cells=tail.replace('$','').split()
 require(len(cells)==count,'Exact number of financial columns required')
 for cell in cells:decimal_cell(cell)
 return cells,index+1,line

def parse_layout_pages(texts,spec):
 require(len(texts)>=2,'Both statement pages required')
 first,second=texts[:2];op=first.splitlines();bs=second.splitlines()
 require('Apple Inc.' in first and 'Apple Inc.' in second,'Original issuer heading required')
 require('CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS' in first and 'CONDENSED CONSOLIDATED BALANCE SHEETS' in second,'Original consolidated financial statements required')
 require('unaudited' in first.lower() and 'unaudited' in second.lower(),'Unaudited statement identity required')
 require(not re.search(r'NON[ -]?GAAP|RECONCILIATION',first+' '+second,re.I),'Non-GAAP/reconciliation page excluded')
 for text in [first,second]:
  require(re.search(r'\(In millions,',text) is not None,'Explicit in-millions statement unit required')
  require('$' in text or '/dollarsign' in text,'Monetary dollar label required')
 header=''.join(op[:9]).replace(' ','')
 require('ThreeMonthsEnded' in header and 'TwelveMonthsEnded' in header and header.index('ThreeMonthsEnded')<header.index('TwelveMonthsEnded'),'Separate quarter and annual duration columns required')
 opdates,opline,opdate,opyears=headers(op,4)
 bsdates,bsline,bsdate,bsyears=headers(bs,2)
 end=spec['fiscal_year_end']
 require(opdates[2]==end and opdates[0]==end,'Current annual end mismatch')
 require(opdates[1]==opdates[3]==bsdates[1] and bsdates[0]==end,'Same-release year-end pairing mismatch')
 prior=bsdates[1];start=(date.fromisoformat(prior)+timedelta(days=1)).isoformat()
 days=(date.fromisoformat(end)-date.fromisoformat(start)).days+1
 require(330<=days<=400,'Nonannual boundary interval')
 require(end<spec['release_date'],'Publication precedes annual end')
 gp,gpline,gpraw=amount_row(op,'Gross margin',4)
 assets,aline,araw=amount_row(bs,'Total assets',2)
 require(Decimal(decimal_cell(assets[0]))>0,'Nonpositive total assets')
 def item(label,cells,column,page,line,raw,dates,head):
  return dict(measure_label=label,displayed_amount=cells[column],amount_decimal=decimal_cell(cells[column]),scale=1000000,currency_literal='$',ISO_currency_verified=False,date=end,
   locator=dict(page_1based=page,line_1based=line,financial_column_index_0based=column),
   all_row_cells=cells,row_text=raw,row_text_sha256=digest(raw.encode()),header_dates=dates,date_header_lines=head)
 return dict(symbol='AAPL',fiscal_year=spec['fiscal_year'],gross_amount=item('Gross margin',gp,2,1,gpline,gpraw,opdates,[opdate,opyears]),
  total_assets=item('Total assets',assets,0,2,aline,araw,bsdates,[bsdate,bsyears]),
  annual_period=dict(start=start,end=end,days_inclusive=days,prior_year_end_explicit=prior,start_is_direct_quote=False,derivation='day_after_prior_year_end_printed_in_same_balance_sheet'),
  annual_duration_literal='Twelve Months Ended',fact_admitted=False,strategy_admitted=False,
  review_flags=['field_semantic_and_identity_admission_pending','annual_start_derived_not_verbatim','first_public_version_not_certified','original_dollar_label_not_ISO_certification'],
  source_representation='current_retrieval_original_issuer_PDF',statement_context='Unaudited condensed consolidated statements of operations and balance sheets')

def boxes(textpage,term):
 search=textpage.search(term,match_case=False);out=[]
 while True:
  hit=search.get_next()
  if hit is None:break
  index,count=hit
  cs=[textpage.get_charbox(j) for j in range(index,index+count)]
  out.append(dict(index=index,count=count,bbox=[min(c[0]for c in cs),min(c[1]for c in cs),max(c[2]for c in cs),max(c[3]for c in cs)],text=textpage.get_text_range(index,count)))
 search.close();return out

def attach_geometry(row,textpage):
 label_hits=boxes(textpage,row['measure_label'])
 require(len(label_hits)==1,'Unique PDFium financial-row label required')
 label=label_hits[0];cy=(label['bbox'][1]+label['bbox'][3])/2
 matches={}
 for value in set(row['all_row_cells']):
  for found in boxes(textpage,value):
   by=(found['bbox'][1]+found['bbox'][3])/2
   if abs(by-cy)<4 and found['bbox'][0]>label['bbox'][2]:matches[found['index']]=found
 ordered=sorted(matches.values(),key=lambda b:b['bbox'][0])
 require(len(ordered)==len(row['all_row_cells']),'PDFium same-line numeric cell count mismatch')
 require([norm(x['text']) for x in ordered]==row['all_row_cells'],'Two PDF extractors disagree on financial row')
 current=ordered[row['locator']['financial_column_index_0based']]
 row['locator'].update(pdfium_label_bbox=label['bbox'],pdfium_cell_bbox=current['bbox'],pdfium_char_index=current['index'],pdfium_char_count=current['count'])
 row['independent_PDFium_cell_text_match']=True
 row['pdfium_all_row_cells']=ordered

def extract(raw,spec,pdf_runtime_path):
 if pdf_runtime_path not in sys.path:sys.path.append(pdf_runtime_path)
 from pypdf import PdfReader
 import pypdfium2 as pdfium
 require(raw.startswith(b'%PDF-'),'Original PDF signature required')
 reader=PdfReader(io.BytesIO(raw));texts=[reader.pages[i].extract_text(extraction_mode='layout') for i in range(min(2,len(reader.pages)))]
 result=parse_layout_pages(texts,spec);doc=pdfium.PdfDocument(raw)
 for field,page in [('gross_amount',0),('total_assets',1)]:
  obj=doc[page];tp=obj.get_textpage();attach_geometry(result[field],tp);tp.close();obj.close()
 doc.close()
 result['source_raw_sha256']=digest(raw);result['PDF_page_count']=len(reader.pages)
 result['reconstructed_page_text_sha256']=[digest(t.encode())for t in texts]
 return result
