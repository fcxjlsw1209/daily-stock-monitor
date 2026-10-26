"""Read only the fixed KO/LRCX census originals; emit no monetary cell values."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,io,json,re
from pathlib import Path
from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid,external_context
import after150_next_source_context as old
from cashflow_source_preflight_v1 import configure_pdf
from after149_batch1_admission import first_session,normalize_sessions

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/after151_next_source_feasibility.json'
CENSUS='outputs/monthly_financial_coverage_census_v1/CENSUS.json'
CONCEPT='work/after149_coverage_extension_concept.json'
PDFPLAN='outputs/monthly_cashflow_source_preflight_v1/PLAN.json'
LABEL=re.compile(r'^(?:(?:consolidated\s+)?net (?:income|loss|earnings)(?:\s*\(loss\))?(?:\s+attributable\s+to\b[^\d$]*)?|total assets)\s*(?:\(\d+\))?\s*:?$',re.I)
PREFIX=re.compile(r'^(?:(?:consolidated\s+)?net (?:income|loss|earnings)(?:\s*\(loss\))?(?:\s+attributable\s+to\b[^\d$]*)?|total assets)\b',re.I)
TITLE=re.compile(r'(?:condensed\s+)?consolidated\s+(?:(?:condensed|unaudited)\s+)*(?:balance\s+sheets?|statements?\s+of\s+(?:income|operations|earnings|cash\s+flows?))',re.I)
IDENTITY={'KO':re.compile(r'The Coca-Cola Company|Coca-Cola Company|NYSE\s*[:(]\s*KO\)?',re.I),
          'LRCX':re.compile(r'Lam Research Corporation|NASDAQ\s*[:(]\s*LRCX\)?',re.I)}
seen={}
def read(name,expected=None):
 p=ROOT/name
 if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or 'CloudStorage' in p.parts:raise ValueError('nonlocal source')
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if expected and h!=expected:raise ValueError('hash mismatch '+name)
 seen[name]=h;return raw

def masked(s):return re.sub(r'\d[\d,.\s]*','[number]',s)
def meta(s):
 s=old.norm(s)
 return bool(old.HEADER.fullmatch(s) or old.UNIT.fullmatch(s) or
             (len(s)<180 and TITLE.search(s) and not re.search(r'[$%]',s)))

def html_info(raw):
 soup=BeautifulSoup(raw,'html.parser')
 for n in soup(['script','style']):n.decompose()
 text=old.norm(soup.get_text(' ',strip=True));tables=[]
 for ti,t in enumerate(soup.find_all('table')):
  if t.find('table'):continue
  try:rows,grid,physical=table_grid(t)
  except ValueError:continue
  targets=[c for c in physical.values() if LABEL.fullmatch(old.norm(c['text']))]
  if not targets:continue
  first=min(c['row'] for c in targets)
  starts=[c['row'] for c in physical.values() if c['row']<=first and old.BODY.fullmatch(old.norm(c['text']))]
  stop=min(starts) if starts else first
  headers=[old.metadata_cell(c) for c in physical.values() if c['row']<stop and meta(c['text'])]
  outside=[old.norm(s) for s in external_context(t) if meta(s)]
  labels=[dict(row=c['row'],physical_cell=c['physical_cell'],columns=c['columns'],label=old.norm(c['text'])) for c in targets]
  tables.append(dict(table=ti,headers=headers,external_metadata=outside,labels=labels,
                     total_physical_rows=len(rows),dollar_symbol_present='$' in t.get_text()))
 pre=[]
 for pi,p in enumerate(soup.find_all('pre')):
  lines=[]
  for li,line in enumerate(p.get_text().splitlines()):
   line=old.norm(line)
   if meta(line):lines.append(dict(line_1based=li+1,kind='header',text=line))
   elif PREFIX.match(line):lines.append(dict(line_1based=li+1,kind='label_only',text=PREFIX.match(line).group(0)))
  if lines:pre.append(dict(pre_block=pi,lines=lines))
 links=[dict(label=masked(old.norm(a.get_text(' ',strip=True))),url=a.get('href')) for a in soup.find_all('a',href=True)
        if re.search(r'\.pdf(?:$|\?)',a['href'],re.I) or re.search(r'financial statements|financial tables',a.get_text(),re.I)]
 return dict(table_count=len(soup.find_all('table')),statement_candidates=tables,pre_candidates=pre,
             attachment_links=links),text

def pdf_info(raw,reader):
 doc=reader(io.BytesIO(raw));pages=[];alltext=[]
 for pi,page in enumerate(doc.pages):
  text=page.extract_text(extraction_mode='layout') or '';alltext.append(text)
  ls=[]
  for li,line in enumerate(text.splitlines()):
   s=old.norm(line)
   if meta(s):ls.append(dict(line_1based=li+1,kind='header',text=s))
   elif PREFIX.match(s):
    parts=re.split(r'\s{2,}',line.strip())
    ls.append(dict(line_1based=li+1,kind='label_only',text=masked(parts[0])))
   else:
    parts=re.split(r'\s{2,}',line.strip())
    if len(parts)>1 and all(meta(part) for part in parts):
     ls.append(dict(line_1based=li+1,kind='header_columns',texts=parts))
  if ls:pages.append(dict(page_1based=pi+1,metadata_lines=ls))
 return dict(page_count=len(doc.pages),text_extractable=any(alltext),page_metadata=pages,
             value_column_geometry_verified=False,visual_review_pending=True),old.norm(' '.join(alltext))

def main():
 census=json.loads(read(CENSUS,old.EXPECTED_CENSUS));read(CONCEPT)
 sessions=normalize_sessions(json.loads(read('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')))
 pdfplan=json.loads(read(PDFPLAN));reader,_=configure_pdf(pdfplan)
 for p,h in pdfplan['pdf_runtime_hashes'].items():
  # These existing pinned local runtime files are checked by configure_pdf.
  seen[p]=h
 docs=[]
 for row in census['annual_documents']:
  if row['symbol'] not in IDENTITY:continue
  raw=read(row['evidence_path'],row['evidence_sha256'])
  path=Path(row['evidence_path'])
  rp=str(path)+'.meta.json' if row['symbol']=='KO' else str(path.with_suffix('.meta.json'))
  receipt=json.loads(read(rp))
  if receipt.get('sha256')!=row['evidence_sha256'] or receipt.get('url')!=row['source_url']:
   raise ValueError('receipt binding '+row['event_id'])
  info,text=(pdf_info(raw,reader) if raw.startswith(b'%PDF') else html_info(raw))
  terms=sorted(set(m.group(0).lower() for m in old.ACCOUNTING.finditer(text)))
  snippets=[]
  for m in old.ACCOUNTING.finditer(text):
   snippet=masked(text[max(0,m.start()-65):m.end()+175])
   if snippet not in snippets:snippets.append(snippet)
  docs.append(dict(row,receipt=dict(path=rp,sha256=seen[rp],status=receipt.get('status',receipt.get('http_status')),
                    retrieved_at=receipt.get('retrieved_at'),url_and_sha_match=True),
                    identity_anchors=sorted(set(m.group(0) for m in IDENTITY[row['symbol']].finditer(text))),
                    pinned_next_session=first_session(row['publication_date'],sessions),
                    available_session_verified=first_session(row['publication_date'],sessions)==row['available_session'],
                    context=info,accounting_terms=terms,accounting_context_numbers_redacted=snippets,
                    values_extracted=0,fact_admitted=False))
 assert len(docs)==32
 for name in ('work/after151_next_source_feasibility.py','work/after150_next_source_context.py',
              'work/after141_amd_source_extract.py','work/cashflow_source_preflight_v1.py','work/after149_batch1_admission.py'):read(name)
 result=dict(schema_version=1,stage='bounded_source_metadata_feasibility_not_admission',
  created_utc=datetime.now(timezone.utc).isoformat(),source_scope=['KO','LRCX'],
  documents=docs,counts=dict(catalog_events=len(docs),by_symbol=dict(Counter(d['symbol'] for d in docs)),
  pdf=sum(d['format']=='pdf' for d in docs),html=sum(d['format']!='pdf' for d in docs)),
  interpretation={
   'KO':'FY2010 PRE and FY2011-2016 HTML contain statement leads; total/consolidated NI must be distinguished from NI attributable to shareowners. FY2017-2025 current pages lack the selected tables but link issuer PDF/annual reports: page without a table does not establish that the report is missing. Attachment bytes and receipts have not been inventoried here.',
   'LRCX':'FY2011-2012 PRE, FY2013-2016 HTML, FY2017-2026 extractable PDFs. Formal annual operations and balance-sheet headings must be tied to the same current publication, with quarterly columns excluded. Fiscal-year starts require explicitly reviewed same-report prior-end-plus-one boundary assumptions.',
   'next_step':'Freeze source-only scope before numeric extraction. Use strict HTML geometry for standard tables; retain PRE and PDF gaps until separate layout readers and independent visual cell review pass. Preserve all32 latest-event sentinels.',
   'units':'Table-local scale and common dollar basis are source leads only; no FX conversion or certified ISO currency inference.',
   'known_time':'Current publication governs all columns. Out-of-range pinned calendar dates remain unverified and no availability is extrapolated.'},
  input_hashes=seen,financial_values_extracted=0,ratios_computed=0,new_factors_computed=0,
  orders_or_returns_computed=0,new_admissions=0,new_strategy_trials=0,
  limitations=['Label and date metadata only; physical monetary columns and accounting comparability require future independent review.',
  'All32 events retained. Missing/held newest annual events must block fallback; no annual-event selection based on returns.',
  'No first-public-version, continuous security-master or full correction-history certification.',
  'LRCX PDF metadata is text-readable but exact cell geometry and visual confirmation remain pending.'])
 if OUT.exists():raise ValueError('Preserve existing report')
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps(dict(counts=result['counts'],sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())))
if __name__=='__main__':main()
