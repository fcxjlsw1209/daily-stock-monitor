"""Same-quarter linked original statements as explicit actual reporting-date evidence."""
import json,re,datetime as dt
from functools import lru_cache
import core as c
@lru_cache(maxsize=None)
def supplements(s):
 p=c.HERE/s/('release_queue.json'if s=='BAC' else 'supplement_queue.json')
 return json.loads(p.read_text()) if p.exists() else []
@lru_cache(maxsize=1024)
def statement_text(path):
 p=c.ROOT/path;body=p.read_bytes();meta=json.loads(p.with_name(p.name+'.meta.json').read_text());assert c.sha(body)==meta['sha256']
 return (c.clean(c.pdf_text(p)) if body.startswith(b'%PDF') else (c.workbook_text(p) if p.suffix=='.xlsx' else None)),meta['sha256']
def candidates(a,s,q,rd):
 from build_events import DATE_RE,date,M,MONTHS
 found=[];items=[a['period_source']] if a.get('period_source') else supplements(s)
 for x in items:
  if 'evidence_path'not in x:continue
  t,checksum=statement_text(x['evidence_path']);assert checksum==x['evidence_sha256']
  if t is None:continue
  head=t[:450]
  fy=rd.year-(1 if q==4 and rd.month<=3 else 0)
  if s=='C':identity=re.search(rf'QUARTERLY FINANCIAL DATA SUPPLEMENT\s*{q}Q\s*(?:20)?{fy%100:02}\b',head,re.I)
  elif s=='T':identity=re.search(rf'(?:{["","First","Second","Third","Fourth"][q]}|{q}(?:st|nd|rd|th))[ -]*QUARTER.{{0,35}}{fy}',head,re.I)
  elif s=='BAC':identity=re.search(rf'Q{q}[- ]{fy%100:02}|{q}Q{fy%100:02}|{["","First","Second","Third","Fourth"][q]}[ -]Quarter.{{0,20}}{fy}',head,re.I)
  else:identity=True
  isxlsx=x['evidence_path'].endswith('.xlsx')
  if isxlsx:
   identity=s=='T' and a['url']in x.get('related_release_urls',[]) and x.get('archive_quarter')==q
  if not identity:continue
  captions=r'(?:Three months|quarter|period)[^.!?]{0,80}ended' if s=='BAC' else r'CONSOLIDATED BALANCE SHEETS?'
  for mt in re.finditer(captions,t,re.I):
   chunk=t[mt.start():mt.end()+250];dates=[]
   for dm in DATE_RE.finditer(chunk):dates.append(date(dm))
   for dm in re.finditer(r'\b(\d{1,2})/(\d{1,2})/(20\d{2})\b',chunk):dates.append(dt.date(int(dm[3]),int(dm[1]),int(dm[2])))
   # Financial tables flatten a month/day header row followed by a year row.
   split=re.search(rf'((?:{M}\.?\s+\d{{1,2}},\s*){{2,6}})((?:20\d{{2}}\s*){{2,6}})',chunk,re.I)
   if split:
    md=list(re.finditer(rf'({M})\.?\s+(\d{{1,2}}),',split[1],re.I));ys=re.findall(r'20\d{2}',split[2])
    if len(md)==len(ys):dates.extend(dt.date(int(y),MONTHS[m[1][:3].lower()],int(m[2]))for m,y in zip(md,ys))
   for end in dates:
    if end.year==fy and end.month==q*3 and 0<(rd-end).days<=115:
     basis='Explicit original current-quarter ended-period table' if s=='BAC' else 'Explicit current reporting column in same-quarter original consolidated balance sheet, with matching quarter/year statement title; no date estimated from calendar'
     if isxlsx:basis='Original issuer workbook balance-sheet header cells give the actual date; issuer archive links this workbook and earnings release within the same quarter package. No calendar date estimated.'
     found.append((end,chunk,basis,x,head))
 return found
