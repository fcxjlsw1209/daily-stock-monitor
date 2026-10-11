"""Cross-extract PDF: content-stream names and isolated numeric cap column."""
from pathlib import Path
import re,json,hashlib
import pdfplumber
from pypdf import PdfReader
P=Path(__file__).resolve().parent/'additional_sources/ft_rankings';f=P/'2013_reference.pdf'
# Read the existing parser's literal country list without running its workload.
s=(P.parents[1]/'parse_ft_reference.py').read_text();countries=re.search(r"countries='([^']+)'",s).group(1)
pattern=re.compile(r'^(\d+)\s+(?:(?:\d+|NEW|New|—|-)\s+)?(.+)\s+('+countries+r')\s+(.+?)\s+\d[\d, ]*\.\s*\d')
numeric={}
with pdfplumber.open(f) as pdf:
 for page in pdf.pages:
  words=page.dedupe_chars().extract_words()
  for w in words:
   if w['x0']<70 and w['text'].isdigit() and 1<=int(w['text'])<=500:
    caps=[x['text'] for x in words if 380<x['x0']<426 and abs(x['top']-w['top'])<1 and re.fullmatch(r'[\d,]+\.\d+',x['text'])]
    if len(caps)==1:numeric[int(w['text'])]=float(caps[0].replace(',',''))*1e6
rows=[];fail=[]
for page in PdfReader(f).pages:
 text=page.extract_text()
 # Wraps inside a row are joined; a new numeric rank starts a new record.
 chunks=re.split(r'(?m)^(?=\d{1,3}\s)',text)
 for chunk in chunks:
  line=' '.join(chunk.split());m=pattern.match(line)
  if not m:continue
  rank,name,country,sector=m.groups();rank=int(rank)
  if rank not in numeric:fail.append(rank);continue
  rows.append(dict(reference_date='2013-03-28',global_rank=rank,company_name_as_printed=name,country=country,sector_as_printed=sector,market_cap_usd=numeric[rank],source_file=f.name,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),us_exchange_eligibility_verified=False,daily_membership_usable=False))
rows.sort(key=lambda r:r['global_rank'])
report=dict(rows=len(rows),numeric_rows=len(numeric),missing_ranks=sorted(set(range(1,501))-{r['global_rank'] for r in rows}),complete=len(rows)==500 and len({r['global_rank'] for r in rows})==500,cap_descending=all(rows[i]['market_cap_usd']>=rows[i+1]['market_cap_usd'] for i in range(len(rows)-1)))
(P/'2013_repaired_rankings.json').write_text(json.dumps(rows,indent=2));(P/'2013_repair_validation.json').write_text(json.dumps(report,indent=2));print(report)
