"""Extract dated primary SPDR sector-fund holdings; no security-name inference."""
from pathlib import Path
import json,re,hashlib
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent/'sources/spdr_sectors'
SECTORS={'Consumer Discretionary':'Consumer Cyclical','Consumer Staples':'Consumer Defensive','Energy':'Energy','Financial':'Financial Services','Health Care':'Healthcare','Industrial':'Industrials','Materials':'Basic Materials','Technology':'Technology','Utilities':'Utilities','Real Estate':'Real Estate','Communication Services':'Communication Services'}
PAT=re.compile(r'(?:The\s+)?('+ '|'.join(SECTORS)+r') Select Sector SPDR(?:®)? (?:Fund|ETF)',re.I)
def compact(t):return ' '.join(t.split())
def parse(html):
 soup=BeautifulSoup(html,'html.parser');rows=[];last_key=None;last_industry=None
 for i,table in enumerate(soup.find_all('table')):
  if table.find('table'):continue
  text=compact(table.get_text(' ',strip=True))
  if 'Shares' not in text or 'Value' not in text:continue
  before=compact(' '.join(reversed([str(x) for x in table.find_all_previous(string=True,limit=120)])))
  match=list(PAT.finditer(before))
  if match and re.search(r'SCHEDULE OF INVESTMENTS',before,re.I):
   fund=match[-1].group(1);key=next(k for k in SECTORS if k.lower()==fund.lower())
   industry=last_industry if key==last_key else None
  elif last_key and text.startswith('Security Description'):
   key=last_key;industry=last_industry
  else:continue
  sector=SECTORS[key];last_key=key
  for j,tr in enumerate(table.find_all('tr')):
   cells=[compact(x.get_text(' ',strip=True)) for x in tr.find_all(['td','th'],recursive=False)]
   cells=[x for x in cells if x and x not in ['$',')','(','%']]
   if not cells:continue
   company=cells[0]
   if re.match(r'(TOTAL COMMON|SHORT.TERM|MONEY MARKET|REPURCHASE|OTHER ASSETS)',company,re.I):break
   if re.search(r'—|--| - ',company) and '%' in company:
    head=re.split(r'—|--| - ',company)[0].strip()
    if not re.search(r'COMMON STOCK|TOTAL INVEST',head,re.I):industry=head
    continue
   numbers=sum(bool(re.fullmatch(r'[\d,.]+',c)) for c in cells[1:])
   if numbers<2 or not re.search(r'[A-Za-z]',company) or re.match(r'(Security Description|Total |Cost|Net Assets)',company,re.I):continue
   assigned=sector
   if sector=='Technology' and industry and 'TELECOMMUNICATION' in industry.upper():assigned='Communication Services'
   rows.append(dict(company=company,sector=assigned,industry=industry,fund=key,table=i,row=j))
  last_industry=industry
 return rows
def parse_plain(text):
 rows=[];key=None;active=False;industry=None;pending=''
 for i,line in enumerate(text.splitlines(),1):
  t=compact(line);m=PAT.fullmatch(t)
  if m:
   key=next(k for k in SECTORS if k.lower()==m.group(1).lower());active=False;pending='';continue
  if key and t.startswith('SCHEDULE OF INVESTMENTS'):active=True;continue
  if not active:continue
  if re.match(r'(TOTAL COMMON|SHORT.TERM|MONEY MARKET|REPURCHASE|OTHER ASSETS)',t,re.I):active=False;continue
  if '--' in t and '%' in t:
   head=t.split('--')[0].strip()
   if 'COMMON STOCK' not in head:industry=head
   pending='';continue
  m=re.match(r'(.+?)\.{2,}\s*([\d,]+)\s+\$?\s*([\d,]+)',t)
  if m:
   company=compact(pending+' '+m.group(1));pending=''
   sector=SECTORS[key]
   if sector=='Technology' and industry and 'TELECOMMUNICATION' in industry:sector='Communication Services'
   rows.append(dict(company=company,sector=sector,industry=industry,fund=key,source_line=i));continue
  if t and re.match(r'[A-Za-z]',t) and not re.match(r'(SECURITY|SHARES|VALUE|SEPTEMBER|COMMON|Cost|See accompanying)',t):pending=t
 return rows

def main():
 reports=[]
 for meta in sorted(P.glob('*_report.provenance.json')):
  m=json.loads(meta.read_text());f=P/(m['year']+'_report.'+m['document'].rsplit('.',1)[-1])
  if hashlib.sha256(f.read_bytes()).hexdigest()!=m['sha256']:raise ValueError('Source changed')
  rows=parse_plain(f.read_text()) if f.suffix=='.txt' else parse(f.read_text());out=dict(source_file=str(f.relative_to(P.parent.parent)),source_sha256=m['sha256'],source_url=m['sec_url'],mirror_url=m['mirror_url'],known_at=m['filed']+'T23:59:59-05:00',observed_as_of=m['as_of'],rows=rows)
  dest=P/(m['year']+'_holdings_extracted.json');dest.write_text(json.dumps(out,indent=2)+'\n');r=dict(year=m['year'],rows=len(rows),funds=sorted({x['fund'] for x in rows}));reports.append(r);print(r,flush=True)
 (P/'extraction_summary.json').write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
