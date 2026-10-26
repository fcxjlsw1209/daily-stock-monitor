"""Read-only pinned source inventory, no factors or component aggregation."""
from pathlib import Path
import json,hashlib,re
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
SCOPE='work/working_capital194_source_scope.json'
SHA='f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be'
def digest(b):return hashlib.sha256(b).hexdigest()
def norm(s):return re.sub(r'\s+',' ',s).strip()
def main():
 b=(ROOT/SCOPE).read_bytes();assert digest(b)==SHA
 for e in json.loads(b)['prospective_targets']:
  if e['symbol'] not in ('NVDA','AMD'):continue
  b=(ROOT/e['source']['path']).read_bytes();assert digest(b)==e['source']['sha256']
  soup=BeautifulSoup(b,'html.parser'); tables=soup.find_all('table');loc=e['current_cfo_locator']
  print('\nEVENT',e['event_id'],'CFO',loc)
  for ti,t in enumerate(tables):
   rows=[r for r in t.find_all('tr') if r.find_parent('table') is t]
   texts=[norm(r.get_text(' ',strip=True)) for r in rows]
   if loc is not None and ti==loc['table_index_zero_based']:
    for ri,row in enumerate(rows):
     print('ROW',ri,texts[ri],'CELLS',[(j,norm(c.get_text(' ',strip=True)),c.get('colspan',1)) for j,c in enumerate(row.find_all(['td','th'],recursive=False))])
   elif loc is None:
    if any(re.search(r'cash flow|operating activities',x,re.I) for x in texts):
     print('OTHER TABLE',ti,'HEAD',texts[:6])
     for ri,x in enumerate(texts):
      if re.search(r'accounts? receivable|inventor|accounts? payable|cash flow|operating activit',x,re.I):print('ROW',ri,x)
if __name__=='__main__':main()
