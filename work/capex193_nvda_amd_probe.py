"""Read-only, pinned local HTML source inventory for CAPEX193."""
from pathlib import Path
import json,hashlib,re
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
SCOPE='work/capex193_source_scope.json'
EXPECTED='71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08'
def sha(b): return hashlib.sha256(b).hexdigest()
def norm(s): return re.sub(r'\s+',' ',s).strip()
def main():
 raw=(ROOT/SCOPE).read_bytes(); assert sha(raw)==EXPECTED
 for e in json.loads(raw)['prospective_targets']:
  if e['symbol'] not in ('NVDA','AMD'): continue
  b=(ROOT/e['source']['path']).read_bytes(); assert sha(b)==e['source']['sha256']
  soup=BeautifulSoup(b,'html.parser'); tables=soup.find_all('table'); loc=e['current_cfo_locator']
  print('\nEVENT',e['event_id'],'CFO',None if loc is None else loc['table_index_zero_based'])
  for ti,t in enumerate(tables):
   rows=[r for r in t.find_all('tr') if r.find_parent('table') is t]
   texts=[norm(r.get_text(' ',strip=True)) for r in rows]
   hits=[i for i,x in enumerate(texts) if re.search(r'(?:purchases|additions|payments).{0,40}(?:property|capital|equipment)|capital expenditures',x,re.I)]
   if not hits: continue
   print('TABLE',ti,'HEAD',texts[:5])
   for i in hits: print('ROW',i,texts[i], 'CELLS',[(j,norm(c.get_text(' ',strip=True)),c.get('colspan',1)) for j,c in enumerate(rows[i].find_all(['td','th'],recursive=False))])
if __name__=='__main__': main()
