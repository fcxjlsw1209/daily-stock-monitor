"""Read frozen report prose for accounting review; no financial arithmetic."""
import argparse, hashlib, json, re
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
PLAN='outputs/monthly_cashflow_source_preflight_v1/PLAN.json'
PLAN_SHA='4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--symbol');p.add_argument('--year',type=int);a=p.parse_args()
 raw=(ROOT/PLAN).read_bytes();assert digest(raw)==PLAN_SHA
 plan=json.loads(raw)
 rx=re.compile(r'restat\w*|recast\w*|reclassif\w*|ASU\s+[0-9-]+|FASB|adopt\w*|retrospect\w*|useful lives|accounting (?:standard|estimate|change)|discontinued operations|cash flow presentation',re.I)
 for e in plan['events']:
  if a.symbol and e['symbol']!=a.symbol:continue
  if a.year and e['fiscal_year']!=a.year:continue
  source=(ROOT/e['source']['path']).read_bytes();assert digest(source)==e['source']['sha256']
  if source.startswith(b'%PDF-'):continue
  soup=BeautifulSoup(source,'html.parser')
  for n in soup(['script','style']):n.decompose()
  text=' '.join(soup.get_text(' ',strip=True).split())
  spans=[]
  for m in rx.finditer(text):
   left=max(0,m.start()-170);right=min(len(text),m.end()+350)
   if spans and left<=spans[-1][1]:spans[-1][1]=max(right,spans[-1][1])
   else:spans.append([left,right])
  print(json.dumps(dict(event_id=e['event_id'],path=e['source']['path'],sha256=e['source']['sha256'],snippets=[dict(locator='normalized visible text chars %d..%d'%(l,r),text=text[l:r])for l,r in spans]),ensure_ascii=False))
if __name__=='__main__':main()
