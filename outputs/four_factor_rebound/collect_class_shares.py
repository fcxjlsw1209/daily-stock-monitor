"""Cache filing-cover class counts; do not infer counts from weighted-average EPS."""
import json,re,time,hashlib
from pathlib import Path
import pandas as pd,requests
from lxml import html
from prepare import P
D=P.parent/'dynamic_top100/registry_sec';O=P/'top32_research';C=O/'filings'
def extract(content,cik):
 tree=html.fromstring(content);classes={};dates=set()
 for el in tree.iter():
  if (el.get('name') or '').lower()!='dei:entitycommonstocksharesoutstanding':continue
  contexts=tree.xpath('//*[@id=$v]',v=el.get('contextref'))
  if len(contexts)!=1:continue
  text=''.join(contexts[0].itertext());ds=re.findall(r'\d{4}-\d{2}-\d{2}',text)
  key='A' if 'CommonClassAMember' in text else 'B' if 'CommonClassBMember' in text else None
  if key is None or len(ds)!=1:continue
  value=float(''.join(el.itertext()).replace(',','').replace(' ',''))*10**int(el.get('scale','0'))
  if key in classes and classes[key]!=value:raise ValueError('Class count conflict')
  classes[key]=value;dates.add(ds[0])
 if set(classes)!={'A','B'} or len(dates)!=1:return None
 if cik=='0001067983':
  # Verify class economic ratio in this same filing, rather than using today's terms backwards.
  full=' '.join(tree.itertext())
  if '1/1,500' not in full and '1/1500' not in full:return None
  total=classes['A']*1500+classes['B']
 else:total=classes['A']+classes['B']
 return dict(end=next(iter(dates)),val=total,classes=classes,concept='EntityCommonStockSharesOutstanding')
def main():
 C.mkdir(exist_ok=True);allrows=[];failures=[]
 for cik,symbol in [('0001326801','research_equity_FB'),('0001067983','BRK.B')]:
  d=json.loads((D/(cik+'_submissions.json')).read_text());tables=[d['filings']['recent']]
  for f in sorted((D/'filing_indexes').glob('CIK'+cik+'-submissions-*.json')):tables.append(json.loads(f.read_text()))
  rows={}
  for table in tables:
   for i,form in enumerate(table.get('form',[])):
    date=table['filingDate'][i]
    if form not in ['10-Q','10-K'] or not '2015-01-01'<=date<='2026-10-06':continue
    acc=table['accessionNumber'][i];rows[acc]=(date,table['primaryDocument'][i])
  for acc,(date,doc) in sorted(rows.items(),key=lambda r:r[1][0]):
   target=C/(acc.replace('-','')+'.html');path=f'{int(cik)}/{acc.replace("-","")}/{doc}';url='https://edgar-mirror.sec-api.io/'+path
   if not target.exists():
    try:
     r=requests.get(url,timeout=30)
     if r.status_code==429:raise RuntimeError('429: stop this source for today; Retry-After '+r.headers.get('Retry-After','unspecified'))
     if r.status_code!=200:failures.append(dict(symbol=symbol,date=date,status=r.status_code));continue
     target.write_bytes(r.content);time.sleep(.15)
    except requests.RequestException as e:failures.append(dict(symbol=symbol,date=date,error=str(e)));continue
   row=extract(target.read_bytes(),cik)
   if row:allrows.append(dict(row,symbol=symbol,filed=date,accn=acc,source='https://www.sec.gov/Archives/edgar/data/'+path,retrieved_from=url,file=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
   else:failures.append(dict(symbol=symbol,date=date,error='No unambiguous inline class counts / rights evidence'))
   (O/'CLASS_SHARE_EVIDENCE.json').write_text(json.dumps(dict(rows=allrows,unresolved=failures),indent=2)+'\n')
   print(symbol,date,'parsed' if row else 'unresolved',flush=True)
if __name__=='__main__':main()
