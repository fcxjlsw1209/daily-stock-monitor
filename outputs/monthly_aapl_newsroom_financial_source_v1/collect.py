# -*- coding: utf-8 -*-
"""Fetch explicit financial-statement PDFs from cached Apple Newsroom articles."""
from pathlib import Path
from urllib.parse import urljoin,urlparse
from bs4 import BeautifulSoup
import json
from source_client import P,fetch,sha,save
ROOT=P.parent.parent;E=ROOT/'outputs/monthly_earnings_repair/apple/events.json';R=ROOT/'outputs/monthly_sec_revenue_data_v3/RECORDS.json'
def emit(p,x):save(p,(json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode())
events=json.loads(E.read_bytes());records=json.loads(R.read_bytes());targets=[]
for e in events:
 p=ROOT/e['evidence_path'];raw=p.read_bytes();assert sha(raw)==e['evidence_sha256'];s=BeautifulSoup(raw,'html.parser');urls=[]
 for a in s.find_all('a',href=True):
  u=urljoin(e['source_url'],a['href']);name=urlparse(u).path.lower()
  if ('financial' in name or 'datasum' in name) and name.endswith('.pdf'):
   assert urlparse(u).hostname=='www.apple.com';row=dict(url=u,href=a['href'],label=a.get_text(' ',strip=True))
   if row not in urls:urls.append(row)
 preferred=[x for x in urls if 'financial' in x['url'].lower()]
 urls=preferred or urls
 assert len(urls)==1,(e['release_date'],urls)
 idx=[i for i,r in enumerate(records) if r['symbol']=='AAPL' and r['catalog'].get('selected') and r['catalog']['selected'].get('report_date')==e['fiscal_period_end_actual']]
 mp=p.with_suffix('.meta.json');m=json.loads(mp.read_bytes());assert m['http_status']==200 and m['sha256']==sha(raw) and m['size']==len(raw) and m['source_url']==e['source_url']
 targets.append(dict(url=urls[0]['url'],anchor=urls[0],release_url=e['source_url'],release_date=e['release_date'],report_date=e['fiscal_period_end_actual'],release_path=e['evidence_path'],release_sha256=sha(raw),attachment_role=('financial_statement' if preferred else 'early_data_summary_boundary_support_only'),release_receipt_path=str(mp.relative_to(ROOT)),release_receipt_sha256=sha(mp.read_bytes()),record_indices=idx))
assert len(targets)==47 and len({t['url'] for t in targets})==47
emit(P/'PLAN.json',dict(schema_version=1,scope='Original financial statement PDFs plus four early data-summary attachments; latter are not presumed to contain financial statements. No semantic admission',events_path=str(E.relative_to(ROOT)),events_sha256=sha(E.read_bytes()),records_path=str(R.relative_to(ROOT)),records_sha256=sha(R.read_bytes()),collector_sha256=sha(Path(__file__).read_bytes()),client_sha256=sha((P/'source_client.py').read_bytes()),targets=targets,strategy_trials=0))
entries=[]
try:
 for t in targets:
  m,p=fetch(t['url']);assert p.read_bytes().startswith(b'%PDF-'),t['url'];entries.append(dict(target=t,receipt=m,path=str(p.relative_to(ROOT)),semantic_verified=False));print(t['report_date'],m['status'],m['decoded_bytes'],flush=True)
except Exception as exc:
 emit(P/'INCOMPLETE.json',dict(status='stopped',processed=len(entries),error_type=type(exc).__name__,error=str(exc),strategy_trials=0));raise
emit(P/'RESULT.json',dict(schema_version=1,status='all_official_linked_PDFs_downloaded',plan_sha256=sha((P/'PLAN.json').read_bytes()),reports=len(entries),candidate_records=len({i for e in entries for i in e['target']['record_indices']}),entries=entries,semantic_verified=False,strategy_trials=0));print('completed',len(entries),flush=True)
