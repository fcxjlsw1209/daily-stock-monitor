"""Join original dates and original financial tables, keeping each proof separate."""
from collect import *
def merge(k,added,pending):
 old=json.loads((HERE/k/'events.json').read_text()) if (HERE/k/'events.json').exists() else []
 by={(e['fiscal_year'],e['fiscal_quarter']):e for e in old}
 for e in added:
  key=e['fiscal_year'],e['fiscal_quarter']
  if key in by:
   if any(by[key][f]!=e[f] for f in ['release_date','fiscal_period_end_actual']):raise ValueError('Conflicting source '+k+str(key))
   # Prefer all-full-byte evidence if both prove the same event.
   if 'public_index' in by[key].get('source_class','') and 'public_index' not in e.get('source_class',''):by[key]=e
  else:by[key]=e
 dump(HERE/k/'events.json',sorted(by.values(),key=lambda e:e['release_date']))
 dump(HERE/k/'JOIN_PENDING.json',pending)
 print(k,len(by),'join pending',len(pending),flush=True)
def intel_notices():
 out=[];pending=[]
 for mp in (HERE/'INTC/raw').glob('*.meta.json'):
  if mp.name.endswith('.text.meta.json'):continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200 or '/all-sec-filings/content/' not in m['url']:continue
  t=clean(extract(p))
  dm=re.search(r'On ('+DATE+r')\s*,?\s*Intel Corporation.{0,100}?issued a press release announcing.{0,70}?'+ORD+r' quarter(?: and full year)? ended ('+DATE+r')',t,re.I)
  if not dm:
   actual=re.search(r'On ('+DATE+r')\s*,?\s*Intel Corporation.{0,100}?issued a press release announcing the financial results of its fiscal quarter ended ('+DATE+r')',t,re.I)
   label=re.search(r'Intel.s Q([1-4]) (20\d{2}) earnings conference',t,re.I)
   if actual and label:
    release=iso(actual.group(1));period=iso(actual.group(2));fy=int(label.group(2));q=int(label.group(1));pd=dt.date.fromisoformat(period)
    ref=dt.date(fy,q*3,calendar.monthrange(fy,q*3)[1]);assert abs((pd-ref).days)<=10
    rr=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=actual.group(),period_excerpt=actual.group(),identity_excerpt=label.group())
    e=event('INTC',p,m,rr,'issuer_original_regulatory_notice_of_release_issuance_full_bytes');e['release_date_basis']='Original Item 2.02 explicitly identifies the actual date of press-release issuance; conference label identifies FY/Q only and is never a release clock.';out.append(e);continue
  if not dm:
   am=re.search(r'financial information for Intel Corporation.{0,60}?quarter(?: and fiscal year)? ended ('+DATE+r').{0,210}?press release (?:of |issued on )?('+DATE+r')',t,re.I)
   if not am:pending.append(dict(source_url=m['url'],reason='Explicit actual release date and period statement absent'));continue
   period=iso(am.group(1));release=iso(am.group(2));pd=dt.date.fromisoformat(period)
   _,fy,q=min((abs((pd-dt.date(y,mo,calendar.monthrange(y,mo)[1])).days),y,mo//3) for y in [pd.year-1,pd.year,pd.year+1] for mo in [3,6,9,12])
   rr=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=am.group(),period_excerpt=am.group(),identity_excerpt=am.group())
   e=event('INTC',p,m,rr,'issuer_original_regulatory_notice_of_release_issuance_full_bytes');e['release_date_basis']='Issuer Item 2.02 explicitly identifies the dated earnings press release and financial period; no SEC filing clock used';out.append(e);continue
  release=iso(dm.group(1));period=iso(dm.group(3));q=QMAP[dm.group(2).lower()]
  pd=dt.date.fromisoformat(period);fy=pd.year-(q==4 and pd.month==1)
  r=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=dm.group(),period_excerpt=dm.group(),identity_excerpt=dm.group())
  e=event('INTC',p,m,r,'issuer_original_regulatory_notice_of_release_issuance_full_bytes')
  e['release_date_basis']='Issuer Item 2.02 explicitly states when the actual earnings release was issued and the actual fiscal period; no filing or acceptance date used.'
  out.append(e)
 merge('INTC',out,pending)
def tesla():
 out=[];pending=[];dateproof={}
 for p in (HERE/'TSLA/public_index_observations').glob('release_*.json'):
  j=json.loads(p.read_text());t=clean(j['rendering'])
  if re.search(r'will release',t,re.I):continue
  dm=re.search(r'(PALO ALTO|AUSTIN)[A-Za-z .,&()-]{0,70}('+DATE+r')',t,re.I)
  if dm:d=iso(dm.group(2));de=dm.group()
  else:
   dm=re.search(r'(PALO ALTO)[^/]{0,60}(\d{2}/\d{2}/\d{2})',t,re.I)
   if not dm:continue
   d=dt.datetime.strptime(dm.group(2),'%m/%d/%y').date().isoformat();de=dm.group()
  qm=re.search(ORD+r' quarter(?: and full year| & full year)?(?: of)? (20\d{2})',t,re.I)
  if not qm:continue
  key=int(qm.group(2)),QMAP[qm.group(1).lower()]
  dateproof[key]=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=j['url'],excerpt=de,date=d,timezone='America/Chicago' if 'AUSTIN' in de else 'America/Los_Angeles')
 for mp in (HERE/'TSLA/raw').glob('*.meta.json'):
  if mp.name.endswith('.text.meta.json'):continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200 or not p.read_bytes().startswith(b'%PDF'):continue
  t=clean(extract(p));head=t[:800]
  qm=re.search(r'Q([1-4])(?: and FY)?\s*(20\d{2})',head)
  if qm:q=int(qm.group(1));fy=int(qm.group(2))
  else:
   qm=re.search(ORD+r' Quarter(?: & Full Year)? (20\d{2})',head,re.I)
   if not qm:pending.append(dict(source_url=m['url'],reason='Quarter/year header absent'));continue
   q=QMAP[qm.group(1).lower()];fy=int(qm.group(2))
  proof=dateproof.get((fy,q))
  dm=re.search('('+DATE+r') Dear Fellow Shareholders',head,re.I)
  if dm:
   d=iso(dm.group(1));proof=dict(evidence_path=rel(p),sha256=m['sha256'],source_url=m['url'],excerpt=dm.group(),date=d,timezone='America/Los_Angeles')
  if not proof:pending.append(dict(source_url=m['url'],reason='Original actual publication dateline absent; archive earnings date or webcast date not substituted'));continue
  packed=re.sub(r'\s+','',t)
  candidates=[]
  for hm in re.finditer(r'(?:(?:CONDENSED)?CONSOLIDATED)?(?:STATEMENTSOFOPERATIONS|STATEMENTOFOPERATIONS|BALANCESHEETS|BALANCESHEET|INCOMESTATEMENT)',packed,re.I):
   h=packed[hm.end():hm.end()+2200]
   for z in re.finditer('('+MONTH+r')(\d{1,2}),?(20\d{2})',h,re.I):
    v=iso(z.group(1)+' '+z.group(2)+' '+z.group(3));pd=dt.date.fromisoformat(v)
    if pd.year==fy and pd.month==q*3:candidates.append((v,'Whitespace-compacted original financial table header: '+h[:600]))
  for z in re.finditer(r'(\d{1,2})-([A-Za-z]{3})-(\d{2})',packed):
   v=iso(z.group(2)+' '+z.group(1)+' 20'+z.group(3));pd=dt.date.fromisoformat(v)
   if pd.year==fy and pd.month==q*3:
    candidates.append((v,'Original financial statement date header: '+packed[max(0,z.start()-120):z.end()+100]))
  if not candidates:
   pending.append(dict(source_url=m['url'],reason='Actual current period date absent from financial table header'));continue
  period,pe=max(candidates);release=proof['date']
  if not '2011-01-01'<=release<=CUTOFF or not 0<(dt.date.fromisoformat(release)-dt.date.fromisoformat(period)).days<100:pending.append(dict(source_url=m['url'],reason='Date outside window or invalid lag'));continue
  r=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=proof['excerpt'],period_excerpt=pe,identity_excerpt=qm.group())
  cls='issuer_original_shareholder_letter_full_bytes' if proof['evidence_path']==rel(p) else 'issuer_financial_update_full_bytes_with_public_index_original_release_dateline'
  e=event('TSLA',p,m,r,cls);e['publication_timezone']=proof['timezone'];e['release_date_evidence']=[{k:v for k,v in proof.items() if k not in ['date','timezone']}]
  e['release_date_basis']='Dated original shareholder letter or original actual-results announcement dateline; never archive/CMS/webcast date'
  if 'public_index' in cls:
   e['additional_evidence']=[e['release_date_evidence'][0]];e['evidence_limitation']='Financial update primary evidence is original PDF bytes; publication dateline secondary evidence is a saved public-index rendering, not original release HTML bytes.'
  out.append(e)
 merge('TSLA',out,pending)
if __name__=='__main__':intel_notices();tesla()
