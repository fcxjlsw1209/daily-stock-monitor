"""Specialized admission for original shareholder letters and issuer-hosted 8-K exhibits."""
from collect import *
def period_header(t):
 for m in re.finditer(r'Consolidated (?:Condensed )?Statements? of (?:Income|Operations)|Condensed Consolidated Statements? of (?:Income|Operations)',t,re.I):
  head=t[m.end():m.end()+700]
  if not re.search(r'(?:Three Months|Quarter|Quarters) Ended',head,re.I):continue
  md=re.search('('+MONTH+r')\s+(\d{1,2})\s*,?',head,re.I)
  if not md:continue
  ym=re.search(r'\b20\d{2}\b',head[md.end():])
  if not ym:continue
  return iso(md.group(1)+' '+md.group(2)+' '+ym.group()),head[:max(md.end()+ym.end(),300)]
 packed=re.sub(r'\s+','',t)
 m=re.search(r'Consolidated(?:Condensed)?Statementsof(?:Operations|Income)',packed,re.I)
 if m:
  h=packed[m.end():m.end()+500]
  if re.search(r'ThreeMonthsEnded',h,re.I):
   md=re.search('('+MONTH+r')(\d{1,2}),?',h,re.I)
   if md:
    ym=re.search(r'20\d{2}',h[md.end():])
    if ym:return iso(md.group(1)+' '+md.group(2)+' '+ym.group()),'Whitespace-compacted original table header: '+h[:300]
 raise ValueError('Actual financial statement quarter-end header absent')
def specialized(k,t):
 t=clean(t)
 if k=='NFLX':
  dm=re.match(r'\s*('+MONTH+r')\s*(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(20\d{2})',t)
  if not dm:raise ValueError('Dated shareholder letter header absent')
  if not re.search(r'Fellow\s+[Ss]hareholders?|Dear\s+[Ss]hareholders?',t[:500]) and not ('Summary:' in t[:100] and re.search(r'Netflix',t,re.I)):raise ValueError('Actual shareholder letter absent')
  release=iso(dm.group(1)+' '+dm.group(2)+' '+dm.group(3));de=dm.group(0)
  period,pe=period_header(t)
 elif k=='PLTR':
  dm=re.search(r'On ('+DATE+r'),? Palantir Technologies Inc\..{0,130}?issued a press release announcing its financial results for (?:the )?[^.]{0,100}?ended ('+DATE+r')',t,re.I)
  if not dm:raise ValueError('Explicit issuer statement of release issuance date absent')
  release=iso(dm.group(1));period=iso(dm.group(2));de=pe=dm.group()
  if not re.search(r'Exhibit 99\.1',t,re.I):raise ValueError('Attached original financial exhibit absent')
 else:raise ValueError(k)
 if not '2011-01-01'<=release<=CUTOFF:raise ValueError('Outside publication window')
 pd=dt.date.fromisoformat(period);rd=dt.date.fromisoformat(release)
 if not 0<(rd-pd).days<100:raise ValueError('Invalid publication lag')
 if pd.month not in [3,6,9,12] or pd.day!=calendar.monthrange(pd.year,pd.month)[1]:raise ValueError('Unexpected observed calendar fiscal quarter end')
 q=pd.month//3;fy=pd.year
 return dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=de,period_excerpt=pe,identity_excerpt='Calendar fiscal-year quarter mapping of original financial statement quarter end '+period)
def build_special():
 for k in ['NFLX','PLTR']:
  events=[];pending=[]
  for mp in sorted((HERE/k/'raw').glob('*.meta.json')):
   if mp.name.endswith('.text.meta.json'):continue
   m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
   if m['status']!=200 or (k=='PLTR' and not p.name.startswith('filing_')) or (k=='NFLX' and not p.read_bytes().startswith(b'%PDF')):continue
   try:
    r=specialized(k,extract(p));e=event(k,p,m,r)
    if k=='PLTR':
     e['release_date_basis']='Original issuer statement in Item 2.02 explicitly says it issued the attached actual financial-results release on this date; SEC filing/acceptance metadata not used.'
     if 'Miami' in extract(p) or 'MIAMI' in extract(p):e['publication_timezone']='America/New_York'
    else:e['release_date_basis']='Date printed at top of actual shareholder results letter'
    events.append(e)
   except ValueError as ex:pending.append(dict(source_url=m['url'],evidence_path=rel(p),reason=str(ex)))
  keys=[(e['fiscal_year'],e['fiscal_quarter']) for e in events]
  assert len(keys)==len(set(keys)),(k,'duplicate')
  dump(HERE/k/'events.json',sorted(events,key=lambda e:e['release_date']));dump(HERE/k/'PENDING.json',dict(unadmitted=pending));print(k,len(events),'pending',len(pending))
if __name__=='__main__':build_special()
