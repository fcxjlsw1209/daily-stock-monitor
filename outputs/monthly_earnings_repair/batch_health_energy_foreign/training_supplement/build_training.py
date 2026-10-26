"""Limited training-window earnings; original datelines and actual ends only."""
import sys,re,datetime as dt
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import collect as c,build as b
TARGETS={'COP':[(2010,4)]+[(2011,q) for q in range(1,5)]+[(2012,1)],'OXY':[(2010,4)]+[(2011,q) for q in range(1,5)],'SLB':[(2010,4)]+[(y,q) for y in range(2011,2014) for q in range(1,5)]+[(2014,1),(2014,2)],'GILD':[(2013,3),(2013,4)]+[(2014,q) for q in range(1,5)]+[(2015,q) for q in range(1,4)]}
CID={'1163165':'COP','797468':'OXY','87347':'SLB'}
def ticker(u):
 for part,t in [('gilead.com','GILD'),('slb.com','SLB'),('conocophillips','COP'),('occidentalpetroleum','OXY')]:
  if part in u:return t
 m=re.search(r'/data/(\d+)/',u);return CID.get(m[1]) if m else None
def captures():
 out=[]
 for src in HERE.glob('*.json'):
  if src.name in ('events.json','STATUS.json','PENDING.json','candidates.json'):continue
  d=c.load(src,{})
  if not isinstance(d,dict) or not d.get('retrieved_at'):continue
  responses=[]
  if 'response' in d:responses.append(d['response'])
  for r in d.get('results',[]):
   if r['status']!='fulfilled':continue
   v=r['value'];responses.append(v['response'] if isinstance(v,dict) else v)
  for response in responses:
   for chunk in re.split(r'-{50,}',response):
    match=re.search(r'\((https://[^\s)]+)\)',chunk[:900])
    if not match:continue
    u=match[1];t=ticker(u)
    if not t:continue
    p=HERE/'indexed'/f'{t}_{c.sha((u+src.name+chunk).encode())[:20]}.json';p.parent.mkdir(exist_ok=True)
    c.dump(p,dict(url=u,retrieved_at=d['retrieved_at'],indexed_source_text=chunk,source_representation='public_search_index_not_original_bytes',original_bytes_available=False,upstream_capture_path=c.rel(src),upstream_capture_sha256=c.sha(src.read_bytes())))
    out.append(dict(ticker=t,url=u,path=p))
 return out

def release(t,tx):
 pats={'COP':r'HOUSTON\s*[- ,]*\(BUSINESS WIRE\)\s*[- ,]*('+b.MDY+')','OXY':r'(?:For Immediate Release:\s*|LOS ANGELES,\s*)('+b.MDY+')','SLB':r'(Houston|Paris|London|The Hague)\s*[-, ]*(?:\(BUSINESS WIRE\)\s*[-, ]*)?('+b.MDY+')','GILD':r'FOSTER CITY,\s*Calif\.?\s*[- ,]*\(BUSINESS WIRE\)\s*[- ,]*('+b.MDY+')'}
 m=re.search(pats[t],tx,re.I)
 if not m:raise ValueError('Original release dateline absent')
 zone={'COP':'America/Chicago','OXY':'America/Los_Angeles','GILD':'America/Los_Angeles','SLB':'America/Chicago'}[t]
 if t=='SLB':zone={'houston':'America/Chicago','paris':'Europe/Paris','london':'Europe/London','the hague':'Europe/Amsterdam'}[m[1].lower()]
 return b.date(m[2] if t=='SLB' else m[1]),m[0],zone

def run():
 events={};rejections=[]
 sources=captures()
 for a in sources:
  t=a['ticker'];p=a['path'];u=a['url'];tx=b.text(p,t)
  if 'form8k' in u:continue
  try:
   rd,rex,zone=release(t,tx)
   if not '2011-01-01'<=rd<='2015-12-31':continue
   matches=[]
   qm=re.search(r'(first|second|third|fourth)[- ]+quarter',tx[:5000],re.I)
   if not qm:raise ValueError('Explicit reporting-quarter heading absent')
   heading_q=next(q for q,w in b.WORDS.items() if w==qm[1].lower())
   for y,q in TARGETS[t]:
    if q!=heading_q:continue
    expected=b.expected_period(t,y,q)
    days=(dt.date.fromisoformat(rd)-dt.date.fromisoformat(expected)).days
    if not 0<days<=120:continue
    word=b.WORDS[q]
    if not re.search(word+r'[- ]+quarter|Q'+str(q)+r'\b',tx,re.I):continue
    try:ep=b.period(t,tx,y,q,a,p,False)
    except ValueError:
     ep=None
     if t=='SLB':
      ed=dt.date.fromisoformat(expected)
      for hm in re.finditer(r'Periods? Ended\s+'+b.MONTHS[ed.month-1]+r'\s+'+str(ed.day)+r'\b',tx,re.I):
       excerpt=tx[max(0,hm.start()-120):hm.end()+230]
       if re.search(r'\b'+str(y)+r'\b',excerpt):
        ep=(expected,b.proof(p,u,excerpt,'Original income-statement header explicitly states Periods Ended month/day and current year column'))
        break
     for sp in sources:
      if ep is not None:break
      if sp['ticker']!=t:continue
      stx=b.text(sp['path'],t)
      cm=re.search(r'For the (?:quarterly period|fiscal year) ended\s*:?\s*('+b.DP+')',stx,re.I)
      if cm and b.date(cm[1])==expected:
       ep=(expected,b.proof(sp['path'],sp['url'],cm[0],'Independent official reporting-form cover confirms actual fiscal end; filing date is not used as publication'))
       break
      if sp['url'].rsplit('/',1)[0]!=u.rsplit('/',1)[0] or 'form8k' not in sp['url']:continue
      try:ep=b.period(t,stx,y,q,sp,sp['path'],False);break
      except ValueError:pass
     if ep is None:raise ValueError('actual fiscal end absent in release and same-submission statement')
    matches.append((y,q,ep))
   if len(matches)!=1:raise ValueError('Unique explicit fiscal identity/period absent')
   y,q,(end,ep)=matches[0];m=c.load(p,{});h=c.sha(p.read_bytes())
   e=dict(ticker=t,release_date=rd,publication_timestamp=None,precision='date_only',publication_timezone=zone,publication_timezone_basis='Conservative issuer-local date policy; original dateline city mapping, no inferred publication instant.',first_public_time_verified=False,fiscal_year=y,fiscal_quarter=q,fiscal_period_end_actual=end,source_url=u,source_class='issuer_original_release',source_representation='public_search_index_excerpt',evidence_path=c.rel(p),evidence_sha256=h,raw_original_bytes_available=False,raw_original_bytes_sha256=None,retrieved_at=m['retrieved_at'],release_date_evidence=[b.proof(p,u,rex,'Original issuer release dateline, not filing time')],fiscal_period_evidence=ep,additional_evidence=[dict(evidence_path=ep['evidence_path'],sha256=ep['sha256'],role='actual_fiscal_period_end')],evidence_excerpt=rex,historical_limitation='Official original accessed as public search-index excerpt. Original HTTP HTML/PDF bytes were NOT downloaded; evidence SHA hashes saved capture only.',timestamp_note='No CMS, call, scheduled announcement or SEC acceptance time substituted.',research_scope='training_candidate_window_and_predecessor',in_study_window=False)
   key=(t,y,q)
   if key in events and any(events[key][k]!=e[k] for k in ['release_date','fiscal_period_end_actual']):raise ValueError('Conflicting verified dates')
   if key not in events or len(tx)>len(b.text(c.ROOT/events[key]['evidence_path'],t)):events[key]=e
  except Exception as ex:rejections.append(dict(ticker=t,url=u,reason=str(ex),evidence_path=c.rel(p)))
 es=sorted(events.values(),key=lambda e:(e['ticker'],e['release_date']));c.dump(HERE/'events.json',es)
 pending=[dict(ticker=t,fiscal_year=y,fiscal_quarter=q,status='unverified',reason='Original publication date and exact actual fiscal end not both established') for t,targets in TARGETS.items() for y,q in targets if (t,y,q) not in events]
 c.dump(HERE/'PENDING.json',pending);c.dump(HERE/'PARSE_PENDING.json',rejections)
 st=dict(verified_events=len(es),events_by_ticker={t:sum(e['ticker']==t for e in es) for t in TARGETS},pending_count=len(pending),stable_for_merge=False,complete=not pending,research_scope='training_candidate_window_and_predecessor',event_file_sha256=c.sha((HERE/'events.json').read_bytes()),generated_utc=c.utc());c.dump(HERE/'STATUS.json',st);print(st)
if __name__=='__main__':run()
