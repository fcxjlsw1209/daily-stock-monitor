"""Pfizer actual-candidate supplement: original datelines and US fiscal periods."""
from source_io import *
import subprocess,sys,calendar
sys.path.insert(0,str(HERE.parent.parent))
from event_schema import normalize_event
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MONTH=r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DATE=MONTH+r'\s+\d{1,2},?\s+20\d{2}'
QMAP=dict(first=1,second=2,third=3,fourth=4)
BASELINE=HERE.parent.parent/'integrated_batches/health_energy_foreign_900/events.json'
TARGETS={(2010,3),(2010,4),(2015,3),(2020,1)}|{(y,q) for y in range(2011,2015) for q in range(1,5)}
def clean(t):return re.sub(r'\s+',' ',t).replace('\xa0',' ').replace('—','-').replace('–','-')
def iso(s):return dt.datetime.strptime(s.replace(',','').title(),'%B %d %Y').date().isoformat()
def extract(p):
 raw=p.read_bytes()
 if not raw.startswith(b'%PDF'):raise ValueError('Original PDF bytes required')
 dst=p.with_suffix('.txt');mp=p.with_suffix('.text.meta.json')
 if dst.exists() and mp.exists():
  m=json.loads(mp.read_text())
  if sha(raw)==m['source_sha256'] and sha(dst.read_bytes())==m['text_sha256']:return dst.read_text()
 code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(sys.argv[1]).pages))'
 t=subprocess.check_output([PDF_PYTHON,'-c',code,str(p)],text=True,stderr=subprocess.DEVNULL)
 dst.write_text(t);dump(mp,dict(source_sha256=sha(raw),text_sha256=sha(dst.read_bytes()),parser='bundled pypdf layout'));return t

def parse(raw):
 t=clean(raw)
 qm=re.search(r'PFIZER REPORTS (FIRST|SECOND|THIRD|FOURTH)-QUARTER(?: AND FULL-YEAR)? (20\d{2}) RESULTS',t[:300],re.I)
 if not qm:raise ValueError('Current Pfizer quarterly-results headline absent; forecasts and filename labels not substituted')
 q=QMAP[qm.group(1).lower()];fy=int(qm.group(2))
 if (fy,q) not in TARGETS:raise ValueError('Outside assigned target quarters')
 dm=re.search(r'NEW YORK,\s*N\.?Y\.?[,]?\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday),\s*('+DATE+r')\s*-\s*Pfizer Inc\.',t[:3500],re.I)
 if not dm:raise ValueError('Original body release dateline absent; CMS or filing dates not substituted')
 release=iso(dm.group(1))
 # The first date in this exact US/consolidated statement note is the current
 # reporting period. The adjacent overseas note is retained as disambiguation.
 pattern=r'(?:The\s+)?(?:above\s+)?financial statements present the three(?:-month(?: and (?:six|nine|twelve)-month)? periods|(?: and (?:six|nine|twelve))? months) ended\s+('+DATE+r')(.{0,100}?)\.\s*Subsidiaries operating outside the (?:United States|U\.S\.) are included.{0,200}?\.'
 hits=list(re.finditer(pattern,t,re.I))
 if not hits:raise ValueError('Explicit US/consolidated financial-statement period note absent; no calendar imputation')
 ends={iso(m.group(1)) for m in hits}
 if len(ends)!=1:raise ValueError('Conflicting current US fiscal period notes')
 end=ends.pop();pd=dt.date.fromisoformat(end);rd=dt.date.fromisoformat(release)
 # Accept issuer-observed 52-week dates spanning the next calendar month.
 # This bound validates an observed date; it never constructs the event date.
 reference=dt.date(fy,q*3,calendar.monthrange(fy,q*3)[1])
 if pd.year!=fy or abs((pd-reference).days)>7:raise ValueError('Observed period inconsistent with fiscal identity')
 if not 0<(rd-pd).days<110:raise ValueError('Release is not after actual period within normal report window')
 revision=re.search(r'\[CORRECTED AS OF[^\]]+\]',t[:200],re.I)
 return dict(fiscal_year=fy,fiscal_quarter=q,release_date=release,fiscal_period_end_actual=end,identity_excerpt=qm.group(),release_excerpt=dm.group(),period_excerpt=hits[0].group(),revision_notice=revision.group() if revision else None)

def event(p,m,r):
 proof=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=m['url'])
 revision=r['revision_notice']
 filename_reissue='reissued' in m['url'].lower()
 version='corrected_archival_pdf' if revision else ('archival_reissue_filename' if filename_reissue else 'original_archival_pdf')
 e=dict(ticker='PFE',release_date=r['release_date'],publication_timestamp=None,precision='date_only',publication_timezone='America/New_York',publication_timezone_basis='Conservative issuer-local New York dateline; exact publication clock is unverified',first_public_time_verified=False,fiscal_year=r['fiscal_year'],fiscal_quarter=r['fiscal_quarter'],fiscal_period_end_actual=r['fiscal_period_end_actual'],source_url=m['url'],source_class='issuer_original_earnings_release_pdf_full_bytes',evidence_path=rel(p),evidence_sha256=proof['sha256'],source_retrieved_at=m['retrieved_at'],release_date_basis='Explicit NEW YORK body dateline of completed quarterly earnings release; CMS, correction and filing clocks are not publication substitutes',release_date_evidence=[dict(proof,excerpt=r['release_excerpt'])],fiscal_identity_evidence=dict(proof,excerpt=r['identity_excerpt']),fiscal_period_evidence=dict(proof,excerpt=r['period_excerpt'],basis='Current US/consolidated financial-statement period explicitly distinguished from prior-year comparatives and one-month-lagged non-US subsidiaries'),publication_version=version,source_revision_notice=revision,historical_limitation='Current issuer archival bytes; historical first-publication instant not verified',event_scope='regular quarterly earnings event metadata only; no revised financial values are imported',in_study_window=r['release_date']>='2016-01-01')
 if version!='original_archival_pdf':
  cp=HERE/'REVISION_DATE_CORROBORATION.json';cs=json.loads(cp.read_text());c=next(c for c in cs if (c['fiscal_year'],c['fiscal_quarter'])==(r['fiscal_year'],r['fiscal_quarter']))
  assert c['release_date']==r['release_date']
  if c.get('fiscal_period_end_actual'):assert c['fiscal_period_end_actual']==r['fiscal_period_end_actual']
  e['corroborating_public_index_evidence']=dict(evidence_path=rel(cp),sha256=sha(cp.read_bytes()),**c)
  e['revision_handling']='Only original event date, fiscal identity and actual fiscal end are used. Original body dateline independently matches issuer public archive/release index. Revised financial values are outside this event-metadata dataset.'
 if revision:
  t=clean(extract(p));cm=re.search(r'.{0,80}inadvertently indicated.{0,350}?40 basis points[.]',t,re.I)
  if not cm:raise ValueError('Correction scope not found')
  e['source_revision_evidence']=dict(proof,excerpt=cm.group())
  e['source_revision_date']='2020-05-05'
 if filename_reissue:e['source_filename_warning']='Filename says Q4/Reissued, but original body explicitly reports THIRD-QUARTER 2015 on October 27, 2015; no fiscal identity inferred from URL'
 if r['release_date']<'2011-01-01':e['training_predecessor_only']=True
 return e

def build():
 by={};pending=[];inventory=[]
 for mp in sorted((HERE/'PFE/raw').glob('*.meta.json')):
  if '.text.' in mp.name:continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  item=dict(source_url=m['url'],evidence_path=rel(p),status=m['status'],evidence_sha256=sha(p.read_bytes()))
  inventory.append(item)
  if m['status']!=200:pending.append(dict(item,reason='Cached failed source preserved'));continue
  assert m['sha256']==item['evidence_sha256']
  try:r=parse(extract(p));e=event(p,m,r)
  except ValueError as ex:pending.append(dict(item,reason=str(ex)));continue
  key=e['fiscal_year'],e['fiscal_quarter'];score={'original_archival_pdf':2,'archival_reissue_filename':1,'corrected_archival_pdf':0}
  if key in by:
   old=by[key];assert all(e[k]==old[k] for k in ['release_date','fiscal_period_end_actual']),(e,old)
   if score[e['publication_version']]>score[old['publication_version']]:
    pending.append(dict(source_url=old['source_url'],evidence_path=old['evidence_path'],reason='Matching duplicate/reissued source preserved; preferred uncorrected archival original'))
    by[key]=e
   else:pending.append(dict(item,reason='Matching duplicate/reissued source preserved; preferred uncorrected archival original'))
  else:by[key]=e
 old=[e for e in json.loads(BASELINE.read_text()) if e['ticker']=='PFE'];oldkeys={(e['fiscal_year'],e['fiscal_quarter']) for e in old}
 es=[e for k,e in sorted(by.items()) if k not in oldkeys]
 dump(HERE/'events.json',es)
 dump(HERE/'SOURCE_INVENTORY.json',inventory)
 dump(HERE/'PENDING.json',dict(unadmitted=pending,missing_target_quarters=sorted(TARGETS-set(by)),baseline=rel(BASELINE),baseline_sha256=sha(BASELINE.read_bytes()),baseline_event_count=len(old),source_stops_path=rel(HERE/'SOURCE_STOPS.json')))
 print('PFE',len(es),'new; missing',sorted(TARGETS-set(by)))
 print([(e['fiscal_year'],e['fiscal_quarter'],e['release_date'],e['fiscal_period_end_actual'],e['publication_version']) for e in es])
if __name__=='__main__':build()

