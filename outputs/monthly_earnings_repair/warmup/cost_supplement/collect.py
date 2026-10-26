"""Costco actual quarter anchors, distinct from monthly sales in the same original release."""
from source_io import *
import sys,subprocess,calendar
sys.path.insert(0,str(HERE.parent.parent))
from event_schema import normalize_event
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MONTH=r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)'
DATE=MONTH+r'\s+\d{1,2}\s*,?\s*\d{4}'
QMAP={'first':1,'second':2,'third':3,'fourth':4}
def clean(t):return re.sub(r'\s+',' ',re.sub(r'L\d+(?:@P[\d-]+)?:\s*',' ',t)).replace('–','-').replace('—','-').replace('\xa0',' ')
def iso(v):
 p=v.replace(',',' ').replace('.','').split();p[0]=p[0][:3];return dt.datetime.strptime(' '.join(p),'%b %d %Y').date().isoformat()
def extract(p):
 raw=p.read_bytes()
 if not raw.startswith(b'%PDF'):return text(raw)
 dst=p.with_suffix('.txt');meta=p.with_suffix('.text.meta.json')
 if dst.exists() and meta.exists():
  j=json.load(open(meta))
  if j['source_sha256']==sha(raw) and j['text_sha256']==sha(dst.read_bytes()):return dst.read_text()
 code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(sys.argv[1]).pages))'
 t=subprocess.check_output([PDF_PYTHON,'-c',code,str(p)],text=True,stderr=subprocess.DEVNULL);dst.write_text(t);dump(meta,dict(source_sha256=sha(raw),text_sha256=sha(dst.read_bytes()),parser='bundled pypdf layout'));return t

def parse(t):
 t=clean(t)
 if 'Costco Wholesale' not in t:raise ValueError('Costco identity absent')
 if not re.search(r'announced.{0,55}operating results',t[:8000],re.I):raise ValueError('Actual operating-results announcement absent; sales-only or scheduled releases excluded')
 dm=re.search(r'ISSAQUAH[A-Za-z .,()-]{0,35}('+DATE+r')',t,re.I)
 if dm:release=iso(dm.group(1))
 else:
  dm=re.search(r'ISSAQUAH[A-Za-z .,()-]{0,45}(\d{2}/\d{2}/\d{2})\s*--',t,re.I)
  if not dm:raise ValueError('Original Issaquah release dateline absent')
  release=dt.datetime.strptime(dm.group(1),'%m/%d/%y').date().isoformat()
 if not '2011-01-01'<=release<=CUTOFF:raise ValueError('Outside 2011-through-cutoff publication window')
 # The headline supplies FY; actual-results paragraph supplies quarter and observed end.
 title=re.search(r'Reports? (First|Second|Third|Fourth) Quarter.{0,110}?Fiscal (?:Year )?(20\d{2})',t[:4500],re.I)
 if not title:raise ValueError('Original quarterly results title FY/Q absent')
 q=QMAP[title.group(1).lower()];fy=int(title.group(2))
 matches=[]
 for pm in re.finditer(r'((?:(?:12|16|17)-week )?(?:first|second|third|fourth) quarter(?:\s*\([^)]{1,40}\))?[^.;]{0,200}?\bended\s+('+DATE+r'))',t,re.I):
  segment=pm.group(1);cq=re.search(r'(first|second|third|fourth) quarter',segment,re.I)
  if QMAP[cq.group(1).lower()]!=q:continue
  wm=re.search(r'(12|16|17)-week|\((twelve|sixteen|seventeen|12|16|17) weeks\)|quarter of fiscal \d{4} \((twelve|sixteen|seventeen|12|16|17) weeks\)',segment,re.I)
  if not wm:continue
  token=next(x for x in wm.groups() if x);weeks={'twelve':12,'sixteen':16,'seventeen':17}.get(token.lower(),int(token) if token.isdigit() else None)
  period=iso(pm.group(2));pd=dt.date.fromisoformat(period);rd=dt.date.fromisoformat(release)
  if not 0<(rd-pd).days<80:continue
  if (q==1 and not (pd.year==fy-1 and pd.month==11)) or (q==2 and not(pd.year==fy and pd.month==2)) or (q==3 and not(pd.year==fy and pd.month==5)) or (q==4 and not(pd.year==fy and pd.month in [8,9])):raise ValueError('Fiscal title disagrees with observed Costco fiscal end')
  if (q<4 and weeks!=12) or (q==4 and weeks not in [16,17]):raise ValueError('Unexpected explicitly reported quarter length')
  matches.append((period,weeks,segment))
 if not matches:raise ValueError('Explicit quarter weeks and its ended date absent; monthly sales not substituted')
 if len({(m[0],m[1]) for m in matches})!=1:raise ValueError('Conflicting current quarter anchors')
 period,weeks,pe=matches[0]
 return dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,fiscal_quarter_weeks=weeks,dateline_excerpt=dm.group(),period_excerpt=pe,identity_excerpt=title.group())
def event(p,m,r,indexed=False):
 proof=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=m['url'])
 e=dict(ticker='COST',release_date=r['release_date'],publication_timestamp=None,precision='date_only',publication_timezone='America/Los_Angeles',publication_timezone_basis='Conservative issuer-local Issaquah calendar date; no clock inferred.',first_public_time_verified=False,fiscal_period_end_actual=r['fiscal_period_end_actual'],fiscal_year=r['fiscal_year'],fiscal_quarter=r['fiscal_quarter'],fiscal_quarter_weeks=r['fiscal_quarter_weeks'],source_url=m['url'],source_class='public_index_observation_of_original_issuer_release_not_full_bytes' if indexed else 'issuer_original_release_pdf_full_bytes',evidence_path=rel(p),evidence_sha256=proof['sha256'],source_retrieved_at=m['retrieved_at'],release_date_basis='Original actual earnings-release Issaquah dateline, excluding CMS and conference clocks',release_date_evidence=[dict(proof,excerpt=r['dateline_excerpt'])],fiscal_period_evidence=dict(proof,excerpt=r['period_excerpt'],basis='Explicit fiscal quarter and week-length anchor; separate monthly sales dates excluded'),fiscal_identity_evidence=dict(proof,excerpt=r['identity_excerpt']),historical_limitation='Current archive observation; historical first-publication time not verified',in_study_window=r['release_date']>='2016-01-01')
 if indexed:e['evidence_limitation']='SHA-256 identifies saved public-index observation JSON, not original HTML/PDF bytes.'
 return e

def candidates():
 out=[];pending=[];visuals=json.load(open(HERE/'VISUAL_VERIFICATIONS.json')) if (HERE/'VISUAL_VERIFICATIONS.json').exists() else {}
 for mp in (HERE/'COST/raw').glob('*.meta.json'):
  if '.text.' in mp.name:continue
  m=json.load(open(mp));p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200:continue
  if 's201.q4cdn.com/287523651/' not in m['url']:pending.append(dict(source_url=m['url'],reason='Not identified Costco original CDN; discovery only'));continue
  try:
   if p.stem in visuals:
    v=visuals[p.stem];assert sha(p.read_bytes())==v['source_sha256'];op=p.with_suffix('.ocr-first.txt');assert sha(op.read_bytes())==v['ocr_sha256'];r=parse(op.read_text()+' '+v.get('visually_transcribed_ocr_omission',''));assert [r[k] for k in ['release_date','fiscal_year','fiscal_quarter','fiscal_period_end_actual','fiscal_quarter_weeks']]==v['visually_verified_values'];e=event(p,m,r);e['source_class']='issuer_original_release_pdf_full_bytes_visually_verified';e['additional_evidence']=[dict(evidence_path=rel(HERE/'VISUAL_VERIFICATIONS.json'),sha256=sha((HERE/'VISUAL_VERIFICATIONS.json').read_bytes()),source_url=m['url'],excerpt='Original first-page actual dateline and fiscal-quarter sentence inspected visually'),dict(evidence_path=v['rendered_page_path'],sha256=v['rendered_page_sha256'])];out.append(e)
   else:out.append(event(p,m,parse(extract(p))))
  except ValueError as e:pending.append(dict(source_url=m['url'],reason=str(e)))
 observations=[(HERE/'discovery.json',j) for j in json.load(open(HERE/'discovery.json'))]+[(p,json.load(open(p))) for p in (HERE/'public_index_observations').glob('*.json')]
 for p,j in observations:
  for block in j['rendering'].split('--------------------------------------------------------------------------------'):
   um=re.search(r'\((https://investor\.costco\.com/[^)\s]+)\)',block)
   if not um:continue
   try:out.append(event(p,dict(url=um.group(1),retrieved_at=j['observed_at']),parse(block),True))
   except ValueError as e:pending.append(dict(source_url=um.group(1),reason=str(e)))
 by={}
 for e in out:
  key=e['fiscal_year'],e['fiscal_quarter']
  if key in by:
   assert all(e[k]==by[key][k] for k in ['release_date','fiscal_period_end_actual','fiscal_quarter_weeks']),(key,by[key],e)
   if 'full_bytes' in e['source_class'] and 'not_full_bytes' not in e['source_class']:by[key]=e
  else:by[key]=e
 return by,pending

def build():
 by,pending=candidates();oldpath=HERE.parent.parent/'integrated_batches/financial_consumer_687/events.json';old=[e for e in json.load(open(oldpath)) if e['ticker']=='COST'];oldby={(e['fiscal_year'],e['fiscal_quarter']):e for e in old}
 additions=[e for key,e in by.items() if key not in oldby];audit=[];corrections=[]
 for key,e in oldby.items():
  new=by.get(key)
  if new and new['release_date']==e['release_date']:
   row=dict(fiscal_year=key[0],fiscal_quarter=key[1],release_date=e['release_date'],old_end=e['fiscal_period_end_actual'],verified_end=new['fiscal_period_end_actual'],status='matches' if e['fiscal_period_end_actual']==new['fiscal_period_end_actual'] else 'confirmed_correction',independent_evidence=new['fiscal_period_evidence'])
   audit.append(row)
   if row['status']=='confirmed_correction':corrections.append(dict(ticker='COST',**row,reason='Old event used monthly sales end instead of the explicit fiscal quarter end',corrected_event=new))
  else:audit.append(dict(fiscal_year=key[0],fiscal_quarter=key[1],release_date=e['release_date'],old_end=e['fiscal_period_end_actual'],status='independent_reverification_pending'))
 dump(HERE/'events.json',sorted(additions,key=lambda e:e['release_date']));dump(HERE/'CORRECTIONS.json',dict(original_input=rel(oldpath),original_input_sha256=sha(oldpath.read_bytes()),corrections=corrections,policy='Explicit correction overlay only; original immutable events not changed'))
 dump(HERE/'ORIGINAL_35_AUDIT.json',audit);dump(HERE/'ALL_VERIFIED_CANDIDATES.json',sorted(by.values(),key=lambda e:e['release_date']));dump(HERE/'PENDING.json',pending)
 print('Added',len(additions),'verified candidates',len(by),'corrections',len(corrections),'original35 independently checked',sum(r['status']!='independent_reverification_pending' for r in audit),'pending',len(pending))
if __name__=='__main__':build()
