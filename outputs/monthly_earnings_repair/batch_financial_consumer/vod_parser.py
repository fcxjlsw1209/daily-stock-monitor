"""Original Vodafone news-release dates and fiscal-quarter identity, with explicit source proofs."""
import json,re,datetime as dt
import core as c
D=r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})'
DR=re.compile(D,re.I)
def date(m):return dt.datetime.strptime(' '.join(m.groups()[:3]).title(),'%d %B %Y').date()
def parse(a,queue):
 p=c.ROOT/a['evidence_path'];b=p.read_bytes();meta=json.loads(p.with_name(p.name+'.meta.json').read_text());assert c.sha(b)==a['evidence_sha256']==meta['sha256'];t=c.clean(c.pdf_text(p));visual=None
 if p.name=='d11714835a24ae1b5411.pdf':
  visual=dict(evidence_path=c.rel(c.HERE/'VOD/q1_fy2014_original_page1.png'),sha256=c.sha((c.HERE/'VOD/q1_fy2014_original_page1.png').read_bytes()),page_number=1,verified_at='2026-10-13',method='Rendered original PDF page visually read; malformed duplicate text layer retained unchanged',transcription='news release Interim management statement for the quarter ended 30 June 2013 19 July 2013 Vodafone Group Plc Vodafone House, The Connection, Newbury, Berkshire RG14 2FN, England')
  head=visual['transcription']
 else:
  if not re.search(r'news release',t[:500],re.I):raise ValueError('Not an original news release')
  head=re.split(r'[•\uf0b7]|Highlights',t[:1500],maxsplit=1,flags=re.I)[0]
 pm=re.search(r'(?:quarter|six months|year) ended\s+'+D,head,re.I)
 if not pm:
  # H1 2010/11 header reports its exact six-month date in the leading table.
  pm=re.search(r'Six months ended.{0,100}?'+D,t[:2200],re.I)
 if not pm:raise ValueError('Explicit actual reporting end absent from release heading/table')
 pe=date(DR.search(pm.group()));pub=[m for m in DR.finditer(head)if date(m)>pe and 0<(date(m)-pe).days<=115]
 if len({date(m)for m in pub})!=1:raise ValueError('Release date is ambiguous or absent from news-release heading')
 dm=pub[0];rd=date(dm)
 if not '2010-10-01'<=rd.isoformat()<='2014-02-28':raise ValueError('Outside assigned training predecessor window')
 q={3:4,6:1,9:2,12:3}.get(pe.month);fy=pe.year+(pe.month>3)
 if not q:raise ValueError('Unexpected Vodafone fiscal end month')
 fiscal_pattern=rf'Q{q}\s*(?:{fy-1}|{(fy-1)%100:02})\s*/\s*(?:{fy}|{fy%100:02})\b'
 identity=None;ip=None;iu=None
 sources=[a]+[x for x in queue if 'report'in x['url'] and 'evidence_path'in x and x['url'].split('/')[-1].split('-pr')[0].split('-report')[0]==a['url'].split('/')[-1].split('-pr')[0]]
 for x in sources:
  xp=c.ROOT/x['evidence_path'];xt=c.clean(c.pdf_text(xp));im=re.search(fiscal_pattern,xt,re.I)
  if im:
   identity=xt[max(0,im.start()-60):im.end()+80];ip=xp;iu=x['url'];break
 if identity is None:
  # Own news release may explicitly name Q2/Q4 and the March financial year separately.
  qi=re.search(rf'\bQ{q}\b',t,re.I);yi=re.search(rf'{fy-1}\s*/\s*{fy%100:02}|year end(?:ed|ing)\s+31 March {fy}|[“"]{fy} financial year',t,re.I)
  if qi and yi:identity=t[max(0,qi.start()-40):qi.end()+70];ip=p;iu=a['url'];yearquote=t[max(0,yi.start()-40):yi.end()+110]
  else:raise ValueError('Contemporaneous explicit fiscal-quarter/year identity absent')
 else:yearquote=identity
 geo=re.search(r'Newbury,?\s+Berkshire',visual['transcription']if visual else t,re.I)
 if not geo:raise ValueError('Original issuer locality absent')
 proof=lambda ex:dict(c.proof(p,a['url'],ex),**({'visual_verification':visual}if visual else {}))
 rq=dm.group();pq=pm.group();gq=geo.group()
 return dict(ticker='VOD',release_date=rd.isoformat(),fiscal_period_end_actual=pe.isoformat(),fiscal_year=fy,fiscal_quarter=q,precision='date_only',publication_timestamp=None,publication_timezone='Europe/London',source_url=a['url'],source_class='issuer_original_earnings_release_pdf',evidence_path=c.rel(p),evidence_sha256=c.sha(b),source_retrieved_at=meta['retrieved_at'],first_public_time_verified=False,official_date_published_raw=rq,release_date_basis='Date on original news-release first-page heading, not the presentation or conference timestamp',release_date_evidence=[dict(proof(rq),kind='original_news_release_heading')],fiscal_period_evidence=dict(proof(pq),basis='Explicit quarter/six-month/year ended in original release; same-period quarterly result identity separately verified'),fiscal_identity_evidence=dict(c.proof(ip,iu,identity),quarter_excerpt=identity,year_excerpt=yearquote,basis='Explicit quarterly fiscal-year label in contemporaneous original statement or presentation; Vodafone March fiscal-year end, not calendar-year numbering'),publication_timezone_evidence=dict(proof(gq),kind='issuer_local_calendar_date_policy',geographic_mapping='Newbury, Berkshire, England -> Europe/London',cms_timezone_observed=False,exact_publication_time_inferred=False),publication_timezone_basis='Conservative original issuer-local calendar date, no release clock inferred',additional_evidence=[c.proof(ip,iu)],historical_limitation='Currently retrieved original issuer archive, not contemporaneous immutable capture',timestamp_note='Date-only; actual release day checked independently from presentation identity evidence.',training_scope='Only GS/VOD assigned candidate windows and required predecessor period',in_study_window=False)

def run():
 queue=json.loads((c.HERE/'VOD/release_queue.json').read_text());events=[];pending=[]
 for a in queue:
  if not a['url'].endswith('pr.pdf')or'evidence_path'not in a:continue
  try:events.append(parse(a,queue))
  except Exception as e:pending.append(dict(url=a['url'],reason=str(e)))
 c.dump(c.HERE/'VOD/training_events.json',events);c.dump(c.HERE/'VOD/TRAINING_PENDING.json',pending);print('VOD',len(events),[(x['fiscal_year'],x['fiscal_quarter'])for x in events]);print(pending)
if __name__=='__main__':run()
