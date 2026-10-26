"""Use explicit original announcement statements, never filing/acceptance dates."""
import json,re
import core as c

def facts(main,press):
 from build_events import DATE_RE,date,QMAP,IDENT,reject_notice
 if not re.search('Bank of America',main,re.I)or not re.search('Bank of America',press,re.I):raise ValueError('BAC identity absent from announcement/exhibit')
 candidates=[]
 for om in re.finditer(r'\bOn\s+',main):
  dm=DATE_RE.match(main,om.end())
  if not dm:continue
  statement=main[om.start():dm.end()+800]
  if not re.search(r'(?:announced|reported)\s+(?:its\s+)?financial results|issued.{0,120}press release.{0,80}announc',statement[:400],re.I):continue
  if re.search(r'preliminary|will announce|will report',statement[:400],re.I):continue
  rd=date(dm);qi=re.search(r'(first|second|third|fourth)[ -]quarter',statement,re.I)
  if not qi:continue
  q=QMAP[qi[1].lower()]
  ended=re.search(r'(?:quarter|year)[^.!?]{0,100}ended\s+',statement,re.I)
  if not ended:continue
  endmatch=DATE_RE.match(statement,ended.end())
  if not endmatch:continue
  pe=date(endmatch)
  if pe.month!=q*3 or not 0<(rd-pe).days<=115:continue
  fy=pe.year
  pattern=rf'Q{q}\s*[- ]\s*{fy%100:02}|{q}Q\s*{fy%100:02}|{qi[1]}[ -]quarter.{{0,35}}{fy}'
  pi=re.search(pattern,press[:10000],re.I)
  if not pi:continue
  candidates.append(dict(release_date=rd.isoformat(),fiscal_period_end_actual=pe.isoformat(),fiscal_year=fy,fiscal_quarter=q,statement=statement,exhibit_identity_excerpt=press[max(0,pi.start()-100):pi.end()+200]))
 if not candidates:raise ValueError('No explicit past announcement statement with matching original release exhibit')
 if len({(r['release_date'],r['fiscal_period_end_actual'])for r in candidates})!=1:raise ValueError('Conflicting original announcement statements')
 return candidates[0]

def extract(a):
 from build_events import reject_notice,notice_heading
 p=c.ROOT/a['evidence_path'];b=p.read_bytes();m=json.loads(p.with_name(p.name+'.meta.json').read_text());assert c.sha(b)==m['sha256']==a['evidence_sha256']
 x=a['release_exhibit'];ep=c.ROOT/x['evidence_path'];eb=ep.read_bytes();em=json.loads(ep.with_name(ep.name+'.meta.json').read_text());assert c.sha(eb)==em['sha256']==x['evidence_sha256']
 main=c.clean(c.pdf_text(p)) if b.startswith(b'%PDF') else c.text(b);press=c.clean(c.pdf_text(ep)) if eb.startswith(b'%PDF') else c.text(eb)
 if ep==p and b.startswith(b'%PDF'):
  head=re.search(r'Bank of America Reports',press[1000:],re.I)
  if not head:raise ValueError('Complete filing PDF has no separately identifiable earnings release')
  press=press[1000+head.start():]
 reject_notice(dict(url=x['url'],notice_heading=notice_heading(eb,press)),press);f=facts(main,press)
 if not c.START<=f['release_date']<=c.CUTOFF:raise ValueError('outside requested release-date window')
 gm=re.search(r'Charlotte\s*,?\s+North Carolina',main,re.I)
 if not gm:raise ValueError('Original BAC issuer location absent')
 proof=lambda text:c.proof(p,a['url'],text);statement=f.pop('statement');pi=f.pop('exhibit_identity_excerpt')
 return dict(f,ticker='BAC',publication_timestamp=None,precision='date_only',publication_timezone='America/New_York',source_url=a['url'],source_class='issuer_hosted_original_announcement_statement_with_full_release_exhibit',evidence_path=c.rel(p),evidence_sha256=c.sha(b),source_retrieved_at=m['retrieved_at'],first_public_time_verified=False,publication_timezone_basis='Conservative issuer-local calendar-date policy based on original issuer address; no clock inferred',official_date_published_raw=statement,release_date_basis='Original body expressly states On [date] the issuer announced financial results; filing header and acceptance timestamps are excluded',release_date_evidence=[dict(proof(statement),kind='original_issuer_explicit_release_announcement_statement')],fiscal_period_evidence=dict(proof(statement),basis='Original announcement statement explicitly gives the quarter and actual period ended'),fiscal_identity_evidence=dict(proof(statement),quarter_excerpt=statement,year_excerpt=statement,basis='Explicit quarter/year statement corroborated against attached original earnings release'),release_exhibit_identity_evidence=c.proof(ep,x['url'],pi),additional_evidence=[c.proof(ep,x['url']),a['attachment_link_evidence']],publication_timezone_evidence=dict(proof(main[max(0,gm.start()-60):gm.end()+80]),kind='issuer_local_calendar_date_policy',geographic_mapping='Charlotte, North Carolina -> America/New_York',cms_timezone_observed=False,exact_publication_time_inferred=False),timestamp_note='Original On [date] announcement statement, not SEC filing/acceptance or conference-call time.',historical_limitation='Issuer-hosted original filing and complete linked exhibit retrieved currently, not contemporaneous immutable capture.',in_study_window=f['release_date']>='2015-01-01')
