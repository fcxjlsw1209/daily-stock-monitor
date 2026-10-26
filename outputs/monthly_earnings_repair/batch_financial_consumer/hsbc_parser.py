"""Original HSBC releases: admit quarterly observations only with an explicit quarter section."""
import json,re,datetime as dt
import core as c

def extract(a):
 from build_events import M,MONTHS,DATE_RE,date
 p=c.ROOT/a['evidence_path'];b=p.read_bytes();meta=json.loads(p.with_name(p.name+'.meta.json').read_text());assert c.sha(b)==meta['sha256']==a['evidence_sha256'];t=c.clean(c.pdf_text(p));rx=re.compile(rf'\b(\d{{1,2}})\s+({M})\.?\s+(20\d{{2}})\b',re.I)
 def d(m):return dt.date(int(m[3]),MONTHS[m[2][:3].lower()],int(m[1]))
 ds=[m for m in rx.finditer(t) if m.start()<200 or re.match(r'\s*(?:HSBC Holdings plc|HSBC HOLDINGS PLC|Media enquiries)',t[m.end():m.end()+80],re.I)]
 if not ds:raise ValueError('HSBC dated release header absent')
 dm=ds[0];rd=d(dm)
 if not c.START<=rd.isoformat()<=c.CUTOFF:raise ValueError('outside requested release-date window')
 # Report identity in the title, followed by source-explicit quarterly result wording.
 h=t[:1200];hqm=re.search(r'([1-4])Q\s*(?:20)?(\d{2})\b',h,re.I)
 if hqm:q=int(hqm[1]);fy=2000+int(hqm[2]);qi=hqm
 else:
  ym=re.search(r'\b(20\d{2})\b.{0,30}(?:interim|results)|(?:interim|results).{0,30}\b(20\d{2})\b',h,re.I)
  if not ym:raise ValueError('HSBC fiscal title year missing')
  fy=int(ym[1] or ym[2])
  # The first date in some annual headers belongs to publication; find fiscal year directly by company title.
  ym2=re.search(r'HSBC HOLDINGS PLC\s+(20\d{2})\s+(?:INTERIM )?RESULTS',t[:600],re.I)
  if ym2:fy=int(ym2[1])
  q=2 if re.search('interim',h,re.I) else 4
  qi=re.search(rf'{q}Q\s*(?:20)?{fy%100:02}\b|'+('second' if q==2 else 'fourth')+rf'[ -]quarter\s+(?:of\s+)?{fy}',t,re.I)
  if not qi:
   pack=a.get('quarter_data_pack')
   if not pack:raise ValueError('HSBC annual/interim source has no explicit current quarterly result identity')
   assert pack['url'].rsplit('/',1)[0]==a['url'].rsplit('/',1)[0], 'Companion not from same official reporting package'
   ip=c.ROOT/pack['evidence_path'];it=c.clean(c.pdf_text(ip));assert c.sha(ip.read_bytes())==pack['evidence_sha256']
   qi=re.search(rf'HSBC HOLDINGS PLC Data Pack {q}Q {fy}',it,re.I)
   if not qi:raise ValueError('Companion data pack current-quarter identity missing')

 candidates=[]
 for mt in re.finditer(r'(?:quarter|half.year|year|period)[^.!?]{0,45}(?:ended|to)',t,re.I):
  chunk=t[mt.start():mt.end()+150]
  for m in rx.finditer(chunk):
   end=d(m)
   if 0<(rd-end).days<=115 and end.month==q*3:candidates.append((end,chunk))
 if not candidates:raise ValueError('HSBC actual period-end absent')
 pe,excerpt=max(candidates,key=lambda z:z[0])
 if pe.year!=fy:raise ValueError('HSBC fiscal title/period mismatch')
 gm=re.search('London',t,re.I)
 if not gm:raise ValueError('HSBC issuer-local London source evidence absent')
 evidence=lambda x:c.proof(p,a['url'],x)
 identity_p=locals().get('ip',p);identity_t=locals().get('it',t);identity_url=a.get('quarter_data_pack',{}).get('url',a['url'])
 release_excerpt=t[max(0,dm.start()-50):dm.end()+70];qexcerpt=identity_t[max(0,qi.start()-60):qi.end()+100]
 return dict(ticker='HSBC',release_date=rd.isoformat(),publication_timestamp=None,fiscal_period_end_actual=pe.isoformat(),fiscal_year=fy,fiscal_quarter=q,source_url=a['url'],source_class='issuer_original_earnings_release_pdf',evidence_path=c.rel(p),evidence_sha256=c.sha(b),source_retrieved_at=meta['retrieved_at'],precision='date_only',publication_timezone='Europe/London',publication_timezone_basis='Conservative issuer-local calendar date: original release identifies London; no publication clock inferred.',first_public_time_verified=False,official_date_published_raw=release_excerpt,release_date_basis='Date printed in the original results-release header/footer, not PDF metadata or filing date',release_date_evidence=[dict(evidence(release_excerpt),kind='original_release_publication_date')],fiscal_period_evidence=dict(evidence(excerpt),basis='Explicit original period-ended/to date; annual/interim source additionally requires explicit current quarter section'),fiscal_identity_evidence=dict(c.proof(identity_p,identity_url,qexcerpt),quarter_excerpt=qexcerpt,year_excerpt=qexcerpt if identity_p!=p else h,basis='Original explicit quarter identity from release or same official reporting package data pack'),publication_timezone_evidence=dict(evidence(t[max(0,gm.start()-80):gm.end()+80]),kind='issuer_local_calendar_date_policy',geographic_mapping='London -> Europe/London',cms_timezone_observed=False,exact_publication_time_inferred=False),official_archive_evidence=a.get('archive'),timestamp_note='Date only; no filing, PDF metadata or conference-call timestamp used.',historical_limitation='Currently retrieved official archive, not contemporaneous immutable capture.',in_study_window=rd.isoformat()>='2015-01-01')
