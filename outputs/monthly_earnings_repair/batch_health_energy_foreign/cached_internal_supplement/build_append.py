import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import collect as c,build as b
u='https://news.abbvie.com/2016-01-29-AbbVie-Reports-Full-Year-2015-and-Fourth-Quarter-Financial-Results';p=c.RAW/'ABBV'/(c.sha(u.encode())[:20]+'.bin');m=c.load(p.with_suffix('.meta.json'),{});tx=b.text(p,'ABBV');task=dict(url=u)
y,q,identity=b.identity('ABBV',tx,task);rd,rex=b.release('ABBV',tx,y,q);end,ep=b.period('ABBV',tx,y,q,task,p)
assert (y,q,rd,end)==(2015,4,'2016-01-29','2015-12-31')
e=dict(ticker='ABBV',release_date=rd,publication_timestamp=None,precision='date_only',publication_timezone='America/Chicago',publication_timezone_basis='Conservative issuer-local calendar-date policy; NORTH CHICAGO original dateline. No exact instant inferred.',first_public_time_verified=False,fiscal_year=y,fiscal_quarter=q,fiscal_period_end_actual=end,source_url=u,source_class='issuer_original_release',source_representation='original_html_bytes',evidence_path=c.rel(p),evidence_sha256=m['sha256'],raw_original_bytes_sha256=m['sha256'],raw_original_bytes_available=True,retrieved_at=m['retrieved_at'],evidence_excerpt=identity,release_date_evidence=[b.proof(p,u,rex,'Original issuer release dateline')],fiscal_period_evidence=ep,additional_evidence=[dict(evidence_path=ep['evidence_path'],sha256=ep['sha256'],role='actual_fiscal_period_end')],historical_limitation='Current official archive is not a contemporaneous immutable feed. First-public instant not established.',timestamp_note='No call time, CMS dateModified, publication calendar, or SEC acceptance substituted.',in_study_window=True)
c.dump(HERE/'events.json',[e]);c.dump(HERE/'PENDING.json',[])
c.dump(HERE/'STATUS.json',dict(verified_events=1,events_by_ticker={'ABBV':1},pending_count=0,complete=True,scope='single cached internal-quarter repair: ABBV2015Q4',stable_for_merge=False,event_file_sha256=c.sha((HERE/'events.json').read_bytes()),generated_utc=c.utc()))
