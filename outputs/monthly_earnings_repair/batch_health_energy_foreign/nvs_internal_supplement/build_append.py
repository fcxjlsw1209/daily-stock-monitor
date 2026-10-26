import sys,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import collect as c,build as b
urls=['https://www.novartis.com/news/media-releases/novartis-delivered-sales-growth-cc-significant-margin-expansion-cc-and-strong-innovation-q11-now-more-focused-portfolio','https://www.novartis.com/news/media-releases/novartis-delivered-solid-q1-despite-gleevec-loss-exclusivity-investing-behind-new-launches-long-term-growth']
base=c.load(HERE.parent/'cached_internal_supplement/events.json',[])[0];events=[]
for u in urls:
 p=c.RAW/'NVS'/(c.sha(u.encode())[:20]+'.bin');m=c.load(p.with_suffix('.meta.json'),{});tx=b.text(p,'NVS');task=dict(url=u);y,q,ident=b.identity('NVS',tx,task);rd,rex=b.release('NVS',tx,y,q);end,ep=b.period('NVS',tx,y,q,task,p)
 e=copy.deepcopy(base);e.update(ticker='NVS',release_date=rd,publication_timezone='Europe/Zurich',publication_timezone_basis='Conservative issuer-local calendar-date policy; Basel original dateline, no exact instant inferred.',fiscal_year=y,fiscal_quarter=q,fiscal_period_end_actual=end,source_url=u,evidence_path=c.rel(p),evidence_sha256=m['sha256'],raw_original_bytes_sha256=m['sha256'],retrieved_at=m['retrieved_at'],evidence_excerpt=ident,release_date_evidence=[b.proof(p,u,rex,'Original issuer release dateline')],fiscal_period_evidence=ep,additional_evidence=[dict(evidence_path=ep['evidence_path'],sha256=ep['sha256'],role='actual_fiscal_period_end')],in_study_window=rd>='2016-01-01');events.append(e)
c.dump(HERE/'events.json',events);c.dump(HERE/'PENDING.json',[]);c.dump(HERE/'STATUS.json',dict(verified_events=2,events_by_ticker={'NVS':2},pending_count=0,complete=True,scope='Two internal-quarter repairs: NVS2015Q1 and2016Q1',stable_for_merge=False,event_file_sha256=c.sha((HERE/'events.json').read_bytes()),generated_utc=c.utc()))
