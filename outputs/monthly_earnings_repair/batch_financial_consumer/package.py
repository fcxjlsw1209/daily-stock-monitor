"""Reconcile original evidence first; label excerpts and every remaining gap."""
import collections,datetime as dt,json
import core as c
OWNED='BAC HD KO MA T COST CMCSA DIS WFC C PM PEP HSBC NKE MCD PYPL ABT TMO'.split()
ROOT_OWNED=['BUD','VZ']
def quarter_range(a,b):return [(n//4,n%4+1) for n in range(a[0]*4+a[1]-1,b[0]*4+b[1])]
def main():
 originals=json.loads((c.HERE/'original_events.json').read_text());indexed=json.loads((c.HERE/'indexed_events.json').read_text());unique={};conflicts=[]
 for x in originals+indexed:
  if x['ticker']not in OWNED:continue
  k=x['ticker'],x['fiscal_year'],x['fiscal_quarter']
  if k in unique:
   for field in ['release_date','fiscal_period_end_actual']:
    if unique[k][field]!=x[field]:conflicts.append(dict(ticker=k[0],fiscal_year=k[1],fiscal_quarter=k[2],field=field,preferred=unique[k][field],rejected=x[field],preferred_source=unique[k]['source_url'],rejected_source=x['source_url'],reason='Original evidence takes precedence; index discrepancy retained for review'))
  else:unique[k]=x
 rows=sorted(unique.values(),key=lambda x:(x['ticker'],x['release_date']));c.dump(c.HERE/'events.json',rows)
 details={};gapcount=0
 for s in OWNED:
  rr=[x for x in rows if x['ticker']==s];a,b=(2010,4),(2026,2)
  if s=='NKE':a,b=(2011,3),(2026,4)
  if s=='COST':a,b=(2011,2),(2026,4)
  if s=='DIS':a,b=(2011,1),(2026,3)
  if s=='PYPL':a=(2015,3)
  have={(x['fiscal_year'],x['fiscal_quarter'])for x in rr};missing=[f'{y}Q{q}'for y,q in quarter_range(a,b)if(y,q)not in have];gapcount+=len(missing)
  details[s]=dict(events=len(rr),original_full_file=sum(not x['source_class'].startswith('public_indexed')for x in rr),indexed_only=sum(x['source_class'].startswith('public_indexed')for x in rr),release_date_min=min((x['release_date']for x in rr),default=None),release_date_max=max((x['release_date']for x in rr),default=None),missing_fiscal_quarters=missing,coverage_complete=not missing)
 c.dump(c.HERE/'COVERAGE.json',details)
 source_pending=json.loads((c.HERE/'SOURCE_PENDING.json').read_text());index_pending=json.loads((c.HERE/'INDEXED_PENDING.json').read_text())
 c.dump(c.HERE/'PENDING.json',dict(target_window=[c.START,c.CUTOFF],missing_quarter_count=gapcount,by_issuer=details,root_owned={s:'Separate parent collection under warmup/'+s.lower()+'; not counted in this export'for s in ROOT_OWNED},source_rejections=source_pending,index_rejections=index_pending,conflicts=conflicts,limitations=['Quarter grid documents missing evidence only; no actual event date or period is fabricated from grid.','PYPL pre-separation historical quarters are outside the independent listed issuer event scope.','Some annual/interim HSBC releases lack an explicit current quarterly identity and remain rejected.','PM old API items can have empty content; their listing dates do not establish release publication.']))
 now=dt.datetime.now(dt.timezone.utc).isoformat()
 c.dump(c.HERE/'STATUS.json',dict(generated_at=now,status='partial_historical_coverage_with_verified_evidence',target_window=[c.START,c.CUTOFF],events=len(rows),original_full_file=sum(not x['source_class'].startswith('public_indexed')for x in rows),indexed_only=sum(x['source_class'].startswith('public_indexed')for x in rows),by_ticker={s:d['events']for s,d in details.items()},root_owned=ROOT_OWNED,missing_quarter_count=gapcount,full_universe_event_factor_ready=False,factor_backtested=False,publication_policy='date_only; issuer-local calendar date; publication_timestamp null; shared schema defers availability beyond release date',protected_scope='Only this batch directory modified; pools, ledgers, strategies and training untouched',events_sha256=c.sha((c.HERE/'events.json').read_bytes()),source_stops=json.loads((c.HERE/'SOURCE_STOPS.json').read_text())))
 print('MERGED',len(rows),'ORIGINAL',sum(not x['source_class'].startswith('public_indexed')for x in rows),'INDEXED',sum(x['source_class'].startswith('public_indexed')for x in rows),'MISSING',gapcount,'CONFLICTS',len(conflicts));print({s:d['events']for s,d in details.items()})
if __name__=='__main__':main()
