import json,collections,datetime as dt
import core as c
D=c.HERE/'candidate_supplement_v3'
def main():
 prior=[];hashes={}
 for rel,n in [('events.json',687),('candidate_supplement/events.json',34),('candidate_supplement_v2/events.json',66)]:
  p=c.HERE/rel;rr=json.loads(p.read_text());assert len(rr)==n;prior+=rr;hashes[rel]=c.sha(p.read_bytes())
 old={(e['ticker'],e['fiscal_year'],e['fiscal_quarter']):e for e in prior};new={}
 for rel in ['WFC/archived_events.json','WFC/final_press_events.json','PM/final_press_events.json']:
  for e in json.loads((c.HERE/rel).read_text()):
   k=e['ticker'],e['fiscal_year'],e['fiscal_quarter']
   if k in old:assert (e['release_date'],e['fiscal_period_end_actual'])==(old[k]['release_date'],old[k]['fiscal_period_end_actual']);continue
   new.setdefault(k,e)
 rows=sorted(new.values(),key=lambda e:(e['ticker'],e['release_date']));assert len(rows)==5;c.dump(D/'events.json',rows)
 have=set(old)|set(new)
 ranges={'T':[(2010,4,2018,1)],'WFC':[(2010,4,2020,1)],'PM':[(2010,3,2014,4),(2015,3,2017,4)],'HSBC':[(2017,2,2017,2),(2017,4,2017,4)]};checks=[]
 for ticker,rangeset in ranges.items():
  for sy,sq,ey,eq in rangeset:
   expected={(ticker,n//4,n%4+1)for n in range(sy*4+sq-1,ey*4+eq)};missing=sorted(expected-have);assert not missing,(ticker,missing)
   checks.append(dict(ticker=ticker,first=f'{sy}Q{sq}',last=f'{ey}Q{eq}',quarter_count=len(expected),missing=missing))
 c.dump(D/'CONTINUITY.json',dict(verified_quarter_sequences=checks,meaning='Quarter sequences required by reviewed frozen candidate/training windows are complete in these ranges. Actual event availability must still use shared conservative date-only schema; reporting on signal date is not assumed available that day.',all_quarters_2011_2026_claim=False))
 c.dump(D/'STATUS.json',dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),status='validation_pending',events=5,by_ticker=dict(collections.Counter(e['ticker']for e in rows)),original_full_file=1,indexed_only=4,events_sha256=c.sha((D/'events.json').read_bytes()),prior_hashes=hashes,conflicts=[],source_note='WFC2014Q3 full issuer filing+attached earnings release in Fast-EDGAR public archive; other4 indexed issuer earnings releases. All publication dates are actual body/dateline/header evidence, never filing dates.'))
 c.dump(D/'PENDING.json',dict(remaining_identified_required_quarters=[],not_claiming_full_2011_2026_history=True,source_stops_unchanged=True,unused_files_retained=True,notes=['Root must rerun all frozen candidates, checking expected most recent reported quarter, not merely the 120day age heuristic.','Public Q4 cross-CIK discovery returned404 and is unused, retained with metadata.','No direct SEC, Yahoo or AlphaVantage requests; no stopped host retries.']))
 print('V3',[(e['ticker'],e['release_date'],e['fiscal_period_end_actual'])for e in rows],checks)
if __name__=='__main__':main()
