import json,datetime as dt,collections
import core as c
D=c.HERE/'candidate_supplement_v2'
def main():
 base=json.loads((c.HERE/'events.json').read_text());prior=json.loads((c.HERE/'candidate_supplement/events.json').read_text());assert len(base)==687 and len(prior)==34
 known={(e['ticker'],e['fiscal_year'],e['fiscal_quarter']):e for e in base+prior};new={};conflicts=[]
 for name in ['T/announcement_events.json','WFC/issued_events.json','PM/issued_events.json','HSBC/candidate_events.json']:
  for e in json.loads((c.HERE/name).read_text()):
   k=e['ticker'],e['fiscal_year'],e['fiscal_quarter']
   if k in known:
    for f in ['release_date','fiscal_period_end_actual']:
     if known[k][f]!=e[f]:conflicts.append(dict(key=k,field=f,old=known[k][f],new=e[f]))
   elif k in new:assert e['release_date']==new[k]['release_date']
   else:new[k]=e
 rows=sorted(new.values(),key=lambda e:(e['ticker'],e['release_date']));c.dump(D/'events.json',rows)
 c.dump(D/'STATUS.json',dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),status='validation_pending',events=len(rows),by_ticker=dict(collections.Counter(e['ticker']for e in rows)),original_full_file=sum(not e['source_class'].startswith('public_indexed')for e in rows),indexed_only=sum(e['source_class'].startswith('public_indexed')for e in rows),events_sha256=c.sha((D/'events.json').read_bytes()),base_687_sha256=c.sha((c.HERE/'events.json').read_bytes()),prior_34_sha256=c.sha((c.HERE/'candidate_supplement/events.json').read_bytes()),conflicts=conflicts,notes=['Incremental new quarters only; preserves prior 687 and 34 exports.','No SEC direct requests; indexed evidence explicitly distinguished from original bytes.','HSBC current-quarter identity comes from original Data Pack in the same official archive reporting package.','Date-only publication times stay null.']))
 have=set(known)|set(new);missing={s:[f'{y}Q{q}'for y in range(2010,2021)for q in range(1,5)if(2010,4)<=(y,q)<=(2020,4)and(s,y,q)not in have]for s in ['T','WFC','PM']}
 c.dump(D/'PENDING.json',dict(missing_quarter_grid_for_research_only=missing,actual_candidate_priority=['PM2012Q2','WFC2012Q2','WFC2012Q3','WFC2013Q1','WFC2014Q3'],no_imputed_dates=True,full_history_complete=False,notes=['The quarter grid is a research aid, not evidence that a particular candidate missed a published earnings release. T2010Q4 through2018Q1 complete; HSBC2017Q2/2017Q4 candidate gaps addressed.']))
 print('V2',len(rows),dict(collections.Counter(e['ticker']for e in rows)),conflicts)
if __name__=='__main__':main()
