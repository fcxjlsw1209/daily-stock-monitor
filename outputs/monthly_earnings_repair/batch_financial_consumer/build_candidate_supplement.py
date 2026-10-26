"""New quarters only, preserving the previously delivered 687-event export."""
import json,datetime as dt,collections
import core as c
D=c.HERE/'candidate_supplement'
def main():
 base=json.loads((c.HERE/'events.json').read_text());assert len(base)==687
 baseline={(x['ticker'],x['fiscal_year'],x['fiscal_quarter']):x for x in base};extra=[]
 for rel in ['T/announcement_events.json','VOD/training_events.json','GS/training_events.json']:
  extra+=json.loads((c.HERE/rel).read_text())
 extra += [x for x in json.loads((c.HERE/'indexed_events.json').read_text())if x['ticker']=='WFC']
 unique={};conflicts=[]
 for x in extra:
  k=x['ticker'],x['fiscal_year'],x['fiscal_quarter']
  if k in baseline:
   for field in ['release_date','fiscal_period_end_actual']:
    if x[field]!=baseline[k][field]:conflicts.append(dict(key=k,field=field,old=baseline[k][field],new=x[field],source_url=x['source_url']))
   continue
  if k in unique:
   assert (x['release_date'],x['fiscal_period_end_actual'])==(unique[k]['release_date'],unique[k]['fiscal_period_end_actual'])
  else:unique[k]=x
 rows=sorted(unique.values(),key=lambda x:(x['ticker'],x['release_date']));c.dump(D/'events.json',rows)
 summary={s:dict(count=sum(x['ticker']==s for x in rows),release_min=min(x['release_date']for x in rows if x['ticker']==s),release_max=max(x['release_date']for x in rows if x['ticker']==s))for s in sorted({x['ticker']for x in rows})}
 c.dump(D/'STATUS.json',dict(generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),status='validation_pending',events=len(rows),original_full_file=sum(not x['source_class'].startswith('public_indexed')for x in rows),indexed_only=sum(x['source_class'].startswith('public_indexed')for x in rows),by_ticker=summary,events_sha256=c.sha((D/'events.json').read_bytes()),base_events=687,base_sha256=c.sha((c.HERE/'events.json').read_bytes()),base_preserved=True,conflicts=conflicts,notes=['Only newly verified quarters; original 687 unchanged.','Vodafone includes 2010-11-09 predecessor for January 2011 training candidate.','No direct SEC requests; T sources are explicitly indexed-only original announcement clauses.','COST correction layer is owned by earnings_visa and parent; no COST changes here.']))
 have={(x['ticker'],x['fiscal_year'],x['fiscal_quarter'])for x in base+rows};pending={s:[f'{y}Q{q}'for y in range(2010,2027)for q in range(1,5)if(2010,4)<=(y,q)<=(2026,2)and(s,y,q)not in have]for s in ['T','WFC','PM','HSBC','C']}
 c.dump(D/'PENDING.json',dict(missing_quarter_grid_for_research_only=pending,source_pending={s:json.loads((c.HERE/f).read_text())for s,f in [('T','T/ANNOUNCEMENT_PENDING.json'),('VOD','VOD/TRAINING_PENDING.json')]},no_imputed_dates=True,full_coverage_claim=False))
 print('NEW',len(rows),summary,'CONFLICTS',conflicts)
if __name__=='__main__':main()
