"""Read-only complete slot inventory after registered252 source-gate failure. No factors/orders/returns."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
import monthly_peer_path252_host as h
import monthly_peer_path252_kernel as k

PLAN_SHA='1f84a1976cbe61cb7079f089689837a7ac8ed745eaa1f1f31d02472ec4a425bf'
DEST=h.ROOT/'work/monthly_peer_path252_SOURCE_GAPS.json'
h.require(not DEST.exists(),'Source gap audit write-once')
plan=h.gate(PLAN_SHA)
study=h.old.local();meta,arrays,px,op=study.load()
raw,info=study.arch.old.panel_load(meta['etfs'])
h.require(study.raw_source_hashes(info)==plan['raw_source_hashes'],'Original source binding')
h.old.verify(plan['raw_source_hashes'])
source=h.read(h.SOURCE/'REPORT.json');calendar=h.read(h.old.CALENDAR)['sessions']
frame,sessions,rows,width=k._metadata({'adj':raw['adj']},meta,calendar,arrays,source,None)
slots={};counts=Counter();gaps=[];by_symbol=defaultdict(Counter);boundaries={}
for i,(day,names,t,partial,ev,records,mature,supported) in enumerate(rows):
    dates=[] if partial else list(sessions[t-252:t-42+1])
    h.require(partial or len(dates)==211,'Complete official clock')
    for r in records:
        counts['stock_cells']+=1
        if r['history_bound'] is not None:boundaries[r['symbol']]=r['history_bound']
        for d in dates:
            symbol=r['symbol'];date=str(d.date());key=(symbol,date)
            if key not in slots:
                try:
                    value,status,reason=k._slot(frame,symbol,d)
                except ValueError as exc:
                    value,status,reason=None,'invalid',str(exc)
                slots[key]=(status,reason)
            status,reason=slots[key];counts['price_observations']+=1;by_symbol[symbol][status]+=1
            before=r['history_bound'] is not None and date<r['history_bound']['not_before']
            issue=('pre_bound_observation' if before and status!='missing' else
                   'post_bound_or_mature_missing' if not before and status!='valid' else None)
            if issue:
                gaps.append(dict(query_index=i,query_date=str(day.date()),column_index=r['column_index'],symbol=symbol,date=date,
                    status=status,reason=reason,issue=issue,history_status=r['history_status'],bound=r['history_bound'],
                    mature=r['column_index'] in mature,supported=r['column_index'] in supported))
unique={}
for r in gaps:
    key=(r['symbol'],r['date'],r['issue'])
    if key not in unique:unique[key]=dict(symbol=r['symbol'],date=r['date'],issue=r['issue'],status=r['status'],reason=r['reason'],query_indices=[])
    unique[key]['query_indices'].append(r['query_index'])
report=dict(schema='peer_path252_source_gap_inventory_v1',created_utc=datetime.now(timezone.utc).isoformat(),source_only=True,
    plan_sha256=PLAN_SHA,source_report_sha256=h.sha(h.SOURCE/'REPORT.json'),script_sha256=h.sha(Path(__file__)),
    raw_source_hashes=plan['raw_source_hashes'],events=len(rows),counts=dict(counts),unique_slots=len(slots),
    gap_occurrences=len(gaps),unique_gap_count=len(unique),gaps=gaps,unique_gaps=list(unique.values()),
    status_by_symbol={s:dict(c) for s,c in sorted(by_symbol.items())},listing_lower_bounds=boundaries,
    features_computed=False,orders_computed=False,returns_computed=False,model_fits=0,admitted=False,
    note='Diagnostics preserve frozen252 strict rule; no source substitution, cohort change, or strategy admission.')
h.dump(DEST,report)
print(json.dumps({a:report[a] for a in ['events','counts','unique_slots','gap_occurrences','unique_gap_count','unique_gaps']}),flush=True)
