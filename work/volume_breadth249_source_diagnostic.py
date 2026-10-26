"""Write-once source-slot diagnosis after the frozen 249 source gate failed.

This does not import any strategy/math/kernel, decode NPZ, compute returns,
rank, covariance or orders. It invokes exactly the original panel loader and
classifies the registered source windows. Raw upstream values are evidence,
not certification that a stored zero means genuine zero trading activity.
"""
from collections import Counter
from datetime import datetime,timezone
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/volume_breadth249_SOURCE_DIAGNOSTIC.json'
PLAN=ROOT/'outputs/monthly_volume_breadth249/PLAN.json'
PLAN_SHA='f3f4a6d9f00c2aacf6877fd39f0763b446873afb0588806972013735a4d8a9f8'
REVIEW=ROOT/'work/volume_breadth249_SOURCE_REVIEW.json'
REVIEW_SHA='b07b6342161f2e90767dc08d059b5710915998b93412ce27877d6e903f2b47ea'
LOADER=ROOT/'outputs/monthly_stability_monitor/inputs.py'
FAILURE=PLAN.parent/'PREPARE_FAILURE.json'
EVIDENCE={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def local(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    require(p.resolve().is_relative_to(ROOT.resolve()),'Evidence escapes local research root')
    return p

def bind(p,expected=None):
    p=local(p);digest=sha(p)
    if expected is not None:require(digest==expected,'Source hash changed: '+str(p))
    EVIDENCE[str(p.relative_to(ROOT))]=digest;return p

def read(p):return json.loads(Path(p).read_text())


def classify(x):
    if x is None or x is pd.NA:return 'missing_value',None
    if isinstance(x,(bool,np.bool_)):return 'invalid_boolean',bool(x)
    if not isinstance(x,Real):return 'invalid_nonreal',repr(x)
    try:v=float(x)
    except (TypeError,ValueError,OverflowError):return 'invalid_conversion',repr(x)
    if math.isnan(v):return 'missing_value',None
    if not math.isfinite(v):return 'nonfinite','+Infinity' if v>0 else '-Infinity'
    if v==0:return 'zero',v
    if v<0:return 'negative',v
    return 'positive',v


def scalar_probes():
    for value,status in [(None,'missing_value'),(pd.NA,'missing_value'),(np.nan,'missing_value'),(True,'invalid_boolean'),(np.bool_(False),'invalid_boolean'),('0','invalid_nonreal'),(float('inf'),'nonfinite'),(-float('inf'),'nonfinite'),(0.,'zero'),(-1.,'negative'),(1.,'positive')]:
        require(classify(value)[0]==status,'Classification self-check')


RAW_CACHE={}
def raw_tokens(path,expected,day,field,column=None):
    """Read only requested raw source fields; preserve the original CSV token."""
    p=local(path)
    if not p.is_file():return dict(path=str(p.relative_to(ROOT)),status='UNKNOWN_MISSING_LOCAL_UPSTREAM',expected_sha256=expected)
    bind(p,expected);key=str(p)
    if key not in RAW_CACHE:
        if p.suffix=='.json':
            data=read(p)
            require(isinstance(data,list),'Expected raw date records')
            rows={}
            for row in data:
                d=str(row.get('date',''))[:10]
                if d in rows:raise ValueError('Duplicate raw source date')
                rows[d]=row
            RAW_CACHE[key]=(None,rows)
        else:
            text=gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text()
            reader=csv.DictReader(io.StringIO(text));datekey=reader.fieldnames[0];rows={}
            for row in reader:
                d=str(row[datekey])[:10]
                if d in rows:raise ValueError('Duplicate raw CSV date')
                rows[d]=row
            RAW_CACHE[key]=(reader.fieldnames,rows)
    columns,rows=RAW_CACHE[key]
    row=rows.get(day);col=column or field
    ans=dict(path=str(p.relative_to(ROOT)),sha256=EVIDENCE[str(p.relative_to(ROOT))],date=day,field=col)
    if row is None:ans.update(status='MISSING_DATE',raw_token=None);return ans
    if col not in row:ans.update(status='UNKNOWN_FIELD_NOT_PRESENT',raw_token=None,available_fields=list(row));return ans
    token=row[col];ans.update(status='OBSERVED_RAW_FIELD',raw_token=token)
    for k in ('bar_valid','quarantine_reason','split','raw_per_signal'):
        if k in row:ans[k]=row[k]
    try:ans['raw_field_is_numeric_zero']=float(token)==0 if token not in ('',None) else False
    except (ValueError,TypeError,OverflowError):ans['raw_field_is_numeric_zero']=False
    return ans


def trace_value(symbol,day,field,chains,raw64):
    traces=[]
    for item in chains:
        path=local(item['path']);operation=item['operation']
        if operation=='base_alias_panel':
            actual_path=path if field=='volume' else path.with_name('adj.csv.gz')
            traces.append(dict(role='loaded_base_alias_field',operation=operation,observation=raw_tokens(actual_path,raw64[str(actual_path)],day,field,column=symbol)))
            upstream=item.get('upstream_source')
            if upstream:
                for role,pkey,hkey in [('normalized_upstream','input_file','input_sha256'),('execution_raw_upstream','raw_file','raw_sha256')]:
                    pp=ROOT/'outputs/index_union'/upstream[pkey]
                    col='volume' if field=='volume' else 'adj_close'
                    traces.append(dict(role=role,provider_symbol=item.get('provider_symbol'),operation=operation,observation=raw_tokens(pp,upstream[hkey],day,col)))
        elif path.suffix=='.json':
            col='volume' if field=='volume' else 'adjClose'
            traces.append(dict(role='loader_override_raw_json',operation=operation,observation=raw_tokens(path,item['sha256'],day,col)))
        else:
            col='Volume' if field=='volume' else 'Adj Close'
            traces.append(dict(role='loader_csv_source',operation=operation,observation=raw_tokens(path,item['sha256'],day,col)))
    return traces


def main():
    require(not OUT.exists(),'Diagnostic output already exists')
    scalar_probes();bind(Path(__file__));bind(PLAN,PLAN_SHA);bind(REVIEW,REVIEW_SHA);bind(FAILURE)
    plan=read(PLAN);review=read(REVIEW);failure=read(FAILURE)
    require(failure['phase']=='prepare' and failure['status']=='FAILED' and failure['plan_sha256']==PLAN_SHA,'Expected failed phase')
    require(not (PLAN.parent/'prepared/COMPLETION.json').exists(),'No failed phase may appear successful')
    raw=plan['raw_source_hashes'];require(len(raw)==64 and raw==review['raw_source_hashes'],'Exact original64 source map')
    for p,h in raw.items():bind(p,h)
    meta=read(bind(review['metadata_path'],review['metadata_sha256']))
    calendar=read(bind(review['calendar_path'],review['calendar_sha256']))['sessions'];lookup={d:i for i,d in enumerate(calendar)}
    require(len(meta['dates'])==122 and all(len(x)==49 for x in meta['symbols']),'Original122x49 metadata')
    for p,h in review['input_sha256'].items():
        if Path(p).suffix=='.py' or p in ('outputs/index_union/alias_resolved_inputs/MANIFEST.json','outputs/index_union/input_diagnostics/MANIFEST.json'):
            bind(p,h)
    spec=importlib.util.spec_from_file_location('source_only249_original_inputs',bind(LOADER,raw[str(LOADER)]));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);actual=dict(info['source_hashes']);actual[str(LOADER)]=sha(LOADER)
    require(actual==raw,'Original loader raw source closure exact')
    print(json.dumps(dict(stage='source_only_original64_verified_and_loaded',source_files=64)),flush=True)
    require(str(panel['adj'].index[0].date())==review['panel_first_session']=='2008-01-02','Original actual source origin')
    for field in ('adj','volume'):
        frame=panel[field];require(frame.index.is_unique and frame.columns.is_unique,'Source axes unique')
    chains=review['source_lineage']['chains'];issues={};events=[];slot_totals=Counter();identity=[]
    bounds={x['symbol']:x for x in review['listing_bounds']}
    for i,(date,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=lookup[date];clock=review['clock_events'][i]
        require(clock['query_index']==i and clock['date']==date and clock['symbols']==names and clock['query_position']==t,'Clock metadata identity')
        partial=date=='2016-08-17';require(not partial or i==0,'Named partial only')
        require(t>=356 and t+1<len(calendar) and (partial or calendar[t+1][:7]!=date[:7]),'Official formation clock')
        window=dict(price_start=calendar[t-294],price_end=calendar[t-42],volume_start=calendar[t-356],volume_end=calendar[t-42])
        require(all(clock[k]==v for k,v in window.items()),'Frozen source window')
        ec=Counter();badmembers=set();missingmembers=set();hardmembers=set();refs=[]
        if not partial:
            for j,symbol in enumerate(names):
                require(symbol in chains,'Source lineage missing')
                bound=bounds.get(symbol)
                for field,start,length in [('adj',294,253),('volume',356,315)]:
                    frame=panel[field];days=calendar[t-start:t-41]
                    require(len(days)==length,'Required window length')
                    if symbol not in frame.columns:values=[None]*length;positions=[-1]*length
                    else:
                        positions=frame.index.get_indexer(pd.to_datetime(days))
                        col=frame.columns.get_loc(symbol)
                        values=[frame.iat[pos,col] if pos>=0 else None for pos in positions]
                    for offset,(d,x,pos) in enumerate(zip(days,values,positions)):
                        if symbol not in frame.columns:status,value='missing_symbol',None
                        elif pos<0:status,value='missing_row',None
                        else:status,value=classify(x)
                        slot_totals[(field,status)]+=1;ec[(field,status)]+=1
                        if status=='positive':
                            if bound and bound['available_from']<=date and d<bound['not_before']:
                                identity.append(dict(query_index=i,date=date,column_index=j,symbol=symbol,field=field,source_date=d,value=value,not_before=bound['not_before'],available_from=bound['available_from']))
                            continue
                        badmembers.add(j)
                        if status.startswith('missing'):missingmembers.add(j)
                        else:hardmembers.add(j)
                        key=(field,symbol,d,status,json.dumps(value,sort_keys=True));rid='issue_'+str(len(issues)).zfill(5)
                        if key not in issues:issues[key]=dict(issue_id=rid,field=field,symbol=symbol,source_date=d,classification=status,loaded_value=value,lineage_id=symbol,occurrences=[])
                        item=issues[key];item['occurrences'].append(dict(query_index=i,date=date,column_index=j,window_start=days[0],window_end=days[-1],offset=offset))
                        refs.append(item['issue_id'])
        events.append(dict(query_index=i,date=date,symbols=names,initial_partial=partial,source_window=window,required_security_cells=0 if partial else 49,
          required_price_slots=0 if partial else 49*253,required_volume_slots=0 if partial else 49*315,
          status='BYPASSED_NAMED_PARTIAL' if partial else ('HARD_INVALID_SOURCE' if hardmembers else ('ORDINARY_MISSING_SOURCE' if missingmembers else 'POSITIVE_FINITE_SOURCE_SLOTS_ONLY')),
          affected_member_indices=sorted(badmembers),ordinary_missing_member_indices=sorted(missingmembers),hard_invalid_member_indices=sorted(hardmembers),
          counts={field:{status:count for (f,status),count in sorted(ec.items()) if f==field} for field in ('adj','volume')},issue_ids=sorted(set(refs))))
        if i in (1,52,87,121):print(json.dumps(dict(stage='source_slot_scan',query_index=i,date=date,unique_issues=len(issues))),flush=True)
    allissues=list(issues.values());hard=[x for x in allissues if not x['classification'].startswith('missing')]
    for x in hard:x['raw_lineage_trace']=trace_value(x['symbol'],x['source_date'],x['field'],chains[x['symbol']],raw)
    oldpath=ROOT/'work/volume195_SOURCE_REVIEW.json'
    prior_note=dict(path=str(oldpath.relative_to(ROOT)),scope='Older273-window evidence is a known-risk lead only, not this315-window validation.')
    if oldpath.exists():
        bind(oldpath);oldreview=read(oldpath)
        def matching(value):
            if isinstance(value,dict):
                if value.get('symbol')=='ARKK' and ('2016-08-31' in str(value.get('date','')) or '2016-08-17' in str(value.get('date',''))):return [value]
                return sum((matching(v) for v in value.values()),[])
            if isinstance(value,list):return sum((matching(v) for v in value),[])
            return []
        prior_note.update(sha256=sha(oldpath),matching_records=matching(oldreview))
    hard_events=[e['query_index'] for e in events if e['hard_invalid_member_indices']];missing_events=[e['query_index'] for e in events if e['ordinary_missing_member_indices']]
    first=events[min(hard_events)] if hard_events else None
    first_issues=[x for x in hard if any(z['query_index']==first['query_index'] for z in x['occurrences'])] if first else []
    for p,h in raw.items():require(sha(p)==h,'Raw source changed during diagnosis')
    for p,h in list(EVIDENCE.items()):require(sha(ROOT/p)==h,'Evidence changed during diagnosis')
    result=dict(schema='volume_breadth249_source_diagnostic_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='SOURCE_ONLY_DIAGNOSTIC_NOT_STRATEGY_ADMISSION',
      passed=True,source_admitted=False,strict_pit=False,plan_sha256=PLAN_SHA,source_review_sha256=REVIEW_SHA,failed_prepare=read(FAILURE),
      summary=dict(events=122,inventory_cells=122*49,formal_events=121,required_security_cells=121*49,partial_bypassed_events=1,required_price_slots=121*49*253,required_volume_slots=121*49*315,
        unique_abnormal_source_observations=len(allissues),unique_hard_invalid_observations=len(hard),abnormal_slot_occurrences=sum(len(x['occurrences']) for x in allissues),
        hard_invalid_slot_occurrences=sum(len(x['occurrences']) for x in hard),hard_invalid_events=len(hard_events),ordinary_missing_events=len(missing_events),
        any_abnormal_events=len(set(hard_events)|set(missing_events)),positive_complete_events=sum(e['status']=='POSITIVE_FINITE_SOURCE_SLOTS_ONLY' for e in events),visible_listing_positive_contradictions=len(identity),
        classifications={field:{status:count for (f,status),count in sorted(slot_totals.items()) if f==field} for field in ('adj','volume')}),
      first_hard_failure=dict(query_index=first['query_index'],date=first['date'],hard_issue_ids=[x['issue_id'] for x in first_issues],
        hard_slot_count=sum(sum(z['query_index']==first['query_index'] for z in x['occurrences']) for x in first_issues)) if first else None,
      failure_execution_scope=dict(original_cached_forecast_ranks_and_pre_may_have_been_reconstructed=True,full_original_core_source_reconstruction_precedes_failure=True,
        earlier_event_new_raw_covariances_may_have_been_computed=True,earlier_formal_events_before_first_hard_failure=sum(not e['initial_partial'] for e in events[:first['query_index']]) if first else None,
        no_global_zero_features_claim=True,new_target_ranks_reached=False,new_episode_calls_reached=False,new_orders_reached=False,new_portfolio_returns_reached=False,
        basis='Frozen kernel scans all members of one event then computes that event raw measurements. Source callback and every target rank/episode/order occur only after all events. This diagnosis does not rerun those functions.'),
      raw_source_hashes={str(local(p).relative_to(ROOT)):h for p,h in raw.items()},raw_source_hashes_verified_before_and_after=True,
      loader=dict(path=str(LOADER.relative_to(ROOT)),sha256=sha(LOADER),api='load_panel(meta[etfs])',equivalence='Exact function invoked by study.arch.old.panel_load, plus the wrapper loader SHA; no study.load, scope.load or NPZ decode.',
        actual_origin=str(panel['adj'].index[0].date()),actual_end=str(panel['adj'].index[-1].date()),rows=len(panel['adj']),frozen_loader_price_splice_operations_preserved=True),
      input_sha256=EVIDENCE,events=events,issues=allissues,lineage_by_symbol={s:chains[s] for s in sorted(set(x['symbol'] for x in allissues))},identity_contradictions=identity,
      prior_known_zero_evidence=prior_note,
      limitations=['Stored raw/provider zero is not proof of genuine zero trading or a provider placeholder; economic semantics remain UNKNOWN.',
        'No normalization/forward fill/zero replacement/source repair or policy relaxation is performed.',
        'Ordinary missing and invalid source slots are separate. The whole-event counts describe source eligibility only, not portfolio holdings or return effects.',
        'Named initial partial has no required source values under the frozen contract and is not numerically scanned.',
        'Alias and adjusted-volume continuity retain the frozen research assumptions; hash equality is not strict point-in-time certification.',
        'Raw lineage traces verify only recorded same-day fields in available local evidence. They do not certify full corporate-action histories.'],
      work_performed=dict(source_prices_and_volumes_loaded=True,required_source_slots_classified=True,raw_upstream_zero_fields_checked=True,npz_decoded=False,
        original_core_recomputed=False,new_covariances_computed=False,new_targets_or_ranks_computed=False,new_orders_computed=False,new_portfolio_returns_computed=False,
        research_phase_reexecuted=False,network_requests=0,observer_files_modified=False,frozen_files_modified=False))
    require(sum(result['summary']['classifications']['adj'].values())==121*49*253 and sum(result['summary']['classifications']['volume'].values())==121*49*315,'Complete source-slot accounting')
    with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(OUT),sha256=sha(OUT),summary=result['summary'],first_hard_failure=result['first_hard_failure'])),flush=True)


if __name__=='__main__':main()
