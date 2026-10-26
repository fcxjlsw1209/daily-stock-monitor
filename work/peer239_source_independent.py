"""Independent full-stock source inspection: no momentum or strategy arithmetic.

The supplied issuer dictionary has no effective-date evidence fields. Its
publication/vintage admission is a host responsibility; this checker verifies
the exact supplied identities, peer construction and consumed source status.
"""
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import ownsector238_source_independent as frozen

FROZEN_SHA = 'c4f5b1400a77b71cf72ef46cd254f74b4f75ac4e662d52b1f19a68e0af6dc705'
COUNT_FIELDS = ('events','stock_cells','distinct_stock_symbols','partial_stock_cells',
    'regular_stock_cells','issuer_missing_cells','duplicate_issuer_cells',
    'sector_missing_cells','structural_supported_cells','structural_unavailable_cells',
    'endpoint_checks','valid_endpoints','missing_endpoints','invalid_endpoints','gap_cells')
FLAGS = ('passed','independent_source_rebuild','all_direct_stocks_before_eligibility',
    'etfs_excluded_from_peers','supplied_issuer_identity_checked','whole_issuer_exclusion_checked',
    'duplicate_issuer_gaps_checked','query_date_sector_mapping_checked',
    'foreign_override_precedence_checked','fixed_dated_repairs_checked','all_xlc_keys_normalized',
    'distinct_peer_count_and_order_checked','structural_shortfall_separated_from_source_gaps',
    'partial_identity_without_endpoints_checked','official_endpoint_clock_checked',
    'all_regular_stock_endpoints_checked','missing_and_invalid_evidence_separated',
    'exact_report_reasons_and_counts_checked','no_future_source_consumption',
    'no_strategy_math_computed')
RULES = {
    'AZN': ('XLV','2023-03-10','2023-03-11',None),
    'BUD': ('XLP','2016-03-14','2016-03-15',None),
    'CHL': ('VOX','2016-04-27','2016-04-28',None),
    'GOOGL': ('XLK','2016-02-11','2016-02-12','2018-09-24'),
    'PDD': ('XLY','2020-04-24','2020-04-25',None),
    'TSLA': ('XLY','2020-02-13','2020-02-14',None),
}
require = frozen.require
record_check = frozen.record_check


def dependency_files():
    return tuple(dict.fromkeys((Path(__file__).resolve(),Path(frozen.__file__).resolve())+
                              frozen.dependency_files()))


def policy_rules(policy):
    require(isinstance(policy,dict) and policy.get('schema')=='ownsector238_repair_source_policy_v1',
            'Fixed repair policy required')
    rows=policy.get('missing_mapping_rules')
    require(isinstance(rows,list) and len(rows)==6,'Six fixed dated rules required')
    result={}
    for row in rows:
        require(isinstance(row,dict),'Repair rule dictionary')
        name=row.get('symbol')
        require(isinstance(name,str) and name in RULES and name not in result,'Unknown/duplicate rule')
        fields=tuple(row.get(k) for k in ('benchmark','public_date','valid_from','valid_until_exclusive'))
        require(fields==RULES[name],'Fixed rule or date changed')
        start=frozen.iso(fields[2]);publication=frozen.iso(fields[1])
        require(start==publication+pd.Timedelta(days=1),'Following-calendar-date clock')
        if fields[3] is not None:require(frozen.iso(fields[3])>start,'Invalid rule end')
        refs=row.get('source_ids')
        require(isinstance(refs,list) and len(refs)>0 and all(isinstance(v,str) and v for v in refs),
                'Explicit rule evidence required')
        require(isinstance(row.get('source_file'),str) and bool(row['source_file']),'Rule source path required')
        result[name]=row
    communication=policy.get('communication_benchmark')
    require(isinstance(communication,dict) and communication.get('old')=='XLC' and
            communication.get('new')=='VOX' and communication.get('price_field')=='adjClose' and
            communication.get('fill_or_splice') is False,'Fixed communication normalization required')
    return result


def resolve(original,name,day,rules):
    if original=='XLC':return 'VOX','full_period_communication_proxy',None
    if original is not None:return original,'unchanged',None
    row=rules.get(name)
    if row is not None and day>=frozen.iso(row['valid_from']) and (
            row['valid_until_exclusive'] is None or day<frozen.iso(row['valid_until_exclusive'])):
        return row['benchmark'],'dated_business_proxy',deepcopy(row['source_ids'])
    return None,'unresolved',None


def source_axes(panel,meta,calendar,issuer_keys,limit):
    require(isinstance(panel,dict) and set(panel)=={'adj','sectors'},'Exact adj/sectors source keys')
    for label,frame in panel.items():frozen.frame_schema(frame,label)
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and
            isinstance(meta.get('symbols'),list) and len(meta['dates'])==len(meta['symbols']),
            'Explicit aligned metadata axes')
    total=len(meta['dates'])
    require(limit is None or type(limit) is int and 0<=limit<=total,'Invalid prefix')
    n=total if limit is None else limit
    etfs=meta.get('etfs')
    require(isinstance(etfs,(list,tuple)) and all(isinstance(s,str) and s and s.strip()==s for s in etfs)
            and len(set(etfs))==len(etfs),'Explicit unique ETF list')
    require(isinstance(issuer_keys,dict),'Issuer metadata dictionary required')
    require(isinstance(calendar,(list,tuple)) and (len(calendar)>0 or n==0),'Official session calendar required')
    sessions=[frozen.iso(x) for x in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),
            'Invalid official calendar axis')
    positions={d:i for i,d in enumerate(sessions)};queries=[];width=None
    for i in range(n):
        day=frozen.iso(meta['dates'][i]);names=meta['symbols'][i]
        require(isinstance(names,list) and len(names)>0 and
                all(isinstance(s,str) and s and s.strip()==s for s in names) and len(set(names))==len(names),
                'Invalid original monthly identities')
        if width is None:width=len(names)
        require(len(names)==width,'Rectangular monthly candidate axes')
        require(day in positions and (not queries or day>queries[-1][0]),'Ordered official query sessions')
        t=positions[day];partial=meta['dates'][i]=='2016-08-17'
        if partial:require(i==0,'Named initial partial must be first')
        else:
            require(t>=252 and t+1<len(sessions) and
                    sessions[t+1].to_period('M')!=day.to_period('M'),'Official endpoint history/month-end witness missing')
        queries.append((day,names.copy(),t,partial))
    return queries,sessions,set(etfs)


def inspect(panel,meta,calendar,issuer_keys,repair_policy,limit=None):
    require(frozen.sha(frozen.__file__)==FROZEN_SHA,'Frozen source helper changed')
    frozen.mapping_constants()
    rules=policy_rules(repair_policy)
    queries,sessions,etfs=source_axes(panel,meta,calendar,issuer_keys,limit)
    counters=dict.fromkeys(COUNT_FIELDS,0);counters['events']=len(queries)
    events=[];records=[];gaps=[];seen_symbols=set()
    for i,(day,names,t,partial) in enumerate(queries):
        raw=str(day.date());stock=[j for j,s in enumerate(names) if s not in etfs]
        offsets=[] if partial else [t-252,t-42]
        days=[sessions[k] for k in offsets];dates=[str(d.date()) for d in days]
        events.append(dict(query_index=i,date=raw,initial_partial=partial,stock_indices=stock.copy(),
                           source_positions=offsets.copy(),source_dates=dates.copy()))
        identities={};sectors={};originals={};operations={};evidence={};origins={}
        for j in stock:
            name=names[j];issuer=issuer_keys.get(name)
            require(issuer is None or isinstance(issuer,str) and issuer and issuer.strip()==issuer,
                    'Malformed supplied issuer key')
            identities[j]=issuer
            originals[j]=frozen.benchmark_for(day,name,panel['sectors'],etfs)
            sectors[j],operations[j],evidence[j]=resolve(originals[j],name,day,rules)
            origins[j]='frozen_FOREIGN_override' if name in frozen.FOREIGN else 'query_date_sector_panel'
        duplicate=Counter(v for v in identities.values() if v is not None)
        for j in stock:
            name=names[j];issuer=identities[j];sector=sectors[j];reasons=[]
            if issuer is None:reasons.append('missing_issuer')
            elif duplicate[issuer]>1:reasons.append('duplicate_issuer')
            if sector is None:reasons.append('missing_sector')
            peers=[]
            if issuer is not None and sector is not None:
                peers=[k for k in stock if identities[k] is not None and identities[k]!=issuer and sectors[k]==sector]
                peers.sort(key=lambda k:(identities[k],names[k],k))
            peer_count=len({identities[k] for k in peers});supported=peer_count>=2
            slots=[frozen.endpoint(panel['adj'],role,name,d) for role,d in zip(('security_start','security_end'),days)]
            for slot in slots:
                counters[slot['status']+'_endpoints']+=1
                if slot['reason'] is not None:reasons.append('endpoint:'+slot['role']+':'+slot['reason'])
            reasons=sorted(set(reasons))
            record=dict(query_index=i,column_index=j,date=raw,symbol=name,issuer_key=issuer,
                original_sector=originals[j],sector=sector,mapping_origin=origins[j],
                mapping_repair=operations[j],mapping_evidence=deepcopy(evidence[j]),
                peer_indices=peers,peer_issuer_count=peer_count,structural_peer_support=supported,
                initial_partial=partial,source_positions=offsets.copy(),source_dates=dates.copy(),
                endpoints=slots,gap_reasons=reasons)
            records.append(record)
            if reasons:gaps.append(deepcopy(record))
            seen_symbols.add(name);counters['stock_cells']+=1
            counters['partial_stock_cells' if partial else 'regular_stock_cells']+=1
            counters['issuer_missing_cells']+=int(issuer is None)
            counters['duplicate_issuer_cells']+=int(issuer is not None and duplicate[issuer]>1)
            counters['sector_missing_cells']+=int(sector is None)
            counters['structural_supported_cells' if supported else 'structural_unavailable_cells']+=1
            counters['endpoint_checks']+=len(slots);counters['gap_cells']+=int(bool(reasons))
    counters['distinct_stock_symbols']=len(seen_symbols)
    admitted=len(gaps)==0
    return dict(schema='peer239_source_report_v1',status='ADMITTED' if admitted else 'SOURCE_GAPS',
        admitted=admitted,events=events,records=records,gaps=gaps,counts=counters,source_only=True,
        momentum_computed=False,rankings_computed=False,orders_computed=False,returns_computed=False,
        model_fits=0,strict_pit=False)


def check_sources(panel,meta,calendar,issuer_keys,repair_policy,report,limit=None):
    expected=inspect(panel,meta,calendar,issuer_keys,repair_policy,limit)
    record_check(report,expected,'peer239 source report')
    return dict(**{key:True for key in FLAGS},counts=expected['counts'].copy(),
        status=expected['status'],admitted=expected['admitted'],
        checked_records=len(expected['records']),checked_endpoints=expected['counts']['endpoint_checks'],
        producer_modules_imported=False,real_source_files_loaded=False,
        issuer_publication_evidence_checked_by_this_api=False,issuer_evidence_clock_is_host_responsibility=True,
        momentum_computed=False,rankings_computed=False,orders_computed=False,returns_computed=False,
        model_fits=0,numerical_tolerance=0.,frozen_helper_sha256=FROZEN_SHA,
        mapping_source_sha256=frozen.MAPPING_SHA)


independent_sources=inspect
verify_sources=check_sources
