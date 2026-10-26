"""Independent mapping/endpoint support audit; no ratios or strategy decisions.

Only supplied source cells are classified. The frozen mapping tables are checked
as source literals, without importing a producer, strategy, model or engine.
"""
import ast
from collections import Counter
from datetime import date
import hashlib
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MAPPING=ROOT/'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA='ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTOR={'Basic Materials':'XLB','Communication Services':'XLC','Energy':'XLE',
    'Financial Services':'XLF','Industrials':'XLI','Technology':'XLK',
    'Consumer Defensive':'XLP','Real Estate':'XLRE','Utilities':'XLU',
    'Healthcare':'XLV','Consumer Cyclical':'XLY'}
FOREIGN={'ASML':'XLK','ARM':'XLK','TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK',
    'TM':'XLY','SHEL':'XLE','SKHY':'XLK','BHP':'XLB','HSBC':'XLF','SMFG':'XLF',
    'NVS':'XLV','SNDK':'XLK'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapping_constants():
    require(sha(MAPPING)==MAPPING_SHA,'Frozen mapping source changed')
    found={}
    for node in ast.parse(MAPPING.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SECTOR','FOREIGN'):
                    require(target.id not in found,'Duplicate mapping constant')
                    found[target.id]=ast.literal_eval(node.value)
    require(found==dict(SECTOR=SECTOR,FOREIGN=FOREIGN),'Frozen mapping table literals differ')
    return dict(SECTOR=SECTOR.copy(),FOREIGN=FOREIGN.copy())


def dependency_files():return (Path(__file__).resolve(),MAPPING)


def iso(raw):
    require(isinstance(raw,str),'Canonical ISO date required')
    d=date.fromisoformat(raw);require(d.isoformat()==raw,'Canonical ISO date required')
    return pd.Timestamp(d)


def frame_schema(frame,label):
    require(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex),label+' DataFrame required')
    idx=frame.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and idx.equals(idx.normalize()),label+' invalid date axis')
    require(frame.columns.is_unique and all(isinstance(s,str) and s and s.strip()==s for s in frame.columns),label+' invalid identity axis')


def inputs(panel,meta,calendar,eligible,limit):
    require(isinstance(panel,dict) and set(panel)=={'adj','sectors'},'Exactly adj/sectors sources required')
    for label,frame in panel.items():frame_schema(frame,label)
    e=np.asarray(eligible);require(e.ndim==2 and e.shape[1]>0,'Eligibility matrix requires nonempty width')
    total,width=e.shape
    require(limit is None or type(limit) is int and 0<=limit<=total,'Invalid prefix')
    n=total if limit is None else limit
    require(all(isinstance(x,(bool,np.bool_)) for x in e[:n].flat),'Consumed eligibility values must be exact bool')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and isinstance(meta.get('symbols'),list)
        and len(meta['dates'])==len(meta['symbols'])==total,'Metadata source axes differ')
    etfs=meta.get('etfs')
    require(isinstance(etfs,(list,tuple)) and len(set(etfs))==len(etfs) and all(isinstance(s,str) and s and s.strip()==s for s in etfs),'Explicit ETF identities required')
    require(isinstance(calendar,(list,tuple)) and (calendar or n==0),'Official calendar required')
    sessions=[iso(x) for x in calendar]
    require(all(d.weekday()<5 for d in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),'Invalid official session order')
    lookup={d:i for i,d in enumerate(sessions)};queries=[]
    for i,(raw,names) in enumerate(zip(meta['dates'][:n],meta['symbols'][:n])):
        day=iso(raw)
        require(day in lookup and (not queries or day>queries[-1][0]),'Queries must be increasing official sessions')
        require(isinstance(names,list) and len(names)==width and len(set(names))==width and all(isinstance(s,str) and s and s.strip()==s for s in names),'Unique full query identities required')
        t=lookup[day];partial=raw=='2016-08-17'
        if partial:require(i==0,'Named initial partial only first')
        else:require(t>=252 and t+1<len(sessions) and sessions[t+1].to_period('M')!=day.to_period('M'),'Required official history/month-end witness missing')
        queries.append((day,names.copy(),t))
    return e[:n].astype(bool,copy=True),queries,sessions,list(etfs),n,width


def benchmark_for(day,symbol,sectors,etfs):
    if symbol in etfs:return None
    if symbol in FOREIGN:return FOREIGN[symbol]
    if day not in sectors.index or symbol not in sectors.columns:return None
    value=sectors.at[day,symbol]
    try:return SECTOR.get(value)
    except TypeError as exc:raise ValueError('Malformed sector classification') from exc


def endpoint(frame,role,symbol,day):
    record=dict(role=role,symbol=symbol,date=str(day.date()),status=None,reason=None)
    if symbol is None:status,reason='missing','missing_benchmark_mapping'
    elif day not in frame.index:status,reason='missing','missing_row'
    elif symbol not in frame.columns:status,reason='missing','missing_column'
    else:
        value=frame.at[day,symbol]
        if value is None or value is pd.NA:status,reason='missing','missing_value'
        elif isinstance(value,(bool,np.bool_)):status,reason='invalid','invalid_boolean'
        elif not isinstance(value,numbers.Real):status,reason='invalid','invalid_nonreal'
        else:
            try:converted=float(value)
            except (ValueError,TypeError,OverflowError):status,reason='invalid','invalid_numeric_conversion'
            else:
                if math.isnan(converted):status,reason='missing','missing_value'
                elif not math.isfinite(converted):status,reason='invalid','invalid_nonfinite'
                elif converted<=0.:status,reason='invalid','invalid_nonpositive'
                else:status,reason='valid',None
    record.update(status=status,reason=reason);return record


def record_check(actual,expected,label='source audit'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+' keys')
        for key,value in expected.items():record_check(actual[key],value,label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+' list')
        for i,(a,b) in enumerate(zip(actual,expected)):record_check(a,b,label+'/'+str(i))
    else:require(type(actual) is type(expected) and actual==expected,label+' exact value/type')

COUNT_FIELDS=('events','symbols_per_event','checked_cells','initial_partial_events','regular_events',
    'original_eligible_cells','required_cells','stock_required_cells','etf_required_cells',
    'mapping_available_cells','mapping_missing_cells','security_benchmark_endpoint_checks',
    'market_endpoint_checks','valid_endpoints','missing_endpoints','invalid_endpoints',
    'admitted_cells','gap_cells','market_gap_events')
FLAGS=('passed','independent_source_rebuild','original_eligibility_scope_checked',
    'named_partial_bypass_checked','official_endpoint_clock_checked','exact_query_mapping_clock_checked',
    'etf_spy_dispatch_checked','foreign_override_precedence_checked','sector_table_precedence_checked',
    'pinned_mapping_constants_checked','supplied_mapper_agreement_checked',
    'missing_and_invalid_evidence_separated','all_required_endpoints_checked','regular_market_endpoints_checked',
    'exact_report_and_reasons_checked','descriptive_counts_checked','no_future_source_consumption',
    'no_strategy_math_computed')


def independent_sources(panel,meta,calendar,eligible,benchmark_map,limit=None):
    mapping_constants()
    e,queries,sessions,etfs,n,width=inputs(panel,meta,calendar,eligible,limit)
    require(callable(benchmark_map),'Original benchmark mapping callable required')
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,symbols_per_event=width,checked_cells=n*width)
    events=[];records=[];gaps=[]
    for i,(day,names,t) in enumerate(queries):
        d=str(day.date());partial=d=='2016-08-17';active=not partial
        old=[j for j in range(width) if e[i,j]];required=[] if partial else old.copy()
        points=[];dates=[];market=[];mapping={};cell_good=0;cell_gaps=0
        if active:
            points=[t-252,t-42];days=[sessions[k] for k in points];dates=[str(x.date()) for x in days]
            mapping=benchmark_map(d,names.copy(),panel['sectors'].loc[:day].copy(deep=True),etfs.copy())
            require(isinstance(mapping,dict) and set(mapping)==set(names),'Mapper must return exact query identities')
            require(all(x is None or isinstance(x,str) and x and x.strip()==x for x in mapping.values()),'Malformed mapping identity')
            for j in required:
                expected=benchmark_for(day,names[j],panel['sectors'],etfs)
                require(mapping[names[j]]==expected,'Mapper disagrees with frozen table/precedence')
            market=[endpoint(panel['adj'],role,'SPY',dd) for role,dd in zip(('market_start','market_end'),days)]
            reasons=list(dict.fromkeys(x['reason'] for x in market if x['reason'] is not None))
            if reasons:
                gaps.append(dict(scope='market',query_index=i,column_index=None,date=d,symbol='SPY',benchmark='SPY',reasons=reasons))
                counts['market_gap_events']+=1
            for slot in market:counts[slot['status']+'_endpoints']+=1
            counts['market_endpoint_checks']+=2
            for j in required:
                symbol=names[j];etf=symbol in etfs
                benchmark='SPY' if etf else mapping[symbol]
                origin='original_META.etfs_SPY' if etf else 'frozen_FOREIGN_override' if symbol in FOREIGN else 'query_date_sector_panel'
                slots=[endpoint(panel['adj'],role,sym,dd) for role,sym,dd in (
                    ('security_start',symbol,days[0]),('security_end',symbol,days[1]),
                    ('benchmark_start',benchmark,days[0]),('benchmark_end',benchmark,days[1]))]
                reasons=[] if benchmark is not None else ['missing_benchmark_mapping']
                for slot in slots:
                    if slot['reason'] is not None and slot['reason'] not in reasons:reasons.append(slot['reason'])
                    counts[slot['status']+'_endpoints']+=1
                admitted=not reasons
                records.append(dict(query_index=i,column_index=j,date=d,symbol=symbol,is_etf=etf,
                    benchmark=benchmark,mapping_source=origin,mapping_status='available' if benchmark is not None else 'missing',
                    endpoints=slots,admitted=admitted,gap_reasons=reasons))
                if not admitted:
                    gaps.append(dict(scope='cell',query_index=i,column_index=j,date=d,symbol=symbol,benchmark=benchmark,reasons=reasons));cell_gaps+=1
                else:cell_good+=1
                counts['etf_required_cells' if etf else 'stock_required_cells']+=1
                counts['mapping_available_cells' if benchmark is not None else 'mapping_missing_cells']+=1
                counts['security_benchmark_endpoint_checks']+=4
        counts['initial_partial_events']+=int(partial);counts['regular_events']+=int(active)
        counts['original_eligible_cells']+=len(old);counts['required_cells']+=len(required)
        counts['admitted_cells']+=cell_good;counts['gap_cells']+=cell_gaps
        events.append(dict(query_index=i,date=d,initial_partial=partial,active=active,
            original_eligible_indices=old,required_indices=required,source_positions=points,source_dates=dates,
            market_endpoints=market,required_count=len(required),admitted_count=cell_good,gap_count=cell_gaps))
    admitted=not gaps
    return dict(schema='ownsector238_source_inspection_v1',status='ADMITTED' if admitted else 'NOT_ADMITTED',
        admitted=admitted,counts=counts,events=events,records=records,gaps=gaps,source_only=True,
        momentum_computed=False,eligibility_changed=False,rankings_computed=False,orders_computed=False,
        returns_computed=False,model_fits=0,strict_pit=False)


def check_sources(panel,meta,calendar,eligible,benchmark_map,report,limit=None):
    expected=independent_sources(panel,meta,calendar,eligible,benchmark_map,limit)
    record_check(report,expected)
    return dict(**{key:True for key in FLAGS},counts=expected['counts'].copy(),status=expected['status'],
        admitted=expected['admitted'],checked_events=expected['counts']['events'],
        checked_required_records=len(expected['records']),checked_endpoint_records=expected['counts']['security_benchmark_endpoint_checks']+expected['counts']['market_endpoint_checks'],
        producer_modules_imported=False,momentum_computed=False,eligibility_changed=False,
        rankings_computed=False,orders_computed=False,returns_computed=False,model_fits=0,
        mapping_source_sha256=MAPPING_SHA,numerical_tolerance=0.)


verify_sources=check_sources
