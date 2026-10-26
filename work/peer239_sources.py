"""Pure source admission for current-month direct-stock issuer peer cohorts.

No I/O, ratio, momentum, rank, score, eligibility, order or return computation.
The frozen238 helpers are used only for source axes, mapping and scalar validity.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandas as pd
import ownsector238_source_audit as original
import ownsector238_repair_sources as repair

COUNT_FIELDS = ('events','stock_cells','distinct_stock_symbols','partial_stock_cells',
    'regular_stock_cells','issuer_missing_cells','duplicate_issuer_cells','sector_missing_cells',
    'structural_supported_cells','structural_unavailable_cells','endpoint_checks',
    'valid_endpoints','missing_endpoints','invalid_endpoints','gap_cells')
RECORD_KEYS = ('query_index','column_index','date','symbol','issuer_key','original_sector',
    'sector','mapping_origin','mapping_repair','mapping_evidence','peer_indices',
    'peer_issuer_count','structural_peer_support','initial_partial','source_positions',
    'source_dates','endpoints','gap_reasons')
EVENT_KEYS = ('query_index','date','initial_partial','stock_indices','source_positions','source_dates')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _symbol(value):
    return isinstance(value,str) and bool(value) and value.strip()==value


def _inputs(panel, meta, calendar, issuer_keys, policy, limit):
    original._frames(panel)
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and
        isinstance(meta.get('symbols'),list) and len(meta['dates'])==len(meta['symbols']),
        'Metadata event axes required')
    total=len(meta['dates'])
    require(limit is None or type(limit) is int and 0<=limit<=total,'Invalid event prefix')
    n=total if limit is None else limit
    etfs=meta.get('etfs')
    require(isinstance(etfs,(list,tuple)) and all(_symbol(s) for s in etfs) and
        len(etfs)==len(set(etfs)),'Explicit unique ETF identities required')
    require(isinstance(issuer_keys,dict) and all(_symbol(s) for s in issuer_keys),'Canonical issuer-map symbol keys required')
    require(isinstance(policy,dict),'Repair policy mapping required')
    repair.check_policy(policy)
    require(isinstance(calendar,(list,tuple)) and (calendar or n==0),'Official sessions required')
    sessions=[original._iso(s) for s in calendar]
    require(all(s.weekday()<5 for s in sessions) and all(a<b for a,b in zip(sessions,sessions[1:])),
        'Official sessions must increase uniquely')
    positions={day:i for i,day in enumerate(sessions)}
    queries=[];width=None
    for i in range(n):
        day=original._iso(meta['dates'][i]);names=meta['symbols'][i]
        require(day in positions and (not queries or day>queries[-1][0]),'Queries must increase on official sessions')
        require(isinstance(names,list) and names and all(_symbol(s) for s in names) and
            len(names)==len(set(names)),'Unique full query identities required')
        if width is None:width=len(names)
        require(len(names)==width,'Query security widths differ')
        pos=positions[day];partial=meta['dates'][i]=='2016-08-17'
        require(not partial or i==0,'Named partial must be first')
        if not partial:
            require(pos>=252 and pos+1<len(sessions) and sessions[pos+1].to_period('M')!=day.to_period('M'),
                'Official endpoint history or month-end witness missing')
        queries.append((day,list(names),pos,partial))
    return queries,sessions,set(etfs)


def _issuer(symbol, issuer_keys):
    value=issuer_keys.get(symbol)
    require(value is None or _symbol(value),'Malformed whole-issuer key for '+symbol)
    return value


def inspect(panel, meta, calendar, issuer_keys, repair_policy, limit=None):
    """Inspect every direct-stock cell; gaps never silently shrink the universe."""
    queries,sessions,etfs=_inputs(panel,meta,calendar,issuer_keys,repair_policy,limit)
    records=[];events=[]
    for i,(day,names,pos,partial) in enumerate(queries):
        d=str(day.date());indices=[j for j,s in enumerate(names) if s not in etfs]
        offsets=[] if partial else [pos-252,pos-42]
        source_dates=[str(sessions[k].date()) for k in offsets]
        cohort=[]
        for j in indices:
            symbol=names[j];issuer=_issuer(symbol,issuer_keys)
            old=original._mapped(day,symbol,panel['sectors'],etfs)
            origin='frozen_FOREIGN_override' if symbol in original.FOREIGN else 'query_date_sector_panel'
            sector,change,evidence=repair.resolve(dict(is_etf=False,benchmark=old,symbol=symbol,date=d),repair_policy)
            cohort.append(dict(column_index=j,symbol=symbol,issuer=issuer,old=old,sector=sector,
                origin=origin,change=change,evidence=deepcopy(evidence)))
        issuer_counts=Counter(c['issuer'] for c in cohort if c['issuer'] is not None)
        for c in cohort:
            issuer,sector=c['issuer'],c['sector'];reasons=[]
            if issuer is None:reasons.append('missing_issuer')
            elif issuer_counts[issuer]>1:reasons.append('duplicate_issuer')
            if sector is None:reasons.append('missing_sector')
            peers=[] if issuer is None or sector is None else [other for other in cohort
                if other['issuer'] is not None and other['issuer']!=issuer and other['sector']==sector]
            peers.sort(key=lambda other:(other['issuer'],other['symbol'],other['column_index']))
            peer_count=len({other['issuer'] for other in peers})
            endpoints=[] if partial else [original.endpoint(panel['adj'],role,c['symbol'],sessions[k])
                for role,k in zip(('security_start','security_end'),offsets)]
            reasons.extend('endpoint:'+point['role']+':'+point['reason'] for point in endpoints if point['reason'] is not None)
            record=dict(query_index=i,column_index=c['column_index'],date=d,symbol=c['symbol'],
                issuer_key=issuer,original_sector=c['old'],sector=sector,mapping_origin=c['origin'],
                mapping_repair=c['change'],mapping_evidence=c['evidence'],
                peer_indices=[other['column_index'] for other in peers],peer_issuer_count=peer_count,
                structural_peer_support=peer_count>=2,initial_partial=partial,
                source_positions=offsets.copy(),source_dates=source_dates.copy(),endpoints=endpoints,
                gap_reasons=sorted(set(reasons)))
            require(tuple(record)==RECORD_KEYS,'Record schema')
            records.append(record)
        event=dict(query_index=i,date=d,initial_partial=partial,stock_indices=indices,
            source_positions=offsets.copy(),source_dates=source_dates.copy())
        require(tuple(event)==EVENT_KEYS,'Event schema');events.append(event)
    points=[p for r in records for p in r['endpoints']]
    gaps=[deepcopy(r) for r in records if r['gap_reasons']]
    counts=dict(events=len(events),stock_cells=len(records),distinct_stock_symbols=len({r['symbol'] for r in records}),
        partial_stock_cells=sum(r['initial_partial'] for r in records),regular_stock_cells=sum(not r['initial_partial'] for r in records),
        issuer_missing_cells=sum('missing_issuer' in r['gap_reasons'] for r in records),
        duplicate_issuer_cells=sum('duplicate_issuer' in r['gap_reasons'] for r in records),
        sector_missing_cells=sum('missing_sector' in r['gap_reasons'] for r in records),
        structural_supported_cells=sum(r['structural_peer_support'] for r in records),
        structural_unavailable_cells=sum(not r['structural_peer_support'] for r in records),
        endpoint_checks=len(points),valid_endpoints=sum(p['status']=='valid' for p in points),
        missing_endpoints=sum(p['status']=='missing' for p in points),invalid_endpoints=sum(p['status']=='invalid' for p in points),
        gap_cells=len(gaps))
    require(tuple(counts)==COUNT_FIELDS,'Counter schema')
    return dict(schema='peer239_source_report_v1',status='SOURCE_GAPS' if gaps else 'ADMITTED',admitted=not gaps,
        events=events,records=records,gaps=gaps,counts=counts,source_only=True,momentum_computed=False,
        rankings_computed=False,orders_computed=False,returns_computed=False,model_fits=0,strict_pit=False)


def physical_prefix(panel, meta, calendar, issuer_keys, repair_policy, n):
    """Detach source axes at the last query, plus one calendar witness session."""
    require(type(n) is int and 0<=n<=len(meta['dates']),'Invalid physical prefix')
    mm=deepcopy(meta)
    mm['dates']=mm['dates'][:n];mm['symbols']=mm['symbols'][:n]
    if isinstance(mm.get('ends'),list):mm['ends']=mm['ends'][:n]
    if n:
        cutoff=original._iso(mm['dates'][-1]);pos=calendar.index(str(cutoff.date()))
        require(pos+1<len(calendar),'Calendar witness missing')
        cc=list(calendar[:pos+2]);pp={k:frame.loc[:cutoff].copy(deep=True) for k,frame in panel.items()}
    else:
        cc=[];pp={k:frame.iloc[:0].copy(deep=True) for k,frame in panel.items()}
    return pp,mm,cc,deepcopy(issuer_keys),deepcopy(repair_policy)


def dependency_files():
    return [Path(original.__file__).resolve(),Path(repair.__file__).resolve()]
