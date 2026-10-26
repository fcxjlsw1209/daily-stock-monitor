"""Prospective metadata projection of the immutable full-stock source audit.

No prices, ratios, ranks, orders, returns or strategy eligibility are loaded.
"""
from copy import deepcopy
from collections import Counter
from datetime import date


def require(ok,message):
    if not ok:raise ValueError(message)


def iso(value):
    require(type(value) is str and date.fromisoformat(value).isoformat()==value,'Canonical ISO date')
    return value


def project(source, meta, policy, limit=None):
    require(source.get('schema')=='peer239_source_report_v1' and source.get('source_only') is True,'Original source report required')
    require(all(source.get(k) is False for k in ('momentum_computed','rankings_computed','orders_computed','returns_computed')) and type(source.get('model_fits')) is int and source['model_fits']==0,'Source-only input')
    require(policy.get('schema')=='peer239_history_bound_policy_v1','Explicit new history policy')
    full=len(meta['dates']);n=full if limit is None else limit
    require(type(n) is int and 0<=n<=full and len(source['events'])==full,'Prefix/event axes')
    bounds=policy['securities'];require(type(bounds) is list,'Evidence list')
    keys=[(b['symbol'],b['issuer_key']) for b in bounds]
    require(len(set(keys))==len(keys),'Duplicate security-history bounds')
    for b in bounds:
        require(set(b)=={'symbol','issuer_key','not_before','available_from','kind','evidence_id'},'History bound schema')
        iso(b['not_before']);iso(b['available_from'])
        require(b['not_before']<=b['available_from'] and b['kind'] in ('exchange_new_listing_lower_bound','issuer_actual_adr_start'), 'Invalid available-time history bound')
        require(all(type(b[k]) is str and b[k] for k in ('symbol','issuer_key','evidence_id')),'Bound identity')
    records=[];events=[]
    for i in range(n):
        ev=deepcopy(source['events'][i]);day=iso(meta['dates'][i]);names=meta['symbols'][i]
        require(ev['query_index']==i and ev['date']==day,'Event identity')
        stock=[j for j,s in enumerate(names) if s not in meta['etfs']]
        rr=[deepcopy(r) for r in source['records'] if r['query_index']==i]
        require([r['column_index'] for r in rr]==stock==ev['stock_indices'],'Full stock inventory')
        counts=Counter(r['issuer_key'] for r in rr if r['issuer_key'])
        for r in rr:
            j=r['column_index'];require(r['symbol']==names[j] and r['date']==day,'Record identity')
            require(r['source_dates']==ev['source_dates'] and r['initial_partial']==ev['initial_partial'],'Record clock')
            original=list(r['gap_reasons']);gap=list(original);deferred=[];chosen=None
            r['original_gap_reasons']=original
            if not r['issuer_key'] and 'missing_issuer' not in gap:gap.append('missing_issuer')
            if r['issuer_key'] and counts[r['issuer_key']]>1 and 'duplicate_issuer' not in gap:gap.append('duplicate_issuer')
            visible=[b for b in bounds if b['symbol']==r['symbol'] and b['issuer_key']==r['issuer_key'] and b['available_from']<=day]
            require(len(visible)<=1,'Conflicting historical bound')
            if r['initial_partial']:
                status='initial_partial'
                require(r['endpoints']==[] and r['source_dates']==[],'Partial has no endpoints')
            else:
                require(len(r['endpoints'])==len(r['source_dates'])==2,'Two endpoint source records required')
                require([x['date'] for x in r['endpoints']]==r['source_dates'],'Endpoint date identity')
                for p in r['endpoints']:
                    require(p['symbol']==r['symbol'],'Endpoint security identity')
                chosen=visible[0] if visible else None
                immature=chosen is not None and r['source_dates'][0]<chosen['not_before']
                status='official_pre_inception' if immature else 'observed_full_window'
                if immature:
                    for p in r['endpoints']:
                        if p['date']<chosen['not_before']:
                            reason='endpoint:'+p['role']+':'+str(p.get('reason'))
                            if p['status']=='missing' and p.get('reason') in ('missing_value','missing_column','missing_row') and reason in gap:
                                gap.remove(reason);deferred.append(reason)
                            else:gap.append('pre_inception_price_or_unclassified_missing:'+p['role'])
                        elif p['status']!='valid':gap.append('post_bound_endpoint_missing:'+p['role'])
                    if 'missing_sector' in gap:
                        gap.remove('missing_sector');deferred.append('missing_sector')
                else:
                    for p in r['endpoints']:
                        if p['status']!='valid':gap.append('mature_endpoint_not_valid:'+p['role'])
                    if r['sector'] is None:gap.append('missing_sector')
            r.update(history_status=status,history_bound=deepcopy(chosen),deferred_gap_reasons=sorted(set(deferred)),gap_reasons=sorted(set(gap)))
        for r in rr:
            mature=r['history_status']=='observed_full_window'
            peers=[x for x in rr if mature and x['history_status']=='observed_full_window' and r['sector'] is not None
                and x['sector']==r['sector'] and r['issuer_key'] and x['issuer_key'] and x['issuer_key']!=r['issuer_key']]
            peers.sort(key=lambda x:(x['issuer_key'],x['symbol'],x['column_index']))
            ids=[p['column_index'] for p in peers];count=len({p['issuer_key'] for p in peers})
            r.update(peer_indices=ids,peer_issuer_count=count,structural_peer_support=bool(mature and count>=2))
        ev.update(mature_indices=[r['column_index'] for r in rr if r['history_status']=='observed_full_window'],
            supported_indices=[r['column_index'] for r in rr if r['structural_peer_support']])
        records.extend(rr);events.append(ev)
    gaps=[r for r in records if r['gap_reasons']]
    counts=dict(events=n,stock_cells=len(records),mature_cells=sum(r['history_status']=='observed_full_window' for r in records),
        warmup_cells=sum(r['history_status']=='official_pre_inception' for r in records),partial_cells=sum(r['initial_partial'] for r in records),
        supported_cells=sum(r['structural_peer_support'] for r in records),deferred_sector_cells=sum('missing_sector' in r['deferred_gap_reasons'] for r in records),
        deferred_endpoints=sum(sum(x.startswith('endpoint:') for x in r['deferred_gap_reasons']) for r in records),gap_cells=len(gaps))
    return dict(schema='peer239_mature_source_report_v1',status='ADMITTED' if not gaps else 'SOURCE_GAPS',admitted=not gaps,
        events=events,records=records,gaps=gaps,counts=counts,source_only=True,original_inventory_preserved=True,
        strict_pit=False,momentum_computed=False,rankings_computed=False,orders_computed=False,returns_computed=False,
        model_fits=0,original_failed_audit_preserved=True)
