"""Independent metadata-only reconstruction of evidence-bounded warmup.

Consumes an already admitted original audit envelope, not prices or eligibility.
Original source certification and official evidence hashes are host duties.
"""
from collections import Counter
from copy import deepcopy
from datetime import date

COUNT_FIELDS=('events','stock_cells','mature_cells','warmup_cells','partial_cells',
              'supported_cells','deferred_sector_cells','deferred_endpoints','gap_cells')
FLAGS=('passed','independent_metadata_rebuild','full_stock_inventory_preserved',
       'exact_security_and_issuer_bound_matching','public_availability_clock_checked',
       'strict_pre_bound_endpoint_tested','bound_date_equality_not_warmup',
       'original_gap_reasons_preserved','only_pre_bound_missing_deferred',
       'pre_bound_valid_price_contamination_blocks','post_bound_missing_blocks',
       'invalid_endpoints_never_deferred','identity_and_duplicates_never_deferred',
       'sector_deferred_only_for_evidenced_warmup','partial_remains_no_feature',
       'mature_peer_reference_rebuilt_before_eligibility','whole_issuer_peer_exclusion',
       'at_least_two_other_issuers','exact_report_counts_and_types_checked',
       'no_strategy_math_computed')


def require(ok,message):
    if not ok:raise ValueError(message)


def iso(raw):
    require(type(raw) is str,'Canonical ISO date')
    try:d=date.fromisoformat(raw)
    except (ValueError,TypeError) as exc:raise ValueError('Canonical ISO date') from exc
    require(d.isoformat()==raw,'Canonical ISO date')
    return d


def exact(actual,expected,path='maturity report'):
    require(type(actual) is type(expected),path+': type')
    if isinstance(expected,dict):
        require(set(actual)==set(expected),path+': keys')
        for key in expected:exact(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected,list):
        require(len(actual)==len(expected),path+': length')
        for j,(a,b) in enumerate(zip(actual,expected)):exact(a,b,path+'/'+str(j))
    else:require(actual==expected,path+': value')


def policy_index(policy):
    require(isinstance(policy,dict) and policy.get('schema')=='peer239_history_bound_policy_v1','History policy required')
    records=policy.get('securities');require(type(records) is list,'Evidence list required')
    answer={}
    for item in records:
        require(type(item) is dict and set(item)=={'symbol','issuer_key','not_before','available_from','kind','evidence_id'},'Bound schema')
        for key in ('symbol','issuer_key','evidence_id'):require(type(item[key]) is str and bool(item[key]),'Bound identity')
        start,known=iso(item['not_before']),iso(item['available_from'])
        require(start<=known and item['kind'] in ('exchange_new_listing_lower_bound','issuer_actual_adr_start'),'History evidence class/clock')
        key=(item['symbol'],item['issuer_key']);require(key not in answer,'Duplicate security bound')
        answer[key]=(start,known,deepcopy(item))
    return answer


def inspect(source,meta,policy,limit=None):
    require(isinstance(source,dict) and source.get('schema')=='peer239_source_report_v1' and source.get('source_only') is True,'Original source-only report')
    for key in ('momentum_computed','rankings_computed','orders_computed','returns_computed'):
        require(source.get(key) is False,'Original source math forbidden')
    require(type(source.get('model_fits')) is int and source['model_fits']==0,'Original fits forbidden')
    require(isinstance(meta,dict) and isinstance(meta.get('dates'),list) and isinstance(meta.get('symbols'),list),'Original metadata axes')
    total=len(meta['dates']);n=total if limit is None else limit
    require(type(n) is int and 0<=n<=total and len(source['events'])==total,'Prefix/event axes')
    evidence=policy_index(policy);events=[];records=[]
    for q in range(n):
        day=iso(meta['dates'][q]);names=meta['symbols'][q];event=deepcopy(source['events'][q])
        require(event['query_index']==q and event['date']==meta['dates'][q],'Event identity')
        indexes=[j for j,s in enumerate(names) if s not in meta['etfs']]
        rows=[deepcopy(r) for r in source['records'] if r['query_index']==q]
        require([r['column_index'] for r in rows]==indexes==event['stock_indices'],'Full stock inventory')
        identities=Counter(r['issuer_key'] for r in rows if r['issuer_key'])
        for row in rows:
            j=row['column_index'];require(row['symbol']==names[j] and row['date']==meta['dates'][q],'Cell identity')
            require(row['source_dates']==event['source_dates'] and row['initial_partial']==event['initial_partial'],'Cell clock')
            original=list(row['gap_reasons']);remaining=set(original);deferred=set();bound=None
            identity=row['issuer_key']
            if not identity:remaining.add('missing_issuer')
            elif identities[identity]>1:remaining.add('duplicate_issuer')
            candidate=evidence.get((row['symbol'],identity))
            if row['initial_partial']:
                require(row['endpoints']==[] and row['source_dates']==[],'Partial endpoints forbidden')
                status='initial_partial'
            else:
                points=row['endpoints'];dates=row['source_dates']
                require(len(points)==len(dates)==2,'Two endpoint records')
                require([p['date'] for p in points]==dates,'Endpoint dates')
                require(all(p['symbol']==row['symbol'] for p in points),'Endpoint security')
                if candidate is not None and candidate[1]<=day:bound=candidate[2]
                warm=bound is not None and dates[0]<bound['not_before']
                status='official_pre_inception' if warm else 'observed_full_window'
                if warm:
                    for point in points:
                        if point['date']<bound['not_before']:
                            key='endpoint:'+point['role']+':'+str(point.get('reason'))
                            allowed=point['status']=='missing' and point.get('reason') in ('missing_value','missing_column','missing_row')
                            if allowed and key in remaining:
                                remaining.remove(key);deferred.add(key)
                            else:remaining.add('pre_inception_price_or_unclassified_missing:'+point['role'])
                        elif point['status']!='valid':remaining.add('post_bound_endpoint_missing:'+point['role'])
                    if 'missing_sector' in remaining:
                        remaining.remove('missing_sector');deferred.add('missing_sector')
                else:
                    for point in points:
                        if point['status']!='valid':remaining.add('mature_endpoint_not_valid:'+point['role'])
                    if row['sector'] is None:remaining.add('missing_sector')
            row.update(original_gap_reasons=original,history_status=status,history_bound=deepcopy(bound),
                       deferred_gap_reasons=sorted(deferred),gap_reasons=sorted(remaining))
        mature=[r for r in rows if r['history_status']=='observed_full_window']
        for row in rows:
            peers=[]
            if row['history_status']=='observed_full_window' and row['sector'] is not None and row['issuer_key']:
                peers=[other for other in mature if other['sector']==row['sector'] and other['issuer_key'] and other['issuer_key']!=row['issuer_key']]
                peers.sort(key=lambda r:(r['issuer_key'],r['symbol'],r['column_index']))
            count=len({r['issuer_key'] for r in peers})
            row.update(peer_indices=[r['column_index'] for r in peers],peer_issuer_count=count,
                       structural_peer_support=bool(row['history_status']=='observed_full_window' and count>=2))
        event['mature_indices']=[r['column_index'] for r in mature]
        event['supported_indices']=[r['column_index'] for r in rows if r['structural_peer_support']]
        records.extend(rows);events.append(event)
    gaps=[r for r in records if r['gap_reasons']]
    counts={key:0 for key in COUNT_FIELDS};counts.update(events=n,stock_cells=len(records),gap_cells=len(gaps))
    for row in records:
        counts['mature_cells']+=int(row['history_status']=='observed_full_window')
        counts['warmup_cells']+=int(row['history_status']=='official_pre_inception')
        counts['partial_cells']+=int(row['initial_partial'])
        counts['supported_cells']+=int(row['structural_peer_support'])
        counts['deferred_sector_cells']+=int('missing_sector' in row['deferred_gap_reasons'])
        counts['deferred_endpoints']+=sum(reason.startswith('endpoint:') for reason in row['deferred_gap_reasons'])
    return dict(schema='peer239_mature_source_report_v1',status='SOURCE_GAPS' if gaps else 'ADMITTED',admitted=not gaps,
        events=events,records=records,gaps=gaps,counts=counts,source_only=True,original_inventory_preserved=True,
        strict_pit=False,momentum_computed=False,rankings_computed=False,orders_computed=False,returns_computed=False,
        model_fits=0,original_failed_audit_preserved=True)


def check_sources(source,meta,policy,result,limit=None):
    expected=inspect(source,meta,policy,limit)
    exact(result,expected)
    return dict(**{key:True for key in FLAGS},counts=expected['counts'].copy(),status=expected['status'],
        admitted=expected['admitted'],checked_records=len(expected['records']),checked_events=len(expected['events']),
        producer_modules_imported=False,actual_price_values_loaded=False,actual_source_audit_rerun=False,
        original_source_certification_checked_by_this_api=False,official_evidence_hashes_checked_by_this_api=False,
        source_certification_and_evidence_hashes_are_host_responsibility=True,
        momentum_computed=False,rankings_computed=False,orders_computed=False,returns_computed=False,
        model_fits=0,numerical_tolerance=0.,strict_pit=False)


verify_sources=check_sources
