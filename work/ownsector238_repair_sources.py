"""Source repair audit only. No new strategy features or performance calculations."""
from copy import deepcopy
from datetime import date
from numbers import Real
import math
import numpy as np
import pandas as pd
import ownsector238_source_audit as prior

RULES = {
    'AZN': ('XLV', '2023-03-11', None),
    'BUD': ('XLP', '2016-03-15', None),
    'CHL': ('VOX', '2016-04-28', None),
    'GOOGL': ('XLK', '2016-02-12', '2018-09-24'),
    'PDD': ('XLY', '2020-04-25', None),
    'TSLA': ('XLY', '2020-02-14', None),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_policy(policy):
    require(policy.get('schema') == 'ownsector238_repair_source_policy_v1', 'Policy schema')
    rows = policy.get('missing_mapping_rules')
    require(isinstance(rows, list) and len(rows) == 6, 'Six mapping rules required')
    seen = {}
    for row in rows:
        symbol = row['symbol']
        require(symbol not in seen, 'Duplicate repair rule')
        for key in ('public_date', 'valid_from'):
            require(date.fromisoformat(row[key]).isoformat() == row[key], 'Canonical date')
        require(row['public_date'] < row['valid_from'], 'No publication-day inference')
        seen[symbol] = (row['benchmark'], row['valid_from'], row['valid_until_exclusive'])
        require(row.get('source_file') and row.get('source_ids'), 'Mapping evidence required')
    require(seen == RULES, 'Prospective rules changed')
    comm = policy.get('communication_benchmark', {})
    require(comm.get('old') == 'XLC' and comm.get('new') == 'VOX' and comm.get('price_field') == 'adjClose'
            and comm.get('fill_or_splice') is False, 'Communication policy changed')


def resolve(record, policy):
    check_policy(policy)
    if record['is_etf']:
        require(record['benchmark'] == 'SPY', 'Original ETF SPY dispatch')
        return 'SPY', 'unchanged', None
    old = record['benchmark']
    if old == 'XLC':
        return 'VOX', 'full_period_communication_proxy', None
    if old is not None:
        return old, 'unchanged', None
    rule = next((r for r in policy['missing_mapping_rules'] if r['symbol'] == record['symbol']), None)
    if rule and record['date'] >= rule['valid_from'] and (rule['valid_until_exclusive'] is None or record['date'] < rule['valid_until_exclusive']):
        return rule['benchmark'], 'dated_business_proxy', rule['source_ids']
    return None, 'unresolved', None


def parse_vox(rows, calendar):
    """Validate all acquired adjusted-close rows, without ratios or filling."""
    require(isinstance(rows, list) and len(rows) > 0, 'Nonempty VOX daily array required')
    sessions = set(calendar)
    days, values = [], []
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('date'), str), 'VOX row/date schema')
        stamp = pd.Timestamp(row['date'])
        require(stamp.tz is not None and stamp.utcoffset().total_seconds() == 0
                and stamp == stamp.normalize(), 'VOX daily UTC midnight required')
        day = str(stamp.date())
        require('2015-01-01' <= day <= '2026-09-24' and day in sessions, 'VOX out-of-window/non-session date')
        require(not days or day > days[-1], 'VOX dates must be strictly increasing and unique')
        value = row.get('adjClose')
        require(not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
                and math.isfinite(float(value)) and float(value) > 0, 'VOX finite positive adjusted close required')
        days.append(day); values.append(float(value))
    return pd.DataFrame({'VOX': values}, index=pd.DatetimeIndex(days))


def repair_sources(original, adj, vox, policy, limit=None):
    check_policy(policy)
    # Validate date/security axes with the already-tested frozen source schema.
    prior._frames({'adj': adj, 'sectors': pd.DataFrame(index=adj.index)})
    prior._frames({'adj': vox, 'sectors': pd.DataFrame(index=vox.index)})
    require(list(vox.columns) == ['VOX'], 'Dedicated VOX price frame required')
    total = len(original['events'])
    require(limit is None or type(limit) is int and 0 <= limit <= total, 'Invalid prefix')
    n = total if limit is None else limit
    events, records, gaps = [], [], []
    for i, old_event in enumerate(original['events'][:n]):
        require(old_event['query_index'] == i, 'Original event order')
        event = deepcopy(old_event)
        event['admitted_count'] = event['gap_count'] = 0
        event['market_endpoints'] = []
        required = [r for r in original['records'] if r['query_index'] == i]
        require([r['column_index'] for r in required] == event['required_indices'], 'Original required-cell axes')
        if event['active']:
            require(len(event['source_dates']) == 2, 'Required source dates')
            for role, raw in zip(('market_start', 'market_end'), event['source_dates']):
                event['market_endpoints'].append(prior.endpoint(adj, role, 'SPY', pd.Timestamp(raw)))
            reasons = list(dict.fromkeys(e['reason'] for e in event['market_endpoints'] if e['reason']))
            if reasons:
                gaps.append(dict(scope='market', query_index=i, column_index=None, date=event['date'], symbol='SPY', benchmark='SPY', reasons=reasons))
        else:
            require(event['date'] == '2016-08-17' and i == 0 and not required, 'Named partial bypass')
        for old in required:
            require(old['date'] == event['date'], 'Original query date')
            new, repair, evidence = resolve(old, policy)
            out = deepcopy(old)
            out.update(original_benchmark=old['benchmark'], benchmark=new, mapping_repair=repair,
                       mapping_evidence=evidence, mapping_status='available' if new else 'missing')
            out['endpoints'] = []
            for role, symbol, raw in (
                ('security_start', old['symbol'], event['source_dates'][0]),
                ('security_end', old['symbol'], event['source_dates'][1]),
                ('benchmark_start', new, event['source_dates'][0]),
                ('benchmark_end', new, event['source_dates'][1])):
                frame = vox if symbol == 'VOX' and role.startswith('benchmark') else adj
                out['endpoints'].append(prior.endpoint(frame, role, symbol, pd.Timestamp(raw)))
            reasons = [] if new else ['missing_benchmark_mapping']
            for endpoint in out['endpoints']:
                if endpoint['reason'] and endpoint['reason'] not in reasons:
                    reasons.append(endpoint['reason'])
            out.update(admitted=not reasons, gap_reasons=reasons)
            # Old source provenance remains separately visible, not presented as the repaired origin.
            out['original_mapping_source'] = old['mapping_source']
            out['mapping_source'] = old['mapping_source'] if repair == 'unchanged' else repair
            records.append(out)
            if reasons:
                event['gap_count'] += 1
                gaps.append(dict(scope='cell', query_index=i, column_index=out['column_index'], date=out['date'], symbol=out['symbol'], benchmark=new, reasons=reasons))
            else:
                event['admitted_count'] += 1
        events.append(event)
    points = [p for e in events for p in e['market_endpoints']] + [p for r in records for p in r['endpoints']]
    counts = dict(events=n, symbols_per_event=original['counts']['symbols_per_event'], checked_cells=n*original['counts']['symbols_per_event'],
                  initial_partial_events=sum(e['initial_partial'] for e in events), regular_events=sum(e['active'] for e in events),
                  original_eligible_cells=sum(len(e['original_eligible_indices']) for e in events), required_cells=len(records),
                  stock_required_cells=sum(not r['is_etf'] for r in records), etf_required_cells=sum(r['is_etf'] for r in records),
                  mapping_available_cells=sum(r['mapping_status']=='available' for r in records), mapping_missing_cells=sum(r['mapping_status']=='missing' for r in records),
                  security_benchmark_endpoint_checks=4*len(records), market_endpoint_checks=2*sum(e['active'] for e in events),
                  valid_endpoints=sum(p['status']=='valid' for p in points), missing_endpoints=sum(p['status']=='missing' for p in points),
                  invalid_endpoints=sum(p['status']=='invalid' for p in points), admitted_cells=sum(r['admitted'] for r in records),
                  gap_cells=sum(not r['admitted'] for r in records), market_gap_events=sum(g['scope']=='market' for g in gaps))
    require(tuple(counts) == prior.COUNT_FIELDS, 'Count schema')
    return dict(schema='ownsector238_repaired_source_report_v1', status='NOT_ADMITTED' if gaps else 'ADMITTED', admitted=not gaps,
                counts=counts, events=events, records=records, gaps=gaps,
                repair_counts={k:sum(r['mapping_repair']==k for r in records) for k in ('unchanged','full_period_communication_proxy','dated_business_proxy','unresolved')},
                source_only=True, momentum_computed=False, eligibility_changed=False, rankings_computed=False, orders_computed=False,
                returns_computed=False, model_fits=0, strict_pit=False)
