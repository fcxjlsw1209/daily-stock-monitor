"""Exact daily-peer versus endpoint-peer reference; source admission is separate.

No loaders, fitting, eligibility changes, episode, orders or portfolio returns.
Frozen239 helpers only validate metadata. Visible actual-listing evidence refines
path boundaries without changing old history status, eligibility or cohorts.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from numbers import Real
import hashlib
import json
import math
import numpy as np
import pandas as pd
import monthly_peer239_kernel as prior

ROOT = Path(__file__).resolve().parents[1]
LISTING_POLICY_PATH = ROOT/'work/monthly_peer_path252_v2_LISTING_POLICY.json'
LISTING_POLICY_SHA256 = '919a698dafe5fafc7b0516ac7fb613d1ff5292e0a831f049dc134b8864ca1e91'
FROZEN = {
    LISTING_POLICY_PATH: LISTING_POLICY_SHA256,
    ROOT/'work/monthly_peer239_kernel.py': '9d0b9cc7d39243a6d9ca3117a71b97c93c2e1be4adb67757ff5cfaa42f0da419',
    ROOT/'work/monthly_relative_gate235_kernel.py': 'bab4a2b9794ef00a63d43422a51cef167daffbb11b31dc0452cfcd2e74a6af8d'}
for _path, _digest in FROZEN.items():
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _digest:
        raise ValueError('Frozen252 metadata dependency changed: '+str(_path))
LISTING_POLICY = json.loads(LISTING_POLICY_PATH.read_text())
if LISTING_POLICY['schema'] != 'monthly_peer_path252_v2_listing_policy_v1':
    raise ValueError('Frozen listing policy schema')
VALUE_KEYS = ('mature','supported','path_target','endpoint_target')
COUNT_FIELDS = ('events','stock_cells','mature_cells','supported_cells','price_observations','new_model_fits')
SOURCE_SCHEMA = 'monthly_peer_path252_daily_source_report_v2'
SOURCE_KEYS = ('schema','status','admitted','events','records','gaps','counts','source_only',
               'original_inventory_preserved','strict_pit','momentum_computed','rankings_computed',
               'orders_computed','returns_computed','model_fits','original_failed_audit_preserved','daily_paths_checked')
SOURCE_EVENT_KEYS = prior.SOURCE_EVENT_KEYS | {'path_dates'}
SOURCE_RECORD_KEYS = prior.SOURCE_RECORD_KEYS | {'path_observations','path_history_bound'}
OBSERVATION_KEYS = ('date','status','reason')
EVENT_KEYS = ('query_index','date','source_dates','source_positions','stock_indices','mature_indices','supported_indices','cells')
CELL_KEYS = ('column_index','symbol','issuer_key','sector','history_status','peer_indices','peer_issuer_count',
             'focal_gross','daily_peer_gross','endpoint_peer_gross','path_raw','endpoint_raw','path_target','endpoint_target')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def dependency_files():
    return list(dict.fromkeys([*FROZEN, *(ROOT/p for p in LISTING_POLICY['input_sha256'])]))


def _exact(actual, expected, label):
    need(type(actual) is type(expected), 'Noncanonical type: '+label)
    if isinstance(expected, dict):
        need(set(actual) == set(expected), 'Schema differs: '+label)
        for key in expected:
            _exact(actual[key], expected[key], label+'.'+key)
    elif isinstance(expected, list):
        need(len(actual) == len(expected), 'Length differs: '+label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            _exact(a, b, label+'['+str(i)+']')
    elif isinstance(expected, float):
        need(actual.hex() == expected.hex(), 'Float bytes differ: '+label)
    else:
        need(actual == expected, 'Evidence differs: '+label)


def _slot(frame, symbol, day):
    """Classify an actual source slot, without ratio/feature calculations."""
    if symbol not in frame.columns:
        return None, 'missing', 'missing_column'
    if day not in frame.index:
        return None, 'missing', 'missing_row'
    value = frame.at[day, symbol]
    if value is None or value is pd.NA:
        return None, 'missing', 'missing_value'
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('invalid_boolean_adjusted_price:'+symbol+'/'+str(day.date()))
    if not isinstance(value, Real):
        raise ValueError('invalid_nonreal_adjusted_price:'+symbol+'/'+str(day.date()))
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('invalid_numeric_conversion:'+symbol+'/'+str(day.date())) from exc
    if math.isnan(number):
        return None, 'missing', 'missing_value'
    if not math.isfinite(number):
        raise ValueError('nonfinite_adjusted_price:'+symbol+'/'+str(day.date()))
    if number <= 0.:
        raise ValueError('nonpositive_adjusted_price:'+symbol+'/'+str(day.date()))
    return number, 'valid', None


def _metadata(panel, meta, calendar, arrays, source239, limit):
    frame, sessions, queries, etfs, width = prior._input(panel, meta, calendar, arrays, limit)
    rows = prior._source_top(source239, len(queries))
    result = []
    for i, (day, names, t, partial) in enumerate(queries):
        d = str(day.date())
        stock = [j for j, s in enumerate(names) if s not in etfs]
        positions = [] if partial else [t-252, t-42]
        dates = [str(sessions[k].date()) for k in positions]
        ev = source239['events'][i]
        need(isinstance(ev, dict) and set(ev) == prior.SOURCE_EVENT_KEYS,
             'Exact239 source event schema')
        need(type(ev['query_index']) is int and ev['query_index'] == i and ev['date'] == d
             and type(ev['initial_partial']) is bool and ev['initial_partial'] == partial,
             'Exact239 event identity')
        for key in ('stock_indices','mature_indices','supported_indices','source_positions'):
            prior._indices(ev[key], key)
        need(ev['stock_indices'] == stock and ev['source_positions'] == positions
             and ev['source_dates'] == dates, 'Original239 inventory/official clock differs')
        rr = rows[i]
        need([r.get('column_index') for r in rr] == stock, 'Complete ordered direct-stock inventory required')
        status = {j: prior._record(r, i, j, d, names, partial, positions, dates)
                  for j, r in zip(stock, rr)}
        issuers = [r['issuer_key'] for r in rr]
        need(len(issuers) == len(set(issuers)), 'Duplicate whole issuer in inventory')
        mapping = {r['column_index']: r for r in rr}
        mature = [j for j in stock if status[j] == 'observed_full_window']
        supported = []
        for j in stock:
            r = mapping[j]
            peers = [k for k in mature if j in mature and mapping[k]['sector'] == r['sector']
                     and mapping[k]['issuer_key'] != r['issuer_key']]
            peers.sort(key=lambda k: (mapping[k]['issuer_key'], names[k], k))
            count = len({mapping[k]['issuer_key'] for k in peers})
            good = j in mature and count >= 2
            need(r['peer_indices'] == peers and type(r['peer_issuer_count']) is int
                 and r['peer_issuer_count'] == count and type(r['structural_peer_support']) is bool
                 and r['structural_peer_support'] == good, 'Exact fixed mature cohort required')
            if good:
                supported.append(j)
        need(ev['mature_indices'] == mature and ev['supported_indices'] == supported,
             'Rebuilt source maturity/support differs')
        result.append((day, names, t, partial, ev, rr, mature, supported))
    return frame, sessions, result, width


def _path_history_bound(record):
    """A visible first-trade proof changes path checks, never239 maturity."""
    override = LISTING_POLICY['path_bound_override']
    if (record['symbol'] == override['symbol']
            and record['issuer_key'] == override['issuer_key']
            and record['date'] >= override['available_from']):
        _exact(record['history_bound'], LISTING_POLICY['original_history_bound'],
               'preserved_original_history_bound')
        return deepcopy(override)
    return deepcopy(record['history_bound'])


def _construct(panel, meta, calendar, arrays, source239, limit):
    frame, sessions, rows, width = _metadata(panel, meta, calendar, arrays, source239, limit)
    events, records, prices = [], [], {}
    stock_count = mature_count = supported_count = price_count = 0
    for i, (day, names, t, partial, ev, rr, mature, supported) in enumerate(rows):
        path = [] if partial else list(sessions[t-252:t-42+1])
        need(partial or len(path) == 211, 'Exactly211 official path observations required')
        event = deepcopy(ev)
        event['path_dates'] = [str(d.date()) for d in path]
        events.append(event)
        prices[i] = {}
        for r in rr:
            j = r['column_index']
            record = deepcopy(r)
            record['path_history_bound'] = _path_history_bound(r)
            path_bound = record['path_history_bound']
            observations, numbers = [], []
            for d in path:
                value, status, reason = _slot(frame, r['symbol'], d)
                price_count += 1
                before = (path_bound is not None
                          and str(d.date()) < path_bound['not_before'])
                if before:
                    need(status == 'missing', 'positive_prebound_identity_conflict:'+r['symbol']+'/'+str(d.date()))
                else:
                    need(status == 'valid', ('post_bound_missing:' if r['history_status'] == 'official_pre_inception'
                         else 'mature_path_missing:')+r['symbol']+'/'+str(d.date())+':'+str(reason))
                observations.append(dict(date=str(d.date()), status=status, reason=reason))
                numbers.append(value)
            record['path_observations'] = observations
            records.append(record)
            if j in mature:
                need(all(o['status'] == 'valid' for o in observations),
                     'mature_path_bound_conflict:'+r['symbol']+'/'+str(day.date()))
                prices[i][j] = tuple(numbers)
        stock_count += len(rr)
        mature_count += len(mature)
        supported_count += len(supported)
    counts = dict(events=len(rows), stock_cells=stock_count, mature_cells=mature_count,
                  supported_cells=supported_count, price_observations=price_count, new_model_fits=0)
    report = dict(schema=SOURCE_SCHEMA, status='ADMITTED', admitted=True, events=events,
                  records=records, gaps=[], counts=counts, source_only=True,
                  original_inventory_preserved=True, strict_pit=False, momentum_computed=False,
                  rankings_computed=False, orders_computed=False, returns_computed=False,
                  model_fits=0, original_failed_audit_preserved=True, daily_paths_checked=True)
    need(tuple(report) == SOURCE_KEYS and tuple(counts) == COUNT_FIELDS, 'Internal source schema')
    return report, prices, width


def build_source(panel, meta, calendar, arrays, source239, limit=None):
    """Rebuild full211 path admission. Any hard gap aborts; no peer is dropped."""
    return _construct(panel, meta, calendar, arrays, source239, limit)[0]


def _legacy_view(report, n):
    need(isinstance(report, dict) and set(report) == set(SOURCE_KEYS), 'Exact daily source report schema')
    need(report['schema'] == SOURCE_SCHEMA and report['daily_paths_checked'] is True, 'Daily path source required')
    need(isinstance(report['events'], list) and len(report['events']) >= n, 'Daily source event scope')
    need(isinstance(report['records'], list), 'Daily source records required')
    counts = report['counts']
    need(isinstance(counts, dict) and set(counts) == set(COUNT_FIELDS)
         and all(type(counts[k]) is int and counts[k] >= 0 for k in COUNT_FIELDS)
         and counts['events'] == len(report['events']) and counts['new_model_fits'] == 0, 'Daily source counts')
    old = {k: deepcopy(v) for k, v in report.items() if k not in ('daily_paths_checked','events','records')}
    old['schema'] = 'peer239_mature_source_report_v1'
    old['events'], old['records'] = [], []
    for ev in report['events'][:n]:
        need(isinstance(ev, dict) and set(ev) == SOURCE_EVENT_KEYS, 'Exact daily source event keys')
        old['events'].append({k: deepcopy(v) for k, v in ev.items() if k != 'path_dates'})
    for r in report['records']:
        need(isinstance(r, dict) and type(r.get('query_index')) is int and r['query_index'] >= 0,
             'Canonical daily record query index')
        if r['query_index'] >= n:
            continue
        need(set(r) == SOURCE_RECORD_KEYS, 'Exact daily source record keys')
        old['records'].append({k: deepcopy(v) for k, v in r.items() if k not in ('path_observations','path_history_bound')})
    return old


def _pair(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    need(n > 0, 'Nonempty exact median')
    return ordered[n//2] if n % 2 else (ordered[n//2-1]+ordered[n//2])/2


def _rank(value, domain):
    n = len(domain)
    need(n > 0, 'Nonempty rank domain')
    less = sum(x < value for x in domain)
    equal = sum(x == value for x in domain)
    need(equal > 0, 'Rank domain contains focal value')
    return float(Fraction(2*less+equal+1-n, 2*n))


def build_features(panel, meta, calendar, arrays, daily_source_report, limit=None):
    """All raw references stay exact rationals; only bounded final ranks float."""
    e = np.asarray(arrays['eligible'])
    need(e.ndim == 2, 'Eligibility axes required')
    total = len(e)
    need(limit is None or type(limit) is int and 0 <= limit <= total, 'Invalid event prefix')
    n = total if limit is None else limit
    legacy = _legacy_view(daily_source_report, n)
    checked, prices, width = _construct(panel, meta, calendar, arrays, legacy, n)
    supplied = {k: deepcopy(v) for k, v in daily_source_report.items() if k not in ('events','records','counts')}
    supplied['events'] = daily_source_report['events'][:n]
    supplied['records'] = [r for r in daily_source_report['records'] if r['query_index'] < n]
    # Counts for a logical prefix are rebuilt from consumed evidence, never from future rows.
    supplied['counts'] = checked['counts'] if n < len(daily_source_report['events']) else daily_source_report['counts']
    _exact(supplied, checked, 'daily_source_report')
    values = {k: np.zeros((n,width), dtype=bool) if k in VALUE_KEYS[:2]
              else np.full((n,width), np.nan, dtype=np.float64) for k in VALUE_KEYS}
    by_query = {i: [] for i in range(n)}
    for r in checked['records']:
        by_query[r['query_index']].append(r)
    events = []
    for i, ev in enumerate(checked['events']):
        mature, supported = ev['mature_indices'], ev['supported_indices']
        mapping = {r['column_index']: r for r in by_query[i]}
        values['mature'][i,mature] = True
        values['supported'][i,supported] = True
        exact = {j: tuple(Fraction.from_float(v) for v in prices[i][j]) for j in mature}
        gross = {j: p[-1]/p[0] for j,p in exact.items()}
        daily = {j: tuple(p[d+1]/p[d] for d in range(210)) for j,p in exact.items()}
        raw, endpoint, refs = {}, {}, {}
        for j in supported:
            peers = mapping[j]['peer_indices']
            g_daily = Fraction(1)
            for d in range(210):
                g_daily *= _median([daily[k][d] for k in peers])
            g_endpoint = _median([gross[k] for k in peers])
            need(g_daily > 0 and g_endpoint > 0, 'Positive exact peer references')
            refs[j] = (g_daily, g_endpoint)
            raw[j] = gross[j]/g_daily-1
            endpoint[j] = gross[j]/g_endpoint-1
        domain_path, domain_endpoint = [raw[j] for j in supported], [endpoint[j] for j in supported]
        for j in supported:
            values['path_target'][i,j] = _rank(raw[j], domain_path)
            values['endpoint_target'][i,j] = _rank(endpoint[j], domain_endpoint)
        cells = []
        for j in ev['stock_indices']:
            r = mapping[j]
            cell = dict(column_index=j,symbol=r['symbol'],issuer_key=r['issuer_key'],sector=r['sector'],
                        history_status=r['history_status'],peer_indices=list(r['peer_indices']),peer_issuer_count=r['peer_issuer_count'],
                        focal_gross=_pair(gross[j]) if j in mature else None,
                        daily_peer_gross=_pair(refs[j][0]) if j in supported else None,
                        endpoint_peer_gross=_pair(refs[j][1]) if j in supported else None,
                        path_raw=_pair(raw[j]) if j in supported else None,
                        endpoint_raw=_pair(endpoint[j]) if j in supported else None,
                        path_target=float(values['path_target'][i,j]) if j in supported else None,
                        endpoint_target=float(values['endpoint_target'][i,j]) if j in supported else None)
            need(tuple(cell) == CELL_KEYS, 'Internal cell schema')
            cells.append(cell)
        event = {k: deepcopy(ev[k]) for k in EVENT_KEYS if k != 'cells'}
        event['cells'] = cells
        need(tuple(event) == EVENT_KEYS, 'Internal event schema')
        events.append(event)
    return values, dict(schema='monthly_peer_path252_features_v2',counts=checked['counts'],events=events)
