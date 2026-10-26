"""Write-once review of closed199; no price, model, or replay imports.

Run only after the root has confirmed199 closure. This reader verifies existing
hash-bound artifacts and summarizes recorded results; it creates no new factor,
selection, order, NAV series, or parameter search.
Prior readers remain unchanged. A next-open-cost fees_paid total may differ by
<=1e-9; all other recorded incumbent metrics remain exact. No residual is required.
Exported fills and daily equity must be exactly identical before this exception.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_nested_trend199'
PRIOR = ROOT / 'outputs/monthly_event_risk_payoff198'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS = ('incumbent_local67', 'nested_min_core', 'nested_mean_control')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
BOUNDS = {'full122': (0, 122), 'first88': (0, 88), 'early53': (0, 53),
          'recent35': (53, 88), 'later34': (88, 122)}
METRICS = ('start', 'end', 'cagr', 'sharpe', 'max_drawdown')
COUNTS = ('valid_price_observations', 'horizon_returns', 'ranked_horizon_observations', 'aggregated_scores')
HORIZONS = (126, 189, 252)
PREPARED_FILES = {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json',
                  'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'}
RESULT_FILES = {'RESULTS.json', 'INDEPENDENT_OUTCOME.json'} | {
    arm+'_'+path+'_'+kind+'.json' for arm in ARMS for path in PATHS
    for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
PRIOR_READERS = {'work/payoff198_review.py': 'b958801dcc1ea84ca6cc84e381681cf61f867c4f1e4f23285d963f0a1b93f975'}
CONTEXT = ROOT/'work/monthly_nested_trend199_context.json'
CONTEXT_SHA = '2a0b157600acf69ea9325fd3e46bc0a41d3337ff852e34677ee0d58d8b407233'
ORIGINAL_DATA = ROOT/'outputs/monthly_local_linear70/DATA.npz'
TEST_RECEIPT_SHA = '48ee357cbe00b4df1f42f97514e0594dc1e621829a1434b32617dd148b5aa0bc'
ENGINE = ROOT/'outputs/monthly_factor_search_v2/validate.py'
ENGINE_SHA = 'b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4'
FEE_TOTAL_ABSOLUTE_TOLERANCE = 1e-9
INDEPENDENT = ROOT/'work/monthly_nested_trend199_independent.py'
INDEPENDENT_SHA = '5e8f5c622130465715d564f9ee101550850b2cdbc15c5cc7cf4ee84a1a93afe7'
NUMERICAL_COMPARISON = 'exact binary64 gains, ranks, aggregates and audit'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    require(path.is_relative_to(ROOT), 'Review source must be inside the local research root')
    return path


class Evidence:
    def __init__(self):
        self.pins = {}

    def verify(self, value, expected=None):
        path = source_path(value)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda: stream.read(1024*1024), b''):
                h.update(data)
        digest = h.hexdigest()
        if expected is not None:
            require(digest == expected, 'Changed evidence: '+str(path))
        relative = str(path.relative_to(ROOT))
        require(relative not in self.pins or self.pins[relative] == digest,
                'Evidence changed during review: '+relative)
        self.pins[relative] = digest
        return digest

    def load(self, value, expected=None):
        path = source_path(value)
        self.verify(path, expected)
        return json.loads(path.read_text())

    def manifest(self, folder, expected, plan_sha):
        record = self.load(folder/'COMPLETION.json', expected)
        require(record['status'] == 'COMPLETE' and record['plan_sha256'] == plan_sha,
                'Incomplete or different phase: '+str(folder))
        expected_names = PREPARED_FILES if folder.name == 'prepared' else RESULT_FILES
        require(set(record['artifacts_sha256']) == expected_names, 'Incomplete or unexpected phase artifact set')
        for name, digest in record['artifacts_sha256'].items():
            require(Path(name).name == name, 'Unexpected artifact path')
            self.verify(folder/name, digest)
        return record


def check_recorded_metrics(actual, expected, label):
    """Compare existing independent statistics, using its registered tolerances."""
    for key, value in actual.items():
        if isinstance(value, str) or value is None or key == 'trading_sessions':
            require(expected[key] == value, label+'/'+key)
        else:
            wanted = expected[key]
            require(math.isfinite(value) and math.isfinite(wanted) and
                    abs(value-wanted) <= 1e-7+2e-11*abs(wanted), label+'/'+key)


def changed_holdings(orders, reference, candidate):
    changes = []
    for i, (before, after) in enumerate(zip(orders[reference], orders[candidate])):
        require(before['date'] == after['date'] and before['end'] == after['end'],
                'Order calendars differ')
        old, new = set(before['symbols']), set(after['symbols'])
        if old != new:
            changes.append(dict(index=i, signal_date=before['date'], holding_end=before['end'],
                reference=before['symbols'], candidate=after['symbols'],
                removed=sorted(old-new), added=sorted(new-old)))
    return dict(counts={key: sum(lo <= x['index'] < hi for x in changes)
                        for key, (lo, hi) in BOUNDS.items()}, months=changes)


def compare_incumbent_record(current, prior, path, location, differences):
    """Exact recursively, apart from the two specifically named fee-total keys."""
    if isinstance(current, dict):
        require(isinstance(prior, dict) and set(current) == set(prior), 'Incumbent record keys changed')
        for key in current:
            compare_incumbent_record(current[key], prior[key], path, location+'/'+key, differences)
    elif isinstance(current, list):
        require(isinstance(prior, list) and len(current) == len(prior), 'Incumbent record list changed')
        for i, (a, b) in enumerate(zip(current, prior)):
            compare_incumbent_record(a, b, path, location+'/'+str(i), differences)
    elif current != prior:
        require(path == 'next_open_cost' and location in ('metrics/fees_paid', 'slices/full122/fees_paid'),
                'Unexpected incumbent difference: '+path+'/'+location)
        require(type(current) in (int, float) and type(prior) in (int, float) and
                math.isfinite(current) and math.isfinite(prior) and
                abs(current-prior) <= FEE_TOTAL_ABSOLUTE_TOLERANCE, 'Fee total difference exceeds strict1e-9 bound')
        differences.append(dict(path=path, field=location, previous198=prior, current199=current,
            signed_difference=current-prior, absolute_difference=abs(current-prior),
            absolute_tolerance=FEE_TOTAL_ABSOLUTE_TOLERANCE))


def diagnose_incumbent_exports(evidence, prior_manifest, current_results, prior_results):
    """Inspect recorded fills/equity and fee summations, never run an engine."""
    for path, digest in PRIOR_READERS.items():
        evidence.verify(path, digest)
    evidence.verify(ENGINE, ENGINE_SHA)
    code = ENGINE.read_text()
    snippets = {
        'unordered_turnover_domain': 'symbols=set(old)|set(chosen)',
        'turnover_sequential_sum': 'def turnover(post):return sum(abs((post/n if s in chosen and n else 0)-old.get(s,0)) for s in symbols)',
        'per_rebalance_fee': 'post=(lo+hi)/2;fee=cost*turnover(post)',
        'sorted_fill_export': 'for s in sorted(symbols):',
        'fee_accumulator': 'totalcost+=fee;trades+=tt',
        'recorded_total': 'fees_paid=totalcost'}
    require(all(text in code for text in snippets.values()), 'Pinned fee implementation changed')
    paths = {}
    for path in PATHS:
        exports = {}
        for kind in ('TRADES', 'DAILY', 'DAILY_POSITIONS', 'HOLDINGS'):
            name = ARMS[0]+'_'+path+'_'+kind+'.json'
            old = evidence.load(PRIOR/'results'/name, prior_manifest['artifacts_sha256'][name])
            new = evidence.load(OUT/'results'/name)
            exports[kind] = (old, new)
            if kind != 'HOLDINGS':
                require(old == new, 'Incumbent exported '+kind+' changed: '+path)
        old_holds, new_holds = exports['HOLDINGS']
        require(len(old_holds) == len(new_holds) == 122, 'Holding periods changed')
        fee_differences = []
        for old, new in zip(old_holds, new_holds):
            require(set(old) == set(new) and {k: v for k, v in old.items() if k != 'fee'} ==
                    {k: v for k, v in new.items() if k != 'fee'}, 'Non-fee holding data changed')
            require(math.isfinite(old['fee']) and math.isfinite(new['fee']), 'Nonfinite exported period fee')
            if old['fee'] != new['fee']:
                require(path == 'next_open_cost' and abs(new['fee']-old['fee']) <= FEE_TOTAL_ABSOLUTE_TOLERANCE,
                        'Unexpected period fee difference')
                fee_differences.append(dict(signal_date=old['signal_date'], execution_date=old['execution_date'],
                    previous198=old['fee'], current199=new['fee'], signed_difference=new['fee']-old['fee']))
        totals = {}
        for label, holds, result in (('previous198', old_holds, prior_results['arms'][ARMS[0]][path]),
                                      ('current199', new_holds, current_results['arms'][ARMS[0]][path])):
            running = 0.
            for row in holds:
                running += row['fee']
            require(running == result['metrics']['fees_paid'], 'Period fees do not exactly reproduce recorded accumulator')
            totals[label] = dict(recorded_total=result['metrics']['fees_paid'], sequential_period_sum=running,
                compensated_period_sum=math.fsum(row['fee'] for row in holds))
        old_trades, new_trades = exports['TRADES']
        paths[path] = dict(individual_fills_exact=True, fill_rows=len(old_trades),
            daily_equity_exact=True, daily_equity_rows=len(exports['DAILY'][0]),
            daily_reconciled_positions_exact=True, holding_nonfee_fields_exact=True,
            period_fee_differences=fee_differences, totals=totals,
            fill_fee_sum=math.fsum(row['cost'] for row in old_trades),
            fill_fee_values_exact=all(a['cost'] == b['cost'] for a, b in zip(old_trades, new_trades)))
    return dict(engine_path=str(ENGINE.relative_to(ROOT)), engine_sha256=ENGINE_SHA,
        source_snippets=snippets, prior_reader_provenance=PRIOR_READERS, paths=paths,
        explanation='The frozen engine forms the turnover domain with set(old)|set(chosen), sums its terms in set iteration order, computes a separate rebalance fee, and accumulates those fees chronologically. Fill exports are sorted by symbol and their costs are computed from exported unit changes. Any recorded fee-only residual is disclosed, while exported fills, daily equity and reconciled positions must remain exactly equal. A permitted residual is consistent with floating summation order; historical process hash seeds and their iteration orders are not claimed to have been recovered. An empty residual list means exact equality, not a failed diagnostic.',
        tolerance_scope='Only next_open_cost metrics.fees_paid and slices.full122.fees_paid use1e-9 absolute tolerance; every other metric/slice key is exact. Per-period fee-only residuals are separately disclosed and bounded1e-9; no source, strategy, selection, or accounting gate is changed.',
        portfolio_replay_performed=False)



def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)


def check_nested_audit(feature, meta, sessions, expected_events=122, width=49):
    """Audit stored endpoints/ranks; no raw price panel or new strategy is evaluated."""
    counts_by_event = {key:feature[key] for key in ('source_available_by_event','available_by_event')}
    require(all(isinstance(v,list) and len(v)==expected_events and all(type(x) is int for x in v)
                for v in counts_by_event.values()), 'Invalid per-event counts')
    require(feature['events']==expected_events and feature['symbols_per_event']==width and
            feature['checked_cells']==expected_events*width and feature['price_count']==253 and
            feature['skip_sessions']==42 and feature['horizons']==list(HORIZONS), 'Nested window contract differs')
    require(feature['common_availability'] is True and feature['ranking_performed'] is True and
            feature['ranking_eligibility_filtered'] is False and feature['trading_eligibility_modified'] is False and
            feature['portfolio_returns_calculated'] is False, 'Feature scope differs')
    require(feature['new_model_fits']==feature['predictive_model_fits']==0, 'Unexpected fit count')
    require(len(feature['event_history'])==len(meta['dates'])==len(meta['symbols'])==expected_events,
            'Event count differs')
    lookup={d:i for i,d in enumerate(sessions)}
    require(len(lookup)==len(sessions) and sessions==sorted(sessions), 'Official session axis differs')
    cells=[]
    for i,event in enumerate(feature['event_history']):
        require(event['date']==meta['dates'][i] and event['symbols']==meta['symbols'][i] and
                len(event['cells'])==width, 'Event identities differ')
        t=lookup[event['date']]
        require(t>=294 and event['session_position']==t and event['price_start']==sessions[t-294] and
                event['price_end']==sessions[t-42] and event['horizons']==list(HORIZONS) and
                event['horizon_start_dates']=={str(h):sessions[t-42-h] for h in HORIZONS},
                'Nested horizon dates differ')
        if not(i==0 and event['date']=='2016-08-17'):
            require(t+1<len(sessions) and sessions[t+1][:7]!=sessions[t][:7], 'Non-month-end query')
        mask=[c['available'] for c in event['cells']]
        n=sum(mask)
        require(event['available']==event['source_available']==mask and
                event['source_available_count']==event['available_count']==n and
                counts_by_event['source_available_by_event'][i]==counts_by_event['available_by_event'][i]==n,
                'Common domain differs')
        valid=[c for c in event['cells'] if c['available']]
        for symbol,cell in zip(event['symbols'],event['cells']):
            require(cell['symbol']==symbol and cell['date']==event['date'] and
                    type(cell['available']) is bool and cell['source_available'] is cell['available'] and
                    type(cell['exact_equal_ranks']) is bool, 'Cell identity/support differs')
            if not cell['available']:
                require(cell['reason'] in ('insufficient_history','missing_symbol','missing_official_session',
                                          'nonnumeric_adjusted_price','invalid_adjusted_price') and
                        all(cell[k] is None for k in ('end_price','start_prices','gain126','gain189','gain252',
                             'rank126','rank189','rank252','candidate','control')) and
                        cell['exact_equal_ranks'] is False, 'Unavailable must remain missing')
            else:
                require(cell['reason']=='available' and not event['missing_official_sessions'] and
                        finite_number(cell['end_price']) and cell['end_price']>0 and
                        set(cell['start_prices'])==set(map(str,HORIZONS)), 'Invalid endpoint evidence')
                rr=[]
                for h in HORIZONS:
                    start=cell['start_prices'][str(h)]
                    require(finite_number(start) and start>0, 'Invalid start price')
                    gain=cell['end_price']/start-1.
                    require(finite_number(gain) and cell['gain'+str(h)]==gain, 'Exact simple gain differs')
                    less=sum(c['gain'+str(h)]<gain for c in valid)
                    equal=sum(c['gain'+str(h)]==gain for c in valid)
                    rank=(less+(equal+1)/2)/n-.5
                    require(cell['rank'+str(h)]==rank, 'Exact original percentile differs')
                    rr.append(rank)
                q=min(rr); equal=rr[0]==rr[1]==rr[2]; c=q if equal else math.fsum(rr)/3.
                require(cell['candidate']==q and cell['control']==c and cell['exact_equal_ranks']==equal,
                        'Exact min/mean/equality branch differs')
            cells.append(dict(cell,signal_index=i))
    available=sum(counts_by_event['available_by_event'])
    counts=dict(valid_price_observations=253*available,horizon_returns=3*available,
                ranked_horizon_observations=3*available,aggregated_scores=2*available)
    require(feature['source_available_cells']==feature['available_cells']==available and
            feature['unavailable_cells']==expected_events*width-available and
            feature['reason_counts']==dict(Counter(c['reason'] for c in cells)), 'Total availability differs')
    require(all(type(feature[key]) is int and feature[key]==value for key,value in counts.items()),
            'Nested descriptive counters differ')
    return cells,counts,counts_by_event


def check_saved_arrays(feature_path, transforms_path, original_path, feature):
    import numpy as np
    with np.load(feature_path,allow_pickle=False) as data:
        values={key:data[key] for key in data.files}
    with np.load(transforms_path,allow_pickle=False) as data:
        stored={key:data[key] for key in data.files}
    with np.load(original_path,allow_pickle=False) as data:
        require(len(data.files)==27, 'Original array count changed')
        original={key:data[key] for key in ('core','core_weight','state_calm10')}
    return check_array_records(values,stored,original,feature)


def check_array_records(values, stored, original, feature):
    """Only reconcile frozen arrays with frozen audit evidence and original C."""
    import numpy as np
    keys={'available','candidate','control','gain126','gain189','gain252','rank126','rank189','rank252'}
    require(set(values)==keys and set(stored)=={prefix+arm for prefix in ('transform_','delta_') for arm in ARMS[1:]},
            'Frozen numerical artifact key set differs')
    n,w=feature['events'],feature['symbols_per_event']
    require(values['available'].shape==(n,w) and values['available'].dtype==bool, 'Availability array differs')
    for key in keys:
        require(values[key].shape==(n,w), 'Feature matrix shape differs')
        if key!='available':
            require(np.issubdtype(values[key].dtype,np.floating) and not np.isinf(values[key]).any() and
                    np.array_equal(np.isfinite(values[key]),values['available']), 'Common finite domain differs')
        for i,event in enumerate(feature['event_history']):
            expected=[cell[key] for cell in event['cells']]
            if key!='available':expected=[np.nan if x is None else x for x in expected]
            require(np.array_equal(values[key][i],np.asarray(expected),equal_nan=True), 'NPZ/audit feature differs')
    core=np.asarray(original['core'])
    calm=np.asarray(original['state_calm10'])
    weights=np.asarray(original['core_weight'])
    require(core.shape==(n,w) and calm.shape==(n,) and calm.dtype==bool, 'Original state/core shape differs')
    require(weights.shape in ((n,),(n,1),(n,w)), 'Original core weight shape differs')
    if weights.ndim==1:weights=weights[:,None]
    weights=np.broadcast_to(weights,(n,w))
    expected=np.broadcast_to(np.where(calm,.5,.52)[:,None],(n,w))
    require(np.array_equal(weights,expected) and np.isfinite(core).all() and
            np.all(np.abs(core)<=weights*.5), 'Original weighted core/state differs')
    changed,available,maxima={},{},{}
    for arm,raw in zip(ARMS[1:],('candidate','control')):
        target=stored['transform_'+arm];delta=stored['delta_'+arm]
        require(target.shape==delta.shape==(n,w) and np.issubdtype(target.dtype,np.floating) and
                np.issubdtype(delta.dtype,np.floating) and not np.isinf(target).any() and np.isfinite(delta).all(),
                'Stored target/delta matrix differs')
        wanted=values[raw].copy()
        if n:wanted[0]=np.nan
        require(np.array_equal(target,wanted,equal_nan=True), 'Saved target has second rank or altered missing/partial')
        expected_delta=np.zeros((n,w))
        for i in range(n):
            for j in range(w):
                if math.isfinite(float(wanted[i,j])):
                    expected_delta[i,j]=.5*(float(weights[i,j])*float(wanted[i,j])-float(core[i,j]))
        require(np.array_equal(delta,expected_delta) and np.all(np.abs(delta)<=.26), 'Exact half-core formula differs')
        changed[arm]=int(np.count_nonzero(delta));available[arm]=int(np.isfinite(target).sum())
        maxima[arm]=float(np.max(np.abs(delta),initial=0.))
    return dict(blend=.5,max_absolute_delta=.26,changed_cells=changed,target_available_cells=available,
        observed_max_absolute_delta=maxima,exact_saved_gain_rank_aggregate_arrays=True,exact_saved_delta=True,
        first_partial_nan_target_zero_delta=True,missing_preserves_original_core=True,no_second_rank=True,
        original_support_and_single_episode='Verified by the bound independent preparation receipt; not rerun in reader')


def selection_from_saved(metrics, orders):
    """Reproduce only registered Boolean gates from recorded first88 metrics."""
    def guard(candidate, reference, gain):
        return (candidate['training']['cagr']>=reference['training']['cagr']+gain-1e-12 and
                candidate['training']['sharpe']>=reference['training']['sharpe']-.05 and
                candidate['training']['max_drawdown']>=reference['training']['max_drawdown']-.03 and
                all(candidate[era]['cagr']>=reference[era]['cagr']-.025 for era in ('early','recent')))
    changed=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
    control=[i for i in range(88) if set(orders[ARMS[1]][i]['symbols'])!=set(orders[ARMS[2]][i]['symbols'])]
    guards=dict(close_vs_incumbent=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[0]]['same_close_zero'],.005),
        open_vs_incumbent=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[0]]['next_open_cost'],0.),
        close_vs_mean=guard(metrics[ARMS[1]]['same_close_zero'],metrics[ARMS[2]]['same_close_zero'],0.),
        open_vs_mean=guard(metrics[ARMS[1]]['next_open_cost'],metrics[ARMS[2]]['next_open_cost'],0.),
        changed_support=len(changed)>=3 and any(i<53 for i in changed) and any(i>=53 for i in changed),
        changed_support_vs_control=len(control)>=3 and any(i<53 for i in control) and any(i>=53 for i in control))
    admitted=all(guards.values())
    return dict(primary=ARMS[1] if admitted else ARMS[0],admitted=admitted,guards=guards,
        changed_months=changed,changed_months_vs_control=control,first88_only=True,later34_used=False)


def self_test():
    """No artifact access: report-level formula and binding rejection tests."""
    import copy
    import numpy as np
    from datetime import date,timedelta
    checks=0
    def rejected(function):
        nonlocal checks
        try:function()
        except ValueError:checks+=1
        else:raise ValueError('Expected synthetic rejection')
    require(len(PREPARED_FILES)==7 and len(RESULT_FILES)==38,'Manifest cardinality');checks+=1
    fees=[]
    compare_incumbent_record({'fees_paid':1.+1e-10,'cagr':.4},{'fees_paid':1.,'cagr':.4},
                             'next_open_cost','metrics',fees)
    require(len(fees)==1,'Fee residual missing');checks+=1
    for path,location,a,b in [('same_close_zero','metrics',{'fees_paid':1.+1e-10},{'fees_paid':1.}),
            ('next_open_cost','metrics',{'cagr':.4+1e-12},{'cagr':.4}),
            ('next_open_cost','slices/first88',{'fees_paid':1.+1e-10},{'fees_paid':1.}),
            ('next_open_cost','metrics',{'fees_paid':1.+1e-7},{'fees_paid':1.})]:
        rejected(lambda:compare_incumbent_record(a,b,path,location,[]))
    check_recorded_metrics({'sharpe':None,'cagr':0.},{'sharpe':None,'cagr':0.},'nullable');checks+=1
    sessions=[];d=date(2014,1,1)
    while d<=date(2016,9,1):
        if d.weekday()<5:sessions.append(d.isoformat())
        d+=timedelta(days=1)
    days=['2016-08-17','2016-08-31'];names=['A','B','FUND']
    # Only report evidence is fabricated here; no actual prices or outcomes are read.
    events=[]
    for day in days:
        t=sessions.index(day);cells=[]
        for j,name in enumerate(names):
            gains=[float(j),float(2-j),float(j)]
            ranks=[(j+1)/3-.5,(3-j)/3-.5,(j+1)/3-.5]
            q=min(ranks);eq=ranks[0]==ranks[1]==ranks[2];control=q if eq else math.fsum(ranks)/3
            cells.append(dict(symbol=name,date=day,source_available=True,available=True,reason='available',
                end_price=6.,start_prices={str(h):6./(g+1.) for h,g in zip(HORIZONS,gains)},
                **{'gain'+str(h):g for h,g in zip(HORIZONS,gains)},
                **{'rank'+str(h):r for h,r in zip(HORIZONS,ranks)},
                candidate=q,control=control,exact_equal_ranks=eq))
        events.append(dict(date=day,symbols=names,session_position=t,price_start=sessions[t-294],
            price_end=sessions[t-42],horizons=list(HORIZONS),
            horizon_start_dates={str(h):sessions[t-42-h] for h in HORIZONS},cells=cells,
            missing_official_sessions=[],source_available=[True]*3,available=[True]*3,
            source_available_count=3,available_count=3))
    feature=dict(events=2,symbols_per_event=3,checked_cells=6,price_count=253,skip_sessions=42,
        horizons=list(HORIZONS),common_availability=True,ranking_performed=True,ranking_eligibility_filtered=False,
        trading_eligibility_modified=False,portfolio_returns_calculated=False,new_model_fits=0,predictive_model_fits=0,
        event_history=events,source_available_by_event=[3,3],available_by_event=[3,3],
        source_available_cells=6,available_cells=6,unavailable_cells=0,reason_counts={'available':6},
        valid_price_observations=1518,horizon_returns=18,ranked_horizon_observations=18,aggregated_scores=12)
    meta=dict(dates=days,symbols=[names,names])
    check_nested_audit(feature,meta,sessions,2,3);checks+=1
    for key in ('gain126','rank126','candidate','control'):
        bad=copy.deepcopy(feature);bad['event_history'][0]['cells'][0][key]+=1e-15
        rejected(lambda:check_nested_audit(bad,meta,sessions,2,3))
    bad=copy.deepcopy(feature);bad['horizon_returns']+=1
    rejected(lambda:check_nested_audit(bad,meta,sessions,2,3))
    keys=('available','candidate','control','gain126','gain189','gain252','rank126','rank189','rank252')
    values={k:np.array([[cell[k] for cell in event['cells']] for event in events]) for k in keys}
    original=dict(core=np.zeros((2,3)),core_weight=np.array([[.5],[.52]]),state_calm10=np.array([True,False]))
    stored={}
    for arm,raw in zip(ARMS[1:],('candidate','control')):
        target=values[raw].copy();target[0]=np.nan;delta=np.zeros((2,3))
        delta[1]=.5*.52*target[1]
        stored['transform_'+arm]=target;stored['delta_'+arm]=delta
    check_array_records(values,stored,original,feature);checks+=1
    bad=copy.deepcopy(stored);bad['transform_'+ARMS[1]][0]=0.
    rejected(lambda:check_array_records(values,bad,original,feature))
    bad=copy.deepcopy(stored);bad['delta_'+ARMS[1]][1,0]+=1e-15
    rejected(lambda:check_array_records(values,bad,original,feature))
    bad=copy.deepcopy(values);bad['gain126'][0,0]+=1e-15
    rejected(lambda:check_array_records(bad,stored,original,feature))
    metric=dict(training=dict(cagr=.5,sharpe=1.5,max_drawdown=-.2),
                early=dict(cagr=.5),recent=dict(cagr=.5))
    metrics={arm:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')} for arm in ARMS}
    orders={arm:[dict(symbols=['A']) for _ in range(122)] for arm in ARMS}
    selection=selection_from_saved(metrics,orders)
    require(not selection['admitted'] and not selection['guards']['close_vs_incumbent'] and
            not selection['guards']['changed_support'],'First88 synthetic rejection');checks+=1
    for path in metrics[ARMS[1]]:metrics[ARMS[1]][path]['training']['cagr']=.51
    for i in (0,1,53):orders[ARMS[1]][i]['symbols']=['B']
    selection=selection_from_saved(metrics,orders)
    require(selection['admitted'],'First88 synthetic pass');checks+=1
    for i in range(88,122):orders[ARMS[1]][i]['symbols']=['LATER']
    require(selection_from_saved(metrics,orders)==selection,'Later34 may not change selection');checks+=1
    print(json.dumps(dict(passed=True,synthetic_checks=checks,actual_artifacts_read=False,
                         actual_features_computed=False,actual_portfolio_returns_computed=False)))


def review(completion_sha, check_only=False):
    require(not (OUT/'RESEARCH_REVIEW.json').exists(), 'Review is write-once')
    evidence = Evidence()
    closure = evidence.load(OUT/'COMPLETION.json', completion_sha)
    require(closure['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED', '199 is not closed')
    plan_sha = closure['plan_sha256']
    plan = evidence.load(OUT/'PLAN.json', plan_sha)
    require(plan['schema'] == 'monthly_nested_trend199_v1' and
            plan['contract']['arms'] == list(ARMS) and plan['contract']['blend'] == .5 and
            plan['contract']['max_absolute_delta'] == .26 and plan['contract']['max_absolute_rank'] == .5,
            'Unexpected nested half-core registration')
    context = evidence.load(CONTEXT, CONTEXT_SHA)
    require(plan['contract']['context'] == context and plan['contract']['prospective_context_sha256'] == CONTEXT_SHA,
            'Prospective context mismatch')
    before = evidence.load(OUT/'STATE_BEFORE.json', plan['prior_central_sha256'])
    require(len(before['completed']) == 196 and sum(x['trials'] for x in before['completed']) == 1105,
            'Wrong pre-registration counters')
    for key in ('actual_new_feature_evaluations_before_registration','actual_new_model_fits_before_registration',
                'actual_horizon_gains_before_registration','actual_horizon_ranks_before_registration',
                'actual_new_returns_before_registration'):
        require(type(plan[key]) is int and plan[key] == 0, 'Pre-registration actual computation reported')
    results_completion = evidence.manifest(OUT/'results', closure['result_completion_sha256'], plan_sha)
    require(results_completion['paths'] == 9 and results_completion['slices'] == 45,
            'Nine paths and45 slices required')
    frozen = evidence.load(OUT/'selected/FROZEN.json', results_completion['frozen_sha256'])
    require(frozen['plan_sha256'] == plan_sha, 'Frozen selection plan mismatch')
    prepared = evidence.manifest(OUT/'prepared', frozen['prepared_sha256'], plan_sha)
    training = evidence.load(OUT/'selected/TRAIN_RESULTS.json', frozen['training_sha256'])
    orders = evidence.load(OUT/'prepared/ORDERS.json', frozen['orders_sha256'])
    results = evidence.load(OUT/'results/RESULTS.json')
    meta = evidence.load(OUT/'prepared/META.json')
    feature = evidence.load(OUT/'prepared/FEATURE_AUDIT.json')
    validation = evidence.load(OUT/'prepared/INDEPENDENT_PREPARE.json')
    outcome = evidence.load(OUT/'results/INDEPENDENT_OUTCOME.json')
    source_parity = evidence.load(OUT/'prepared/SOURCE_PARITY.json')
    require(source_parity['raw_source_hashes'] == plan['raw_source_hashes'] and
            source_parity['price_source_binding'] == plan['price_source_binding'],
            'Prepared raw source lineage differs from PLAN')
    require(results.get('nested_trend') is True and results.get('same_support_both_arms') is True and
            results.get('observer_promotion') is False and results.get('strict_pit') is False,
            'Result interpretation flags differ')

    review_binding = plan['numerical_code_review']
    code_review = evidence.load(review_binding['path'], review_binding['sha256'])
    require(code_review['passed'] is True and code_review['unresolved_blockers'] == [],
            'Numerical code review unresolved')
    for name, digest in {**code_review['code_sha256'], **code_review['input_sha256']}.items():
        evidence.verify(name, digest)
    tests_path = ROOT/'work/monthly_nested_trend199_TESTS.json'
    require(plan['input_sha256'][str(tests_path)] == TEST_RECEIPT_SHA, 'Unexpected formal test receipt')
    tests = evidence.load(tests_path, TEST_RECEIPT_SHA)
    require(tests['passed'] is True and tests['synthetic_only'] is True and
            type(tests['tests']) is int and tests['tests'] == 62, '62 successful formal synthetic tests required')
    for name, digest in tests['code_sha256'].items():
        evidence.verify(name, digest)

    # The already-closed predecessor pins the exact incumbent replay comparator.
    predecessor = plan['predecessor_binding']
    require(predecessor['directory'] == PRIOR.name, 'Unexpected predecessor')
    prior_pins = predecessor['input_sha256']
    prior_closed = evidence.load(PRIOR/'COMPLETION.json', prior_pins[str(PRIOR/'COMPLETION.json')])
    prior_completed = evidence.load(PRIOR/'results/COMPLETION.json', prior_closed['result_completion_sha256'])
    prior_results = evidence.load(PRIOR/'results/RESULTS.json', prior_completed['artifacts_sha256']['RESULTS.json'])
    require(prior_closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            prior_completed['status'] == 'COMPLETE', 'Prior incumbent evidence incomplete')
    fee_diagnostic = diagnose_incumbent_exports(evidence, prior_completed, results, prior_results)
    fee_total_differences = []

    require(set(orders) == set(ARMS) == set(results['arms']) == set(outcome['paths']) == set(training),
            'Three arms required')
    require(len(meta['dates']) == len(meta['ends']) == 122 and
            all(len(x) == 122 for x in orders.values()), '122 orders per arm required')
    require(results['selection'] == frozen['selection'], 'Selection mismatch')
    require(frozen['selection'] == selection_from_saved(training, orders), 'Saved first88 guards differ')
    selection = results['selection']
    require(selection['first88_only'] is True and selection['later34_used'] is False,
            'Selection may only use first88')
    require(selection['admitted'] == all(selection['guards'].values()) and
            selection['primary'] == (ARMS[1] if selection['admitted'] else ARMS[0]), 'Selection routing differs')
    require(validation['prefixes'] == 123 and validation['feature_prefixes'] == [0, 1, 53, 88, 122] and
            validation['feature_prefix_sources_truncated'] is True and
            validation['original_orders_exact'] is True and validation['original_arrays_unchanged'] is True and
            validation['original_array_count'] == prepared['original_array_count'] == closure['original_array_count'] == 27,
            'Original arrays or prefix checks incomplete')
    for record in (validation, prepared, closure):
        for key in ('new_model_fits', 'predictive_model_fits', 'new_predictive_model_fits'):
            require(type(record[key]) is int and record[key] == 0, 'Unexpected fit count: '+key)
    require(feature['new_model_fits'] == feature['predictive_model_fits'] == 0 and
            validation['feature']['predictive_model_fits'] == 0, 'Feature fit count differs')
    require(validation['feature']['passed'] is True and validation['delta']['passed'] is True and
            validation['delta']['blend'] == .5 and validation['delta']['max_absolute_delta'] == .26,
            'Independent prepare or fixed delta bound missing')
    require(outcome['passed'] is True and outcome['checked_paths'] == 9 and outcome['checked_slices'] == 45 and
            outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True and
            outcome['terminal_liquidation'] is False, 'Independent outcome incomplete')
    evidence.verify(ROOT/'work/ptr169_results_independent.py', outcome['accounting_source_sha256'])
    require(closure['protected_files'] == 21444 and closure['observers_modified'] is False and
            closure['new_trials'] == 3, 'Closure protection or trial count differs')
    central = evidence.load(CENTRAL)
    require(central['current'] == OUT.name and central['current_status'] == 'completed_independently_verified' and
            central['registered_queue'] == [] and len(central['completed']) == 197 and
            sum(row['trials'] for row in central['completed']) == 1108,
            'Central199 closure/counters differ; review requires the immediate closed state')
    latest = central['completed'][-1]
    require(latest['directory'] == OUT.name and latest['completion']['sha256'] == completion_sha and
            source_path(latest['completion']['path']) == OUT/'COMPLETION.json' and
            latest['primary'] == selection['primary'] and latest['new_admitted'] == int(selection['admitted']) and
            central['best_training_selected_candidate'] == closure['best_training_selected_candidate'],
            'Central closure receipt or selection differs')
    calendar = evidence.load(CALENDAR, plan['input_sha256'][str(CALENDAR)])
    cells, counts, series = check_nested_audit(feature, meta, calendar['sessions'])
    source_count, available = feature['source_available_cells'], feature['available_cells']
    for record in (feature, validation['feature'], validation, prepared, closure):
        require(all(type(record[key]) is int and record[key] == value for key, value in counts.items()),
                'Nested horizon accounting differs')
    require(all(validation['feature'][key] == value for key, value in series.items()) and
            validation['feature']['source_available_cells'] == source_count and
            validation['feature']['available_cells'] == available and
            validation['feature']['checked_cells'] == 5978, 'Independent source/feature domains differ')
    evidence.verify(INDEPENDENT, INDEPENDENT_SHA)
    require(validation['feature']['numerical_comparison'] == NUMERICAL_COMPARISON,
            'Independent exact gain/rank/aggregate rule changed')
    require(validation['feature']['computes_returns'] is False and
            validation['feature']['new_model_fits'] == 0, 'Independent feature scope differs')
    for flag in ('independent_scalar_rebuild','nested_horizons_checked','common_endpoint_checked',
                 'common_rank_domain_checked','original_pct_ranks_checked','matched_aggregation_checked',
                 'full253_price_support_checked','official_session_windows_checked','monthend_queries_checked',
                 'exact_common_availability','audit_checked','no_future_or_outcome_inputs'):
        require(validation['feature'].get(flag) is True, 'Missing independent nested proof: '+flag)
    for flag in ('half_core_formula_checked','original_pct_ranks_preserved','first_partial_unchanged',
                 'missing_preserves_core','core_weights_checked','no_second_rank','independent_delta_rebuild'):
        require(validation['delta'].get(flag) is True, 'Missing independent half-core proof: '+flag)
    require(validation['original_core_metadata_and_support_preserved'] is True and
            validation['single_original_episode'] is True, 'Original support or one-episode proof missing')
    evidence.verify(ORIGINAL_DATA, plan['input_sha256'][str(ORIGINAL_DATA)])
    integration = check_saved_arrays(OUT/'prepared/FEATURES.npz', OUT/'prepared/TRANSFORMS_DELTAS.npz',
                                     ORIGINAL_DATA, feature)
    require(integration['changed_cells'] == validation['delta']['changed_cells'] and
            integration['target_available_cells'] == validation['delta']['target_available_cells'],
            'Stored half-core count differs from independent receipt')

    summaries = {}
    checked_slices = 0
    checked_training = 0
    for arm in ARMS:
        require(set(results['arms'][arm]) == set(PATHS) == set(outcome['paths'][arm]), 'Missing execution paths')
        summaries[arm] = {}
        for path in PATHS:
            result, independent = results['arms'][arm][path], outcome['paths'][arm][path]
            require(independent['passed'] is True and independent['monthly_executions'] == 122 and
                    set(independent['slices']) == set(BOUNDS) == set(result['slices']), 'Incomplete path slices')
            require(result['metrics'] == result['slices']['full122'], 'Full slice mismatch')
            for period in BOUNDS:
                check_recorded_metrics(independent['slices'][period], result['slices'][period], arm+'/'+path+'/'+period)
                checked_slices += 1
            if arm == ARMS[0]:
                prior = prior_results['arms'][arm][path]
                compare_incumbent_record(result['metrics'], prior['metrics'], path, 'metrics', fee_total_differences)
                compare_incumbent_record(result['slices'], prior['slices'], path, 'slices', fee_total_differences)
            summaries[arm][path] = dict(metrics={k: result['metrics'][k] for k in METRICS},
                periods={period: {k: result['slices'][period][k] for k in METRICS} for period in BOUNDS})
        require(set(training[arm]) == {'same_close_zero', 'next_open_cost'}, 'Unexpected selection paths')
        for path in training[arm]:
            require(set(training[arm][path]) == {'training', 'early', 'recent'}, 'Unexpected selection eras')
            for small, full in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                check_recorded_metrics(outcome['paths'][arm][path]['slices'][full], training[arm][path][small],
                                       'training/'+arm+'/'+path+'/'+small)
                checked_training += 1

    changes = dict(candidate_vs_incumbent=changed_holdings(orders, ARMS[0], ARMS[1]),
        control_vs_incumbent=changed_holdings(orders, ARMS[0], ARMS[2]),
        candidate_vs_control=changed_holdings(orders, ARMS[2], ARMS[1]))
    require([x['index'] for x in changes['candidate_vs_incumbent']['months'] if x['index'] < 88]
            == selection['changed_months'], 'Candidate selection changes differ')
    require([x['index'] for x in changes['candidate_vs_control']['months'] if x['index'] < 88]
            == selection['changed_months_vs_control'], 'Control selection changes differ')
    retained = closure['best_training_selected_candidate'] == prior_closed['best_training_selected_candidate']
    if not selection['admitted']:
        require(retained, 'Rejected candidate changed historical incumbent')
    require(closure['target_reached'] == (closure['best_training_selected_candidate']['cagr'] >= .7),
            'Target status differs')
    evidence.verify(Path(__file__))
    # Rehash every consumed artifact before writing; the reader never rewrites it.
    for path, digest in list(evidence.pins.items()):
        evidence.verify(path, digest)
    receipt = dict(schema='monthly_nested_trend199_research_review_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_refs=evidence.pins,
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED', selection=selection,
        failed_training_guards=[key for key, passed in selection['guards'].items() if not passed],
        best_retained=closure['best_training_selected_candidate'], historical_best_unchanged=retained,
        exact_incumbent_metrics_equal_predecessor=not fee_total_differences,
        exact_incumbent_metrics_except_disclosed_fee_totals=True,
        incumbent_fee_total_differences=fee_total_differences, incumbent_export_diagnostic=fee_diagnostic,
        new_candidate_summary=summaries,
        holding_changes=changes,
        nested_trend_support=dict(all_cells=5978, source_available_cells=source_count,
            available_cells=available, unavailable_cells=5978-available, **series, **counts,
            source_unavailable=[x for x in cells if not x['source_available']],
            reason_counts=dict(Counter(x['reason'] for x in cells)),
            equal_three_rank_cells=sum(x['exact_equal_ranks'] for x in cells),
            candidate_control_different_cells=sum(x['available'] and x['candidate'] != x['control'] for x in cells),
            exact_simple_gain_endpoints_checked=True, exact_original_average_percentiles_checked=True,
            exact_minimum_and_matched_mean_checked=True, no_second_rank=True,
            numerical_comparison=NUMERICAL_COMPARISON,
            full253_source_check='Accepted from the hash-bound independent preparation receipt. This reader checks saved endpoint prices, gains, full common-domain ranks, aggregates, arrays and half-core transformations without rereading raw price panels.'),
        half_core_integration=integration,
        validation=dict(synthetic_tests=tests['tests'], independent_cells=5978, independent_execution_paths=9,
            independent_period_slices=checked_slices, training_metric_groups=checked_training,
            order_prefixes=123, source_prefixes=[0, 1, 53, 88, 122], protected_files=21444,
            central_completed_entries=197, central_trials=1108,
            new_model_fits=0, new_predictive_model_fits=0),
        limitations=plan['contract']['limitation'], reused_historical_sample=True, pristine_oos=False,
        observers_modified=False, new_factors_computed_in_this_review=False,
        new_portfolio_returns_computed_in_this_review=False, target_reached=closure['target_reached'])
    require(checked_slices == 45 and checked_training == 18, 'Review statistics coverage incomplete')
    if check_only:
        print(json.dumps(dict(status='CHECK_ONLY_PASS', writes=False, fee_total_differences=fee_total_differences,
            export_diagnostic=fee_diagnostic, checked_slices=checked_slices, checked_training=checked_training),
            ensure_ascii=False, allow_nan=False))
        return
    path = OUT/'RESEARCH_REVIEW.json'
    with path.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        result=receipt['result'], selection=selection, historical_best_unchanged=retained,
        target_reached=receipt['target_reached'], holding_change_counts={key: value['counts'] for key, value in changes.items()},
        summary=summaries), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-sha256')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    arguments = parser.parse_args()
    if arguments.self_test:
        require(arguments.completion_sha256 is None and not arguments.check_only, 'Self-test is synthetic only')
        self_test()
    else:
        require(arguments.completion_sha256 is not None, 'Completion SHA required')
        review(arguments.completion_sha256, arguments.check_only)

