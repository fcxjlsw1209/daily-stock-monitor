"""Write-once review of closed198; no price, model, or replay imports.

Run only after the root has confirmed198 closure. This reader verifies existing
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
OUT = ROOT / 'outputs/monthly_event_risk_payoff198'
PRIOR = ROOT / 'outputs/monthly_recovery_durability197'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
ARMS = ('incumbent_local67', 'event_risk_payoff', 'inverse_risk_mass_control')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
BOUNDS = {'full122': (0, 122), 'first88': (0, 88), 'early53': (0, 53),
          'recent35': (53, 88), 'later34': (88, 122)}
METRICS = ('start', 'end', 'cagr', 'sharpe', 'max_drawdown')
COUNTS = ('valid_price_observations', 'selected_anchors', 'pre_event_return_observations',
          'matured_payoffs', 'valid_event_payoffs')
PRIOR_READERS = {
    'work/majority196_review.py': 'edfce347bd4eee4c82235023a5e1b93870a9f17b40628ba4a8a2f431c7dc2b3f',
    'work/majority196_review_v2.py': '7282e8734f6d0b01ce64b8e9ffa6088500cc0d49c06024bf2f40bfb212861e69',
    'work/durability197_review.py': '42ceace08c86ed04118d58d50b23628162f8f0383c5269ed5ac0972f921a0693'}
ENGINE = ROOT/'outputs/monthly_factor_search_v2/validate.py'
ENGINE_SHA = 'b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4'
FEE_TOTAL_ABSOLUTE_TOLERANCE = 1e-9
INDEPENDENT = ROOT/'work/monthly_event_risk_payoff198_independent.py'
INDEPENDENT_SHA = '2b33b8ce09ff719fee5b4e6a591cd9c5353da7411d3c22b4ef956dd6b04d6b1e'
FLOAT_RTOL, FLOAT_ATOL = 2e-12, 1e-14
NUMERICAL_COMPARISON = dict(relative_tolerance=FLOAT_RTOL, absolute_tolerance=FLOAT_ATOL,
    sigma_absolute_tolerance=0., raw_ranks_exact=True,
    scalar_risk='math.fsum two-pass sample variance')


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
        differences.append(dict(path=path, field=location, previous197=prior, current198=current,
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
                    previous197=old['fee'], current198=new['fee'], signed_difference=new['fee']-old['fee']))
        totals = {}
        for label, holds, result in (('previous197', old_holds, prior_results['arms'][ARMS[0]][path]),
                                      ('current198', new_holds, current_results['arms'][ARMS[0]][path])):
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


def self_test():
    differences = []
    compare_incumbent_record({'fees_paid': 1.+1e-10, 'cagr': .4}, {'fees_paid': 1., 'cagr': .4},
                             'next_open_cost', 'metrics', differences)
    require(len(differences) == 1, 'Allowed fee residual was not recorded')
    for path, location, a, b in [('next_open_cost', 'metrics', {'fees_paid': 1.+1e-7}, {'fees_paid': 1.}),
            ('same_close_zero', 'metrics', {'fees_paid': 1.+1e-10}, {'fees_paid': 1.}),
            ('next_open_cost', 'metrics', {'cagr': .4+1e-12}, {'cagr': .4}),
            ('next_open_cost', 'slices/first88', {'fees_paid': 1.+1e-10}, {'fees_paid': 1.})]:
        try:
            compare_incumbent_record(a, b, path, location, [])
        except ValueError:
            pass
        else:
            raise ValueError('Overbroad fee exception')
    exact = []
    compare_incumbent_record({'fees_paid': 1., 'cagr': .4}, {'fees_paid': 1., 'cagr': .4},
                             'next_open_cost', 'metrics', exact)
    require(not exact, 'Exact fees must not create an exception')
    import copy
    def sample(risks_and_labels):
        anchors = []
        for i, (returns, intended_y) in enumerate(risks_and_labels):
            s = 64+21*i
            mean = math.fsum(returns)/63
            sigma = math.sqrt(math.fsum((r-mean)**2 for r in returns)/62)
            maturity = 90.*math.exp(intended_y)
            y = math.log(maturity)-math.log(90.)
            anchors.append(dict(anchor_index=s, reference_index=s-1, reference_price=100., anchor_price=90.,
                risk_price_start_index=s-64, risk_price_end_index=s-1,
                risk_return_first_index=s-63, risk_return_last_index=s-1,
                pre_event_simple_returns=returns, sigma=sigma, maturity_index=s+21,
                maturity_price=maturity, payoff=y, inverse_sigma=1./sigma if sigma else None,
                standardized_payoff=y/sigma if sigma else None))
        n = len(anchors); zero = any(a['sigma'] == 0 for a in anchors)
        ysum = math.fsum(a['payoff'] for a in anchors); denom = (n+20)*math.sqrt(21)
        invsum = None if zero else math.fsum(a['inverse_sigma'] for a in anchors)
        stdsum = None if zero else math.fsum(a['standardized_payoff'] for a in anchors)
        branch = (None if zero else 'single_event' if n == 1 else
                  'equal_risks' if all(a['sigma'] == anchors[0]['sigma'] for a in anchors) else
                  'equal_payoffs' if all(a['payoff'] == anchors[0]['payoff'] for a in anchors) else None)
        q = None if zero else stdsum/denom
        c = None if zero else q if branch else (invsum/n*ysum)/denom
        return dict(source_available=True, available=not zero,
            reason='zero_pre_event_sigma' if zero else 'available', selected_anchors=n,
            valid_event_payoffs=0 if zero else n, anchor_indices=[a['anchor_index'] for a in anchors],
            anchor_events=anchors, payoff_sum=ysum, inverse_risk_sum=invsum,
            standardized_payoff_sum=stdsum, mean_inverse_risk=None if zero else invsum/n,
            denominator=denom, identity_branch=branch, candidate=q, control=c)
    r = [0., .01]*31+[0.]
    samples = [sample([(r, -.1)]), sample([(r, .2), ([2*x for x in r], -.1)]),
               sample([(r, -.1), (r, .2)]), sample([(r, .1), ([2*x for x in r], .1)]),
               sample([([0.]*63, .2), (r, -.1)])]
    for cell in samples:
        check_payoff_cell(cell)
    require(samples[0]['candidate'] < 0 and samples[1]['candidate'] > samples[1]['control'],
            'Synthetic cases must exercise signed scores and Q greater than C')
    checks = 11
    for index, key, value in [(0, 'identity_branch', None), (1, 'candidate', .123),
                              (4, 'valid_event_payoffs', 1), (4, 'available', True)]:
        changed = copy.deepcopy(samples[index]); changed[key] = value
        try:
            check_payoff_cell(changed)
        except ValueError:
            checks += 1
        else:
            raise ValueError('Synthetic mutation was not rejected: '+key)
    changed = copy.deepcopy(samples[1]); changed['anchor_events'][0]['risk_price_end_index'] += 1
    try:
        check_payoff_cell(changed)
    except ValueError:
        checks += 1
    else:
        raise ValueError('Loss-day risk leakage was not rejected')
    print(json.dumps(dict(synthetic_checks=checks, passed=True, actual_artifacts_read=False)))


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close_number(actual, expected, label, absolute=FLOAT_ATOL):
    require(finite_number(actual) and finite_number(expected) and
            abs(actual-expected) <= absolute+FLOAT_RTOL*abs(expected), label)


def check_payoff_cell(cell):
    """Recheck recorded event returns and formulas, never load source panels."""
    require(type(cell['source_available']) is bool and type(cell['available']) is bool,
            'Boolean source/feature support required')
    anchors = cell['anchor_events']
    indices = [a['anchor_index'] for a in anchors]
    n = cell['selected_anchors']
    require(type(n) is int and 0 <= n <= 8 and len(anchors) == n and
            cell['anchor_indices'] == indices and indices == sorted(set(indices)) and
            all(b-a >= 21 for a, b in zip(indices, indices[1:])), 'Selected-anchor cooldown/count differs')
    require(type(cell['valid_event_payoffs']) is int, 'Integer valid payoff count required')
    summary_fields = ('payoff_sum', 'inverse_risk_sum', 'standardized_payoff_sum',
                      'mean_inverse_risk', 'denominator', 'identity_branch')
    if not cell['source_available'] or not anchors:
        require(not anchors and not cell['available'] and cell['valid_event_payoffs'] == 0 and
                cell['candidate'] is None and cell['control'] is None and
                all(cell[key] is None for key in summary_fields), 'Empty source/event must remain missing')
        require(cell['reason'] == 'no_selected_anchors' if cell['source_available'] else
                cell['reason'] in ('insufficient_history', 'missing_symbol', 'missing_official_session',
                                   'nonnumeric_adjusted_price', 'invalid_adjusted_price'),
                'Empty source/event reason differs')
        return
    risks, labels, inverses, standardized = [], [], [], []
    for anchor in anchors:
        s = anchor['anchor_index']
        require(type(s) is int and 64 <= s <= 231 and anchor['reference_index'] == s-1 and
                anchor['risk_price_start_index'] == s-64 and anchor['risk_price_end_index'] == s-1 and
                anchor['risk_return_first_index'] == s-63 and anchor['risk_return_last_index'] == s-1 and
                anchor['maturity_index'] == s+21 <= 252, 'Pre-loss risk/payoff chronology differs')
        require(all(finite_number(anchor[k]) and anchor[k] > 0 for k in
                    ('reference_price', 'anchor_price', 'maturity_price')) and
                anchor['anchor_price'] < anchor['reference_price'], 'Invalid strict negative anchor evidence')
        returns = anchor['pre_event_simple_returns']
        require(isinstance(returns, list) and len(returns) == 63 and
                all(finite_number(r) and r >= -1 for r in returns), '63 finite simple returns required')
        mean = math.fsum(returns)/63
        scalar_sigma = math.sqrt(math.fsum((r-mean)**2 for r in returns)/62)
        sigma = anchor['sigma']
        require(finite_number(sigma) and sigma >= 0, 'Nonfinite or negative required sigma')
        close_number(sigma, scalar_sigma, 'Stored risk differs from independent two-pass sample std', absolute=0.)
        payoff = math.log(anchor['maturity_price'])-math.log(anchor['anchor_price'])
        require(finite_number(payoff) and anchor['payoff'] == payoff, 'Recorded log payoff differs')
        risks.append(sigma); labels.append(payoff)
        if sigma == 0:
            require(anchor['inverse_sigma'] is None and anchor['standardized_payoff'] is None,
                    'Zero risk cannot have finite-normalized evidence')
        else:
            inverse, normalized = 1./sigma, payoff/sigma
            require(finite_number(inverse) and finite_number(normalized) and
                    anchor['inverse_sigma'] == inverse and anchor['standardized_payoff'] == normalized,
                    'Inverse risk or paired normalized payoff differs')
            inverses.append(inverse); standardized.append(normalized)
    total_y, denominator = math.fsum(labels), (n+20)*math.sqrt(21)
    require(cell['payoff_sum'] == total_y and cell['denominator'] == denominator,
            'Matured payoff count/shrinkage denominator differs')
    if any(sigma == 0 for sigma in risks):
        require(cell['available'] is False and cell['reason'] == 'zero_pre_event_sigma' and
                cell['valid_event_payoffs'] == 0 and cell['candidate'] is None and cell['control'] is None and
                all(cell[k] is None for k in ('inverse_risk_sum', 'standardized_payoff_sum',
                                              'mean_inverse_risk', 'identity_branch')),
                'Zero risk must invalidate the whole shared cell without deleting events')
        return
    sum_inverse, sum_standardized = math.fsum(inverses), math.fsum(standardized)
    mean_inverse = sum_inverse/n
    q = sum_standardized/denominator
    branch = ('single_event' if n == 1 else 'equal_risks' if all(x == risks[0] for x in risks)
              else 'equal_payoffs' if all(x == labels[0] for x in labels) else None)
    c = q if branch else (mean_inverse*total_y)/denominator
    require(all(finite_number(x) for x in (sum_inverse, sum_standardized, mean_inverse, q, c)),
            'Derived risk/payoff arithmetic must remain finite')
    require(cell['available'] is True and cell['reason'] == 'available' and cell['valid_event_payoffs'] == n and
            cell['inverse_risk_sum'] == sum_inverse and cell['standardized_payoff_sum'] == sum_standardized and
            cell['mean_inverse_risk'] == mean_inverse and cell['candidate'] == q and cell['control'] == c and
            cell['identity_branch'] == branch, 'Q/C formula or exact-equivalence branch differs')


def check_payoff_audit(feature, meta, sessions):
    """Stored indices, support, arithmetic, ranks and all four event-count lists."""
    keys = ('source_available_by_event', 'available_by_event', 'selected_anchors_by_event',
            'valid_event_payoffs_by_event')
    series = {key: feature[key] for key in keys}
    require(all(isinstance(v, list) and len(v) == 122 and all(type(x) is int for x in v)
                for v in series.values()), 'Invalid per-event risk/payoff counts')
    require(feature['checked_cells'] == 5978 and feature['events'] == 122 and
            feature['symbols_per_event'] == 49 and feature['price_count'] == 253 and
            feature['skip_sessions'] == 42 and feature['anchor_first'] == 64 and
            feature['anchor_last'] == 231 and feature['risk_returns'] == 63 and
            feature['payoff_horizon'] == feature['cooldown'] == 21 and feature['prior_count'] == 20,
            'Risk/payoff window contract differs')
    require(feature['common_availability'] is True and feature['ranking_performed'] is True and
            feature['ranking_eligibility_filtered'] is False and feature['trading_eligibility_modified'] is False and
            feature['portfolio_returns_calculated'] is False, 'Feature scope differs')
    require(len(feature['event_history']) == 122, 'Missing event audits')
    lookup = {date: i for i, date in enumerate(sessions)}
    require(len(lookup) == len(sessions), 'Duplicate official sessions')
    cells = []
    for i, event in enumerate(feature['event_history']):
        source, available, selected, valid = (series[key][i] for key in keys)
        require(0 <= available <= source <= 49 and 0 <= selected <= 8*source and
                available <= valid <= 8*available and valid <= selected, 'Invalid event support/count ordering')
        require(event['date'] == meta['dates'][i] and event['symbols'] == meta['symbols'][i] and
                len(event['cells']) == 49, 'Event identities differ')
        t = lookup[event['date']]
        require(t >= 294 and event['session_position'] == t and
                event['price_start'] == sessions[t-294] and event['price_end'] == sessions[t-42] and
                event['anchor_first_date'] == sessions[t-230] and event['anchor_last_date'] == sessions[t-63],
                'Causal formation/anchor window differs')
        if not (i == 0 and event['date'] == '2016-08-17'):
            require(t+1 < len(sessions) and sessions[t+1][:7] != sessions[t][:7], 'Non-month-end query')
        require(event['source_available_count'] == source and event['available_count'] == available and
                event['source_available'] == [c['source_available'] for c in event['cells']] and
                event['available'] == [c['available'] for c in event['cells']] and
                sum(event['source_available']) == source and sum(event['available']) == available,
                'Event source/feature support differs')
        require(sum(c['selected_anchors'] for c in event['cells']) == selected and
                sum(c['valid_event_payoffs'] for c in event['cells']) == valid, 'Event anchor accounting differs')
        valid_cells = [c for c in event['cells'] if c['available']]
        for symbol, cell in zip(event['symbols'], event['cells']):
            require(cell['symbol'] == symbol and cell['date'] == event['date'], 'Cell identity mismatch')
            check_payoff_cell(cell)
            for raw in ('candidate', 'control'):
                if cell['available'] and available > 1:
                    less = sum(c[raw] < cell[raw] for c in valid_cells)
                    equal = sum(c[raw] == cell[raw] for c in valid_cells)
                    rank = (less+(equal+1)/2-1)/(available-1)-.5
                else:
                    rank = 0.
                require(cell[raw+'_rank'] == rank and -.5 <= rank <= .5,
                        'Common-universe centered average-tie rank differs')
            cells.append(dict(cell, signal_index=i))
    counts = dict(valid_price_observations=253*sum(series[keys[0]]), selected_anchors=sum(series[keys[2]]),
                  pre_event_return_observations=63*sum(series[keys[2]]), matured_payoffs=sum(series[keys[2]]),
                  valid_event_payoffs=sum(series[keys[3]]))
    require(feature['source_available_cells'] == sum(series[keys[0]]) and
            feature['available_cells'] == sum(series[keys[1]]) and
            feature['unavailable_cells'] == 5978-sum(series[keys[1]]), 'Total support mismatch')
    require(feature['reason_counts'] == dict(Counter(c['reason'] for c in cells)), 'Reason counts differ')
    return cells, counts, series


def review(completion_sha, check_only=False):
    require(not (OUT/'RESEARCH_REVIEW.json').exists(), 'Review is write-once')
    evidence = Evidence()
    closure = evidence.load(OUT/'COMPLETION.json', completion_sha)
    require(closure['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED', '198 is not closed')
    plan_sha = closure['plan_sha256']
    plan = evidence.load(OUT/'PLAN.json', plan_sha)
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

    review_binding = plan['numerical_code_review']
    code_review = evidence.load(review_binding['path'], review_binding['sha256'])
    require(code_review['passed'] is True and code_review['unresolved_blockers'] == [],
            'Numerical code review unresolved')
    for name, digest in {**code_review['code_sha256'], **code_review['input_sha256']}.items():
        evidence.verify(name, digest)
    tests_path = ROOT/'work/monthly_event_risk_payoff198_TESTS.json'
    tests = evidence.load(tests_path, plan['input_sha256'][str(tests_path)])
    require(tests['passed'] is True and tests['synthetic_only'] is True and
            type(tests['tests']) is int and tests['tests'] > 0, 'Successful synthetic test receipt required')
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
            validation['delta']['dose'] == .01 and validation['delta']['max_absolute_delta'] == .005,
            'Independent prepare or fixed delta bound missing')
    require(outcome['passed'] is True and outcome['checked_paths'] == 9 and outcome['checked_slices'] == 45 and
            outcome['independent_accounts'] is True and outcome['independent_of_production_model_and_engine'] is True and
            outcome['terminal_liquidation'] is False, 'Independent outcome incomplete')
    evidence.verify(ROOT/'work/ptr169_results_independent.py', outcome['accounting_source_sha256'])
    require(closure['protected_files'] == 21444 and closure['observers_modified'] is False and
            closure['new_trials'] == 3, 'Closure protection or trial count differs')
    central = evidence.load(CENTRAL)
    require(central['current'] == OUT.name and central['current_status'] == 'completed_independently_verified' and
            central['registered_queue'] == [] and len(central['completed']) == 196 and
            sum(row['trials'] for row in central['completed']) == 1105,
            'Central198 closure/counters differ; review requires the immediate closed state')
    latest = central['completed'][-1]
    require(latest['directory'] == OUT.name and latest['completion']['sha256'] == completion_sha and
            source_path(latest['completion']['path']) == OUT/'COMPLETION.json' and
            latest['primary'] == selection['primary'] and latest['new_admitted'] == int(selection['admitted']) and
            central['best_training_selected_candidate'] == closure['best_training_selected_candidate'],
            'Central closure receipt or selection differs')
    calendar = evidence.load(CALENDAR, plan['input_sha256'][str(CALENDAR)])
    cells, counts, series = check_payoff_audit(feature, meta, calendar['sessions'])
    source_count, available = feature['source_available_cells'], feature['available_cells']
    for record in (feature, validation['feature'], validation, prepared, closure):
        require(all(type(record[key]) is int and record[key] == value for key, value in counts.items()),
                'Risk-payoff accounting differs')
    require(all(validation['feature'][key] == value for key, value in series.items()) and
            validation['feature']['source_available_cells'] == source_count and
            validation['feature']['available_cells'] == available and
            validation['feature']['checked_cells'] == 5978, 'Independent source/feature domains differ')
    evidence.verify(INDEPENDENT, INDEPENDENT_SHA)
    require(validation['feature']['numerical_comparison'] == NUMERICAL_COMPARISON,
            'Independent numerical tolerances or exact-rank rule changed')
    require(validation['feature']['computes_returns'] is False and
            validation['feature']['new_model_fits'] == 0, 'Independent feature scope differs')
    for flag in ('independent_scalar_rebuild', 'negative_anchor_selection_checked', 'pre_event_risk_checked',
                 'matured_payoffs_checked', 'matched_inverse_risk_mass_checked',
                 'full253_price_support_checked', 'official_session_windows_checked',
                 'monthend_queries_checked', 'exact_common_availability', 'audit_checked',
                 'no_future_or_outcome_inputs', 'centered_high_tie_ranks_checked'):
        require(validation['feature'].get(flag) is True, 'Missing independent risk/payoff proof: '+flag)
    for flag in ('full_candidate_universe', 'initial_delta_zero', 'common_available_only',
                 'no_eligibility_or_model_filter', 'centered_cross_sectional_ranking',
                 'direct_additive_pre_score', 'fixed_linear_mapping_checked'):
        require(validation['delta'].get(flag) is True, 'Missing independent delta proof: '+flag)

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
    receipt = dict(schema='monthly_event_risk_payoff198_research_review_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_refs=evidence.pins,
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED', selection=selection,
        failed_training_guards=[key for key, passed in selection['guards'].items() if not passed],
        best_retained=closure['best_training_selected_candidate'], historical_best_unchanged=retained,
        exact_incumbent_metrics_equal_predecessor=not fee_total_differences,
        exact_incumbent_metrics_except_disclosed_fee_totals=True,
        incumbent_fee_total_differences=fee_total_differences, incumbent_export_diagnostic=fee_diagnostic,
        new_candidate_summary=summaries,
        holding_changes=changes,
        event_risk_payoff_support=dict(all_cells=5978, source_available_cells=source_count, available_cells=available,
            unavailable_cells=5978-available, **series, **counts,
            source_unavailable=[x for x in cells if not x['source_available']],
            source_valid_unavailable=[dict(signal_index=x['signal_index'], symbol=x['symbol'], date=x['date'],
                reason=x['reason'], selected_anchors=x['selected_anchors'],
                zero_sigma_anchor_indices=[a['anchor_index'] for a in x['anchor_events'] if a['sigma'] == 0])
                for x in cells if x['source_available'] and not x['available']],
            reason_counts=dict(Counter(x['reason'] for x in cells)),
            exact_equivalence_branch_counts=dict(Counter(x['identity_branch'] or 'general_formula'
                                                        for x in cells if x['available'])),
            signed_q_and_c_formulas_checked=True, no_q_less_than_c_restriction=True,
            pre_loss_63_simple_risk_checked=True, maturity_within_skip42_cutoff_checked=True,
            common_greedy21_anchor_spacing_checked=True, zero_risk_whole_cell_missing_checked=True,
            numerical_comparison=NUMERICAL_COMPARISON,
            full253_source_and_greedy_selection_check='Accepted from the hash-bound independent preparation receipt; this reader checks recorded strict-loss anchors, minimum spacing, risk endpoints/returns, sigma, mature payoff and Q/C formulas without rereading raw panels. It does not independently prove the absence of an unrecorded earlier negative day.'),
        validation=dict(synthetic_tests=tests['tests'], independent_cells=5978, independent_execution_paths=9,
            independent_period_slices=checked_slices, training_metric_groups=checked_training,
            order_prefixes=123, source_prefixes=[0, 1, 53, 88, 122], protected_files=21444,
            central_completed_entries=196, central_trials=1105,
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
