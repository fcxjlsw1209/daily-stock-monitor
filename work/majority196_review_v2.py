"""Version2 write-once review of closed196; no price, model, or replay imports.

Run only after the root has confirmed196 closure. This reader verifies existing
hash-bound artifacts and summarizes recorded results; it creates no new factor,
selection, order, NAV series, or parameter search.
The original reader remains unchanged. Only the known next-open-cost fees_paid
total may differ by <=1e-9; all other recorded incumbent metrics remain exact.
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
OUT = ROOT / 'outputs/monthly_pairwise_majority196'
PRIOR = ROOT / 'outputs/monthly_volume_cluster195'
ARMS = ('incumbent_local67', 'tournament_majority', 'tournament_balance_control')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
BOUNDS = {'full122': (0, 122), 'first88': (0, 88), 'early53': (0, 53),
          'recent35': (53, 88), 'later34': (88, 122)}
METRICS = ('start', 'end', 'cagr', 'sharpe', 'max_drawdown')
COUNTS = ('valid_price_observations', 'block_comparisons', 'pair_majority_decisions')
FAILED_READER = ROOT/'work/majority196_review.py'
FAILED_READER_SHA = 'edfce347bd4eee4c82235023a5e1b93870a9f17b40628ba4a8a2f431c7dc2b3f'
ENGINE = ROOT/'outputs/monthly_factor_search_v2/validate.py'
ENGINE_SHA = 'b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4'
FEE_TOTAL_ABSOLUTE_TOLERANCE = 1e-9


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
        differences.append(dict(path=path, field=location, previous195=prior, current196=current,
            signed_difference=current-prior, absolute_difference=abs(current-prior),
            absolute_tolerance=FEE_TOTAL_ABSOLUTE_TOLERANCE))


def diagnose_incumbent_exports(evidence, prior_manifest, current_results, prior_results):
    """Inspect recorded fills/equity and fee summations, never run an engine."""
    evidence.verify(FAILED_READER, FAILED_READER_SHA)
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
                    previous195=old['fee'], current196=new['fee'], signed_difference=new['fee']-old['fee']))
        totals = {}
        for label, holds, result in (('previous195', old_holds, prior_results['arms'][ARMS[0]][path]),
                                      ('current196', new_holds, current_results['arms'][ARMS[0]][path])):
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
        source_snippets=snippets, failed_reader_path=str(FAILED_READER.relative_to(ROOT)),
        failed_reader_sha256=FAILED_READER_SHA, paths=paths,
        explanation='The frozen engine forms the turnover domain with set(old)|set(chosen), sums its terms in set iteration order, computes a separate rebalance fee, and accumulates those fees chronologically. Fill exports are sorted by symbol and their costs are computed from exported unit changes. The recorded period-fee differences reproduce the fee-total residual while every exported fill, daily equity value and reconciled position remains exactly equal. This is consistent with floating summation order, not an economic or strategy difference; historical process hash seeds are unavailable, so their specific iteration orders are not claimed to have been recovered.',
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
    print(json.dumps(dict(synthetic_checks=5, passed=True, actual_artifacts_read=False)))


def review(completion_sha, check_only=False):
    require(not (OUT/'RESEARCH_REVIEW.json').exists(), 'Review is write-once')
    evidence = Evidence()
    closure = evidence.load(OUT/'COMPLETION.json', completion_sha)
    require(closure['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED', '196 is not closed')
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
    tests_path = ROOT/'work/monthly_pairwise_majority196_TESTS.json'
    tests = evidence.load(tests_path, plan['input_sha256'][str(tests_path)])
    require(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests'] == 57,
            'Expected57 successful synthetic tests')
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

    sizes = feature['source_available_by_event']
    require(len(sizes) == 122 and all(type(n) is int and 0 <= n <= 49 for n in sizes), 'Invalid source counts')
    require(sum(sizes) == feature['source_available_cells'] == 5973 and
            feature['checked_cells'] == 5978 and feature['symbols_per_event'] == 49 and
            feature['price_count'] == 253 and feature['block_count'] == 12, 'Unexpected196 source coverage')
    available = sum(n if n >= 2 else 0 for n in sizes)
    pairs = sum(n*(n-1)//2 for n in sizes)
    counts = dict(valid_price_observations=253*sum(sizes), block_comparisons=12*pairs,
                  pair_majority_decisions=pairs)
    require(feature['available_cells'] == available and feature['unavailable_cells'] == 5978-available,
            'Common feature coverage differs')
    for record in (feature, validation['feature'], validation, prepared, closure):
        require(all(record[key] == value for key, value in counts.items()), 'Pair accounting differs')
    require(validation['feature']['source_available_by_event'] == sizes and
            validation['feature']['source_available_cells'] == 5973, 'Independent source domains differ')
    require(len(feature['event_history']) == 122, 'Missing event audits')
    cells = []
    for i, (event, n) in enumerate(zip(feature['event_history'], sizes)):
        require(event['date'] == meta['dates'][i] and len(event['cells']) == 49 and
                event['domain_count'] == n and event['source_available_count'] == n and
                len(event['source_indices']) == n and len(event['pairs']) == n*(n-1)//2,
                'Event source/pair accounting differs')
        require(sum(x['source_available'] for x in event['cells']) == n and
                sum(x['available'] for x in event['cells']) == (n if n >= 2 else 0), 'Cell support differs')
        cells.extend(dict(x, signal_index=i) for x in event['cells'])

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
    require(len(fee_total_differences) == 2 and
            {x['field'] for x in fee_total_differences} == {'metrics/fees_paid', 'slices/full122/fees_paid'},
            'Observed fee-total exception differs from the diagnosed two keys')
    receipt = dict(schema='monthly_pairwise_majority196_research_review_v2',
        created_utc=datetime.now(timezone.utc).isoformat(), source_refs=evidence.pins,
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED', selection=selection,
        failed_training_guards=[key for key, passed in selection['guards'].items() if not passed],
        best_retained=closure['best_training_selected_candidate'], historical_best_unchanged=retained,
        exact_incumbent_metrics_equal_predecessor=False,
        exact_incumbent_metrics_except_disclosed_fee_totals=True,
        incumbent_fee_total_differences=fee_total_differences, incumbent_export_diagnostic=fee_diagnostic,
        new_candidate_summary=summaries,
        holding_changes=changes,
        pairwise_support=dict(all_cells=5978, source_available_cells=5973, available_cells=available,
            unavailable_cells=5978-available, source_available_by_event=sizes, **counts,
            source_unavailable=[x for x in cells if not x['source_available']],
            reason_counts=dict(Counter(x['reason'] for x in cells))),
        validation=dict(synthetic_tests=57, independent_cells=5978, independent_execution_paths=9,
            independent_period_slices=checked_slices, training_metric_groups=checked_training,
            order_prefixes=123, source_prefixes=[0, 1, 53, 88, 122], protected_files=21444,
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
