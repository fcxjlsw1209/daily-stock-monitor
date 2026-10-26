"""Write-once review of completed196 evidence; no price, model, or replay imports.

Run only after the root has confirmed196 closure. This reader verifies existing
hash-bound artifacts and summarizes recorded results; it creates no new factor,
selection, order, NAV series, or parameter search.
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


def review(completion_sha):
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
                require(result['metrics'] == prior['metrics'] and result['slices'] == prior['slices'],
                        'IncumbentLocal67 metrics changed: '+path)
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
    receipt = dict(schema='monthly_pairwise_majority196_research_review_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), source_refs=evidence.pins,
        result='ADMITTED' if selection['admitted'] else 'NOT_ADOPTED', selection=selection,
        failed_training_guards=[key for key, passed in selection['guards'].items() if not passed],
        best_retained=closure['best_training_selected_candidate'], historical_best_unchanged=retained,
        exact_incumbent_metrics_equal_predecessor=True, new_candidate_summary=summaries,
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
    parser.add_argument('--completion-sha256', required=True)
    review(parser.parse_args().completion_sha256)
