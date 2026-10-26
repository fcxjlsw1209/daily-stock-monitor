"""Independent scalar oracle for the fixed eligible-core top-six architecture.

Imports no producer or strategy. Only caller-supplied B/C/cw/E and metadata are
consumed. Original missing-factor neutral scores are values, not new exclusions.
The accounting helper is the immutable, separately implemented169 auditor.
"""
from datetime import date
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'leader6_gate', 'baseline_score')
NEW_ARMS = ('leader6_gate', 'baseline_score')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
COUNT_FIELDS = ('query_months', 'eligible_cells', 'shortlisted_cells', 'gated_months',
    'tie_expansion_months', 'tie_added_cells', 'changed_holding_months',
    'changed_ordered_months', 'cash_months', 'two_holding_months', 'three_holding_months')
ZERO_FIELDS = ('new_model_fits', 'predictive_model_fits', 'new_ridge_fits',
               'new_tree_fits', 'new_parameters')
PREP_FLAGS = ('passed', 'independent_scalar_rebuild', 'original_eligibility_preserved',
    'eligible_only_core_division', 'top6_cutoff_checked', 'exact_boundary_ties_included',
    'baseline_scores_unchanged', 'lexical_ties_checked', 'two_three_cash_checked',
    'no_industry_filter', 'initial_partial_same_rule', 'no_labels_or_models_consumed',
    'prefix_only_inputs', 'audit_checked', 'zero_fits_and_parameters_checked')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('leader_gate218_accounting', ACCOUNTING)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def close(actual, expected, label, atol=0.):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), label + ': finite')
    require(np.all(np.abs(actual - expected) <= atol), label + ': value')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, e) in enumerate(zip(actual, expected)):
            record_check(a, e, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact type/value')
        if isinstance(expected, float) and expected == 0. and '/scores/' in label:
            require(math.copysign(1., actual) == math.copysign(1., expected),
                    label + ': copied signed zero')


def iso(value, label):
    require(isinstance(value, str) and len(value) == 10, label + ': exact ISO day')
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(label + ': invalid calendar day') from exc
    require(parsed.isoformat() == value, label + ': canonical day')
    return parsed


def validated(arrays, meta, n):
    require(isinstance(arrays, dict) and isinstance(meta, dict), 'Explicit dictionaries required')
    require(all(k in arrays for k in ('baseline', 'core', 'core_weight', 'eligible')),
            'B/C/cw/E required')
    require(isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_)),
            'Nonboolean integral prefix required')
    baseline, core, weight, eligible = [np.asarray(arrays[k]) for k in
        ('baseline', 'core', 'core_weight', 'eligible')]
    require(baseline.ndim == 2 and baseline.shape[1] > 0, 'Nonempty candidate width')
    months, width = baseline.shape
    require(core.shape == eligible.shape == baseline.shape, 'B/C/E axes')
    require(weight.shape == (months, 1), 'Core weight must have one cell per month')
    require(eligible.dtype == np.dtype(bool), 'Exact boolean eligibility')
    for key, value in (('baseline', baseline), ('core', core), ('core_weight', weight)):
        require(value.dtype.kind in 'iuf', key + ': real numeric dtype required')
    require(0 <= n <= months, 'Prefix outside source axis')
    require(all(k in meta and isinstance(meta[k], list) and len(meta[k]) == months
                for k in ('dates', 'ends', 'symbols')), 'Metadata axes')
    n = int(n)
    # Entire requested score prefix is finite, including currently ineligible cells.
    # Unrequested future contents are never validated or used.
    for value in (baseline, core, weight):
        require(np.isfinite(value[:n]).all(), 'Nonfinite requested cached score prefix')
    require((weight[:n] > 0).all(), 'Nonpositive core weight')
    previous = None
    formal = set()
    for i in range(n):
        day, end = iso(meta['dates'][i], 'date'), iso(meta['ends'][i], 'end')
        initial = meta['dates'][i] == '2016-08-17'
        require(not initial or (i == 0 and meta['ends'][i] == '2016-08-31'), 'Initial partial identity')
        require(end > day, 'Holding end after signal')
        month = day.year * 12 + day.month
        require(initial or end.year * 12 + end.month == month + 1, 'Next natural-month end')
        require(previous is None or day > previous, 'Strictly increasing query dates')
        if not initial:
            require(month not in formal, 'One formal signal per natural month')
            formal.add(month)
        names = meta['symbols'][i]
        require(isinstance(names, list) and len(names) == width and
                all(isinstance(s, str) and s and s.strip() == s for s in names) and
                len(set(names)) == width, 'Unique exact security identities')
        previous = day
    return baseline, core, weight, eligible, n, width


def reconstruct(arrays, meta, n):
    baseline, core, weight, eligible, n, width = validated(arrays, meta, n)
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts['query_months'] = n
    orders = {arm: [] for arm in NEW_ARMS}
    shortlist = np.zeros((n, width), dtype=bool)
    records = []
    for i in range(n):
        names = meta['symbols'][i]
        members = [j for j in range(width) if bool(eligible[i, j])]
        scores = {j: float(baseline[i, j]) for j in members}
        ratio = {}
        for j in members:
            ratio[j] = float(core[i, j]) / float(weight[i, 0])
            if not math.isfinite(ratio[j]):
                raise FloatingPointError('Nonfinite eligible C/cw')
        total = len(members)
        cutoff, tie_count = None, 0
        if total > 6:
            # Rank position found by pair counts, independently of producer sorting.
            distinct = set(ratio.values())
            cutoff = next(value for value in distinct
                          if sum(v > value for v in ratio.values()) < 6 <=
                          sum(v >= value for v in ratio.values()))
            kept = [j for j in members if ratio[j] >= cutoff]
            tie_count = sum(ratio[j] == cutoff for j in members)
        else:
            kept = list(members)
        for j in kept:
            shortlist[i, j] = True
        selections = {}
        for arm, use in (('leader6_gate', kept), ('baseline_score', members)):
            # Full ordinal pair count including exact-symbol lexical tie breaking.
            ordered = sorted(use, key=lambda j: sum(
                scores[k] > scores[j] or (scores[k] == scores[j] and names[k] < names[j])
                for k in use))
            selected = ordered[:3] if len(ordered) >= 2 else []
            selected_symbols = [names[j] for j in selected]
            selections[arm] = selected_symbols
            orders[arm].append(dict(date=meta['dates'][i], end=meta['ends'][i],
                signal_date=meta['dates'][i], symbols=selected_symbols,
                scores=[scores[j] for j in selected]))
        changed = set(selections['leader6_gate']) != set(selections['baseline_score'])
        changed_order = selections['leader6_gate'] != selections['baseline_score']
        added = max(len(kept) - 6, 0)
        counts['eligible_cells'] += total
        counts['shortlisted_cells'] += len(kept)
        counts['gated_months'] += int(total > 6)
        counts['tie_expansion_months'] += int(added > 0)
        counts['tie_added_cells'] += added
        counts['changed_holding_months'] += int(changed)
        counts['changed_ordered_months'] += int(changed_order)
        counts['cash_months'] += int(total < 2)
        counts['two_holding_months'] += int(total == 2)
        counts['three_holding_months'] += int(total >= 3)
        records.append(dict(row=i, date=meta['dates'][i], eligible_count=total,
            shortlist_count=len(kept), sixth_core_threshold=cutoff, sixth_tie_count=int(tie_count),
            tie_added_count=added, leader6_symbols=selections['leader6_gate'],
            baseline_symbols=selections['baseline_score'], changed_holdings=changed,
            changed_order=changed_order))
    audit = dict(method='eligible_core_top6_then_baseline_top3', counts=counts,
                 records=records, **{key: 0 for key in ZERO_FIELDS})
    return dict(orders=orders, shortlist=shortlist, audit=audit)


def audit_preparation(arrays, meta, n, result):
    expected = reconstruct(arrays, meta, n)
    require(isinstance(result, dict) and set(result) == {'orders', 'shortlist', 'audit'},
            'Preparation result schema')
    record_check(result['orders'], expected['orders'], 'orders')
    mask = np.asarray(result['shortlist'])
    require(mask.dtype == np.dtype(bool) and mask.shape == expected['shortlist'].shape and
            np.array_equal(mask, expected['shortlist']), 'Exact boolean shortlist')
    record_check(result['audit'], expected['audit'])
    return dict(**{key: True for key in PREP_FLAGS}, counts=expected['audit']['counts'],
                orders_checked={arm: int(n) for arm in NEW_ARMS},
                computes_returns=False, producer_modules_imported=False,
                **{key: 0 for key in ZERO_FIELDS})


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = accounting()
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arm names differ')
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['ends'][:-1] == meta['dates'][1:], 'Outcome interval axis differs')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and meta['ends'][52] == meta['dates'][53] == '2020-12-31'
            and meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Outcome fixed slices differ')
    reports = {}
    for arm in ARMS:
        require(len(orders[arm]) == 122 and [o['date'] for o in orders[arm]] == meta['dates'] and
                [o['end'] for o in orders[arm]] == meta['ends'], 'Order axis differs')
        require(set(exports[arm]) == set(results[arm]) == {p[0] for p in PATHS}, 'Execution paths differ')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            data = exports[arm][name]
            require(set(data) == {'DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS'}, 'Incomplete execution exports')
            audited = helper.audit_path(px, opening, orders[arm], *[data[k] for k in
                ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')], execution, cost)
            slices = helper.statistics_by_split(audited['days'], audited['values'], meta)
            wanted = results[arm][name]
            require(set(wanted['slices']) == set(slices), 'Fixed five slices differ')
            helper.check_metrics(slices['full122'], wanted['metrics'], arm + '/' + name)
            for split, value in slices.items():
                helper.check_metrics(value, wanted['slices'][split], arm + '/' + name + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                helper.near(audited[key], wanted['metrics'][key], key)
            annual = []
            for year in sorted({d[:4] for d in audited['days']}):
                ids = [i for i, day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                prior = audited['values'][ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=audited['values'][ids[-1]] / prior - 1.,
                                   partial=int(year) in (2016, 2026)))
            observed = wanted['metrics']['annual_returns']
            require(isinstance(observed, list) and len(observed) == len(annual), 'Annual return count')
            for actual, expected in zip(observed, annual):
                require(set(actual) == set(expected) and actual['year'] == expected['year'] and
                        actual['partial'] == expected['partial'], 'Annual return identity')
                close(actual['return_'], expected['return_'], 'Annual return', atol=1e-12)
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
