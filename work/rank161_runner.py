"""Fixed three-arm rank trend wiring; the registration host owns all file I/O.

These functions consume supplied arrays only. The two kernel outputs are
already score deltas, not ranks requiring another dose multiplication.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_annual_lag158_runner as previous

TRAIN, MONTHS, WIDTH = 88, 122, 49
NAMES = ('incumbent_local67', 'rank_residence_control', 'rank_trend_challenge')
HANDS = EXTRA = ('delta_residence', 'delta_rank_trend')
MAX_ABS_DELTA = .01
INITIAL_PARTIAL = '2016-08-17'
OUT = 'outputs/monthly_rank161'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json',
               'PRICE_PARITY.json', 'FEATURE_AUDIT.json', 'STATE_VALUES.json', 'SUPPORT_AUDIT.json')
require = base.require


def measurement_contract():
    return dict(
        arms=list(NAMES), history_months=6, momentum_offsets=[-252, -42],
        warmup_sessions=295, rank='average ascending percentile minus .5 among original historical pool finite positive endpoints',
        observation_gate='positive momentum and 295 exact-session finite positive prices ending at each snapshot',
        history='six exact calendar-month-end snapshots including current month; historical membership for each snapshot',
        missing='both additions exactly zero unless the same security has six usable observations; no fill or older substitutes',
        residence_delta='0.02*clip(mean(r)-population_std(r),-0.5,0.5)',
        trend_delta='0.01*sum((k-2.5)*r[k],k=0..5)/4.5; clip only roundoff to [-0.01,0.01]',
        max_abs_delta=MAX_ABS_DELTA, score_inputs='both deltas already include the complete dose',
        control_matching='same common support and maximum absolute dose; score distributions are not matched',
        population='original historical pools; current trading eligibility and all original27 arrays remain unchanged',
        prices='Backend.load close panel for rank history and execution parity only; no outcome labels',
        placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',
        original_arrays=27, extra_arrays=list(EXTRA), initial_partial='exact_incumbent; never supplies later monthly history',
        zero_delta='exact_incumbent', core='unchanged_long_momentum_skip42',
        portfolio='original_EW3_two_half_less_than_two_cash; fixed_intramonth_units',
        strict_pit_vintage=False, strict_pit_membership=False,
        historical_series='inherited price vintages and estimated Top32 plus assumed theme17 historical membership proxy',
        new_predictive_model_fits=0, nuisance_estimates=0)


def selection_contract():
    return dict(
        selection_count=1, first88_only=True, early53=True, recent35=True,
        later34_diagnostic_only=True, prefix_reselection=False,
        source_support='all structural and missing-neutral validation passes; coverage is reported without a threshold',
        minimum_common_support_cells=None, initial_partial_exempt=True,
        minimum_close_cagr_gain=.005, cagr_epsilon=1e-12, open_cost_noninferior=True,
        sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
        guards_apply_to='same_close and next_open_5bp; only same_close requires .005 CAGR gain',
        minimum_changed_held_set_months=3, minimum_changed_each_era=1,
        choice='highest_same_close_first88_CAGR; ties_within1e-12_residence_before_trend',
        execution_paths=9)


def integration_contract():
    return dict(
        host_owns='registration, hash verification, source gate, write-once caches and phase receipts',
        prepare_order=['verify_PLAN_code_runtime_and_source_pins',
                       'load_original27_and_verify_price_parity',
                       'kernel_features_all123_prefixes_and_report_common_support_before_orders',
                       'panels_all123_prefixes',
                       'publish_cache_then_independent_prepare_before_returns'],
        select_order=['reverify_PLAN_cache_and_independent_prepare',
                      'Backend.training_once_first88', 'changed_support_then_select_first88',
                      'freeze_all_three_order_streams'],
        diagnose_order=['reverify_frozen_selection_and_all_pins',
                        'Backend.diagnostics_fixed_nine_paths',
                        'independent_outcome_and_protected_archive_before_completion'],
        required_receipt_counts=dict(candidate_cells=5978, orders=366, prefixes=123,
                                     feature_matrices=2, paths=9),
        source_support_api='source_support(audit, meta, deltas)',
        changed_support_api='changed_support(orders, arm_name)')


def validated_deltas(arrays, meta, n):
    shape = np.asarray(arrays['baseline']).shape
    require(len(shape) == 2 and shape[1] == WIDTH and type(n) is int and 0 <= n <= shape[0],
            'Valid original shape and prefix required')
    require(len(meta['dates']) == len(meta['symbols']) == shape[0], 'Exact full original axes required')
    eligible = np.asarray(arrays['eligible'])
    require(eligible.shape == shape and eligible.dtype == bool, 'Original boolean eligibility required')
    for i, names in enumerate(meta['symbols'][:n]):
        require(len(names) == len(set(names)) == WIDTH and all(isinstance(s, str) and s for s in names),
                '49 unique original candidates required')
    result = {}
    for key in HANDS:
        raw = np.asarray(arrays[key])
        require(raw.shape == shape, 'Delta shape must match the original panel')
        part = raw[:n]
        require(all(isinstance(x, (int, float, np.integer, np.floating)) and
                    not isinstance(x, (bool, np.bool_)) for x in part.flat), 'Numeric score delta required')
        delta = part.astype(float, copy=True)
        require(np.isfinite(delta).all() and np.all(np.abs(delta) <= MAX_ABS_DELTA),
                'Finite score delta within the fixed one-percent bound required')
        for i, day in enumerate(meta['dates'][:n]):
            if day == INITIAL_PARTIAL:
                require(np.all(delta[i] == 0), 'Initial partial must be exactly neutral')
        result[key] = delta
    return result


def choices_all(arrays, meta, n, backend, policy):
    """Apply the supplied delta once; run the frozen episode once per arm."""
    require(all(policy.get(k) == v for k, v in dict(budget=.05, ceiling=5, gap=.02).items()),
            'Original episode policy required')
    deltas = validated_deltas(arrays, meta, n)
    original, original_scores = backend.baseline(arrays, meta, n, policy)
    pre_original = np.asarray(backend.pre(arrays, n))
    shape = np.asarray(arrays['baseline']).shape
    require(pre_original.shape == shape, 'Full original Local67 config4 pre-score required')
    choices = {NAMES[0]: original}
    scores = {NAMES[0]: original_scores}
    for name, key in zip(NAMES[1:], HANDS):
        delta = deltas[key]
        changed = delta != 0.
        if not changed.any():
            choices[name] = deepcopy(original)
            scores[name] = original_scores.copy()
            continue
        pre = pre_original.copy()
        view = pre[:n].copy()
        view[changed] += delta[changed]
        pre[:n] = view
        chosen, score = backend.adapter.score_and_choose(
            'incumbent66', 0., np.zeros(shape), arrays, meta, n, pre_score=pre,
            effects=arrays['effects'], episode_policy=policy, boundary_selector=backend.selector)
        for i in np.flatnonzero(~changed.any(axis=1)):
            require(chosen[i] == original[i] and score[i].tobytes() == original_scores[i].tobytes(),
                    'Exact neutral-row incumbent parity required')
        choices[name], scores[name] = chosen, score
    audit = [dict(signal_index=i, date=meta['dates'][i], initial_partial=meta['dates'][i] == INITIAL_PARTIAL,
                  nonzero_deltas={name: bool(np.any(deltas[key][i])) for name, key in zip(NAMES[1:], HANDS)})
             for i in range(n)]
    return choices, scores, audit


def panels(arrays, meta, backend, policy):
    require(len(meta['dates']) == MONTHS, 'Full122 pipeline required')
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    choices, scores, audit = choices_all(arrays, meta, MONTHS, backend, policy)
    for n in range(MONTHS + 1):
        cc, ss, _ = choices_all(arrays, meta, n, backend, policy)
        require(all(cc[k] == choices[k][:n] and ss[k].tobytes() == scores[k][:n].tobytes() for k in NAMES),
                'Prefix choices/scores changed')
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays mutated')
    return {k: backend.make_orders(choices[k], scores[k], meta) for k in NAMES}, scores, audit


def source_support(audit, meta, deltas=None):
    """Check structure/common support; report coverage without outcome thresholds."""
    require(len(meta['dates']) == len(meta['symbols']) == MONTHS and meta['dates'][0] == INITIAL_PARTIAL and
            meta['dates'].count(INITIAL_PARTIAL) == 1, 'Original122 dates with one initial partial required')
    require(audit.get('events') == MONTHS and audit.get('symbols_per_event') == WIDTH and
            audit.get('months') == audit.get('minimum_observations') == 6,
            'Exact122x49 and six-observation audit required')
    require(audit.get('trading_eligibility_modified') is False and audit.get('outcomes_used') is False,
            'Rank features must not alter eligibility or use outcomes')
    rows = audit.get('event_history')
    require(isinstance(rows, list) and len(rows) == MONTHS, 'All122 historical source records required')
    if deltas is not None:
        require(set(deltas) == set(EXTRA), 'Exact two score delta matrices required')
        for key in EXTRA:
            a = np.asarray(deltas[key])
            require(a.shape == (MONTHS, WIDTH) and a.dtype == np.float64 and np.isfinite(a).all() and
                    np.all(np.abs(a) <= MAX_ABS_DELTA), 'Finite bounded float64 deltas required')
    masks, histogram = [], {}
    for i, row in enumerate(rows):
        require(isinstance(row, dict) and row.get('date') == meta['dates'][i] and
                row.get('symbols') == meta['symbols'][i] and len(meta['symbols'][i]) == WIDTH and
                len(set(meta['symbols'][i])) == WIDTH, 'Source monthly axes/order changed')
        initial = i == 0
        require(type(row.get('initial_forced_neutral')) is bool and row['initial_forced_neutral'] == initial,
                'Explicit original initial-partial status required')
        month = pd.Period(row['date'], freq='M')
        months = [str(x) for x in pd.period_range(month - 5, month, freq='M')]
        require(row.get('calendar_months') == months, 'Exactly six consecutive calendar months required')
        snapshots = row.get('snapshots')
        require(isinstance(snapshots, list) and len(snapshots) == 6, 'Six snapshot slots required')
        for j, value in enumerate(snapshots):
            require(value is None or (isinstance(value, str) and value[:7] == months[j] and
                    value <= row['date'] and (initial or value != INITIAL_PARTIAL)), 'Past-only exact-month snapshot required')
        require(row.get('missing_snapshot_months') == [m for m, x in zip(months, snapshots) if x is None],
                'Missing snapshot months must be explicit')
        raw_mask, counts, raw_support = row.get('usable_mask'), row.get('usable_observations'), row.get('common_support')
        require(isinstance(raw_mask, list) and len(raw_mask) == 6 and
                all(isinstance(x, list) and len(x) == WIDTH and all(type(v) is bool for v in x) for x in raw_mask),
                'Exact boolean observation mask required')
        mask = np.asarray(raw_mask, dtype=bool)
        require(isinstance(counts, list) and len(counts) == WIDTH and all(type(x) is int for x in counts) and
                counts == mask.sum(axis=0).tolist(), 'Exact usable-observation counts required')
        require(isinstance(raw_support, list) and len(raw_support) == WIDTH and all(type(x) is bool for x in raw_support),
                'Exact boolean common support required')
        support = np.asarray(raw_support, dtype=bool)
        require(np.array_equal(support, (mask.sum(axis=0) == 6) & (not initial)), 'Both additions require all six observations')
        trajectory = row.get('rank_trajectories')
        require(isinstance(trajectory, list) and len(trajectory) == 6 and
                all(isinstance(x, list) and len(x) == WIDTH for x in trajectory), 'Six by49 usable rank trajectory required')
        for j, values in enumerate(trajectory):
            require(snapshots[j] is not None or not mask[j].any(), 'Missing snapshot cannot supply usable ranks')
            for k, value in enumerate(values):
                require((value is None and not mask[j, k]) or
                        (type(value) in (int, float) and math.isfinite(value) and -.5 <= value <= .5 and mask[j, k]),
                        'Usable ranks must match the common observation mask')
        if deltas is not None:
            require(all(np.all(np.asarray(deltas[k])[i, ~support] == 0.) for k in EXTRA),
                    'Missing support and initial rows must have exact zero deltas')
        for count in counts:
            histogram[str(count)] = histogram.get(str(count), 0) + 1
        masks.append(support)
    common = np.asarray(masks)
    supported = int(common.sum())
    require(audit.get('common_support_cells') == supported and audit.get('neutral_cells') == MONTHS*WIDTH-supported and
            audit.get('observation_count_histogram') == histogram, 'Audit aggregate counts must match row evidence')
    coverage = {}
    for name, lo, hi in (('full122', 0, MONTHS), ('first88', 0, TRAIN), ('early53', 0, 53),
                         ('recent35', 53, TRAIN), ('later34', TRAIN, MONTHS)):
        total, count = (hi-lo)*WIDTH, int(common[lo:hi].sum())
        coverage[name] = dict(candidate_cells=total, common_support_cells=count,
                              neutral_cells=total-count, coverage=count/total)
    return dict(passed=True, candidate_cells=MONTHS*WIDTH, common_support_cells=supported,
                neutral_cells=MONTHS*WIDTH-supported, initial_partial_exempt=True,
                coverage_threshold=None, coverage=coverage,
                observation_count_histogram=histogram)


def changed_support(orders, name):
    """Count distinct first88 holding sets, not ordering or source identities."""
    require(name in NAMES[1:] and all(len(orders[k]) >= TRAIN for k in (NAMES[0], name)),
            'First88 baseline and challenger orders required')
    months = []
    for i in range(TRAIN):
        sets = []
        for key in (NAMES[0], name):
            symbols = orders[key][i]['symbols']
            require(isinstance(symbols, list) and len(symbols) == len(set(symbols)) and
                    all(isinstance(s, str) and s for s in symbols), 'Unique held symbols required')
            sets.append(set(symbols))
        if sets[0] != sets[1]:
            months.append(i)
    early = sum(i < 53 for i in months)
    recent = len(months) - early
    return dict(passed=len(months) >= 3 and early >= 1 and recent >= 1,
                months=months, early_months=early, recent_months=recent,
                rule='first88 unordered held-set changes; later34 excluded')


def metric_guard(candidate, incumbent, minimum_gain):
    """Same fixed finite guards as160; no outcome-dependent thresholds."""
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    values = [r.get(a, {}).get(b) for r in (candidate, incumbent) for a, b in fields]
    if not all(type(x) in (int, float) and math.isfinite(x) for x in values):
        return False
    x, b = values[:5], values[5:]
    return x[0] >= b[0] + minimum_gain - 1e-12 and all(
        v >= w - tolerance for v, w, tolerance in zip(x[1:], b[1:], (.05, .03, .025, .025)))


def select_first88(rows, opened, support, changed):
    require(all([r['name'] for r in group] == list(NAMES) for group in (rows, opened)), 'Fixed three-arm order required')
    require(type(support.get('passed')) is bool and set(changed) == set(NAMES[1:]),
            'Source and changed-support receipts required')
    eligible, guards = [], {}
    for i, name in enumerate(NAMES[1:], 1):
        guard = dict(source_support=support['passed'], changed_holdings=changed[name].get('passed') is True,
                     close=metric_guard(rows[i], rows[0], .005), cost_open=metric_guard(opened[i], opened[0], 0.))
        guards[name] = guard
        if all(guard.values()):
            eligible.append(i)
    winner = 0
    for i in eligible:
        if winner == 0 or rows[i]['training']['cagr'] > rows[winner]['training']['cagr'] + 1e-12:
            winner = i
    return dict(primary=NAMES[winner], admitted=winner != 0, selection_count=1,
                guards=guards, later34_used_for_selection=False)


class Backend(base.Local67Backend):
    price_parity = previous.Backend.price_parity

    def training(self, orders, meta, px, op):
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        rows, opened = [], []
        for name in NAMES:
            indices = [[meta['symbols'][i].index(s) for s in row['symbols']]
                       for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(indices, blocks, valid)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
            _, eq, _, _ = self.local.ex.e.engine.replay(
                px, op, orders[name][:TRAIN], cost=.0005, execution='open',
                start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][87]))
            item = {'name': name}
            for key, lo, hi in (('training', 0, 87), ('early', 0, 52), ('recent', 53, 87)):
                start, end = meta['dates'][lo], meta['ends'][hi]
                part = eq.loc[start:end]
                require(len(part) >= 2, 'Training interval too short')
                item[key] = rp.stats_rr(part.to_numpy()[1:] / part.to_numpy()[:-1] - 1, start, end)
            opened.append(item)
        return rows, opened

    def diagnostics(self, orders, meta, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'),
                                    ('open', .0005, 'next_open_cost')):
                met, eq, trades, holds = self.local.ex.e.engine.replay(
                    px, op, orders[name], cost=cost, execution=mode,
                    start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][-1]))
                for year in met['annual_returns']:
                    if year['year'] in (2016, 2026):
                        year['partial'] = True
                results[name][key] = dict(metrics=met, slices=self.local.arch.old.prior.old.previous.cuts(eq, meta))
                writer(name + '_' + key + '_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name + '_' + key + '_TRADES.json', trades)
                writer(name + '_' + key + '_HOLDINGS.json', holds)
        return results


def source_files(root):
    import basic_shares159_runner_draft as shared
    root = Path(root)
    paths = [Path(__file__), root/'work/test_rank161_runner.py',
             root/'work/rank161_kernel.py', root/'work/test_rank161_kernel.py',
             root/'work/basic_shares159_host.py']
    closure = dict(previous.prior.source_files(root))
    closure.update(shared.source_files(root))
    closure.update({str(p.resolve()): base.sha(p) for p in paths})
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contracts', action='store_true')
    args = parser.parse_args()
    if args.contracts:
        print(json.dumps(dict(measurement=measurement_contract(), selection=selection_contract(),
                             integration=integration_contract()), indent=2))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
