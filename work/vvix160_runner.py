"""Fixed three-arm VVIX wiring; the registration host owns all file I/O.

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
NAMES = ('incumbent_local67', 'vix_control', 'vvix_challenge')
HANDS = EXTRA = ('delta_vix', 'delta_vvix')
MAX_ABS_DELTA = .01
INITIAL_PARTIAL = '2016-08-17'
OUT = 'outputs/monthly_vvix160'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json',
               'PRICE_PARITY.json', 'FEATURE_AUDIT.json', 'STATE_VALUES.json', 'SUPPORT_AUDIT.json')
require = base.require


def measurement_contract():
    return dict(
        arms=list(NAMES), sources=['VIX', 'VVIX'], window_sessions=252,
        state='2*(average_ascending_rank_of_last_value-1)/(252-1)-1; all_equal_zero',
        state_cutoff='exact_previous_official_session; 252 consecutive official sessions ending there',
        missing='either source missing or invalid in its exact window blocks both; no stale fill or skipped sessions',
        vol_source='outputs/monthly_2016_extension/FEATURES.npz:vol126',
        vol_source_sha256='1d44b505ddcfd2979e30ba4e73ff528cd1d72efc5ca7e745c3f1c51401fbf5b5',
        axes_sha256='3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
        vol_rank='reuse original rank(pct=True).fillna(.5)-.5 over original49; no rerank or recenter',
        delta='-.02*source_state*original_vol126_rank', max_abs_delta=MAX_ABS_DELTA,
        score_inputs='delta_vix and delta_vvix already include the complete dose',
        population='original49 including ETFs and originally ineligible names; original eligibility selects holdings',
        placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',
        original_arrays=27, extra_arrays=list(EXTRA), initial_partial='exact_incumbent',
        zero_delta='exact_incumbent', core='unchanged_long_momentum_skip42',
        portfolio='original_EW3_two_half_less_than_two_cash; fixed_intramonth_units',
        strict_pit_vintage=False, historical_series='current official reconstruction; not contemporaneous release vintages',
        new_predictive_model_fits=0, nuisance_estimates=0)


def selection_contract():
    return dict(
        selection_count=1, first88_only=True, early53=True, recent35=True,
        later34_diagnostic_only=True, prefix_reselection=False,
        required_complete_source_months=121, initial_partial_exempt=True,
        minimum_close_cagr_gain=.005, cagr_epsilon=1e-12, open_cost_noninferior=True,
        sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
        guards_apply_to='same_close and next_open_5bp; only same_close requires .005 CAGR gain',
        minimum_changed_held_set_months=3, minimum_changed_each_era=1,
        choice='highest_same_close_first88_CAGR; ties_within1e-12_VIX_before_VVIX',
        execution_paths=9)


def integration_contract():
    return dict(
        host_owns='registration, hash verification, source gate, write-once caches and phase receipts',
        prepare_order=['verify_PLAN_code_runtime_source_and_vol_cache_pins',
                       'source_support_all121_exact_windows_before_orders',
                       'load_original27_and_verify_price_parity',
                       'kernel_states_and_deltas_then_panels_all123_prefixes',
                       'publish_cache_then_independent_prepare_before_returns'],
        select_order=['reverify_PLAN_cache_and_independent_prepare',
                      'Backend.training_once_first88', 'changed_support_then_select_first88',
                      'freeze_all_three_order_streams'],
        diagnose_order=['reverify_frozen_selection_and_all_pins',
                        'Backend.diagnostics_fixed_nine_paths',
                        'independent_outcome_and_protected_archive_before_completion'],
        required_receipt_counts=dict(candidate_cells=5978, orders=366, prefixes=123,
                                     feature_matrices=2, paths=9),
        source_support_api='source_support(state_result["audit"]["monthly"], meta)',
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


def source_support(monthly, meta):
    """Validate source-window support only; this never admits a strategy."""
    require(len(meta['dates']) == MONTHS and meta['dates'][0] == INITIAL_PARTIAL and
            meta['dates'].count(INITIAL_PARTIAL) == 1, 'Original122 dates with one initial partial required')
    require(isinstance(monthly, list) and len(monthly) == MONTHS, 'All122 monthly source records required')
    passed_months = []
    for i, row in enumerate(monthly):
        require(isinstance(row, dict) and type(row.get('signal_index')) is int and
                (row['signal_index'], row.get('date')) == (i, meta['dates'][i]), 'Source monthly axes/order changed')
        initial = i == 0
        require(type(row.get('initial_partial')) is bool and row['initial_partial'] == initial,
                'Explicit original initial-partial status required')
        expected = 0 if initial else 252
        require(type(row.get('window_size')) is int and row['window_size'] == expected, 'Exact source-window size required')
        flags = []
        for source in ('VIX', 'VVIX'):
            item = row.get(source)
            require(isinstance(item, dict) and type(item.get('passed')) is bool and
                    type(item.get('observations')) is int, 'Explicit dual-source window checks required')
            flags.append(item['passed'] and item['observations'] == expected)
        if initial:
            require(all(flags) and row.get('window_start') is None and row.get('window_end') is None and
                    all(row[s].get('m') == 0. for s in ('VIX', 'VVIX')), 'Initial source exemption must remain neutral')
        elif all(flags):
            passed_months.append(i)
    return dict(passed=len(passed_months) == MONTHS - 1, full_window_months=len(passed_months),
                required_complete_source_months=MONTHS - 1, initial_partial_exempt=True,
                first88=sum(i < TRAIN for i in passed_months), early53=sum(i < 53 for i in passed_months),
                recent35=sum(53 <= i < TRAIN for i in passed_months), later34=sum(i >= TRAIN for i in passed_months),
                missing_months=[i for i in range(1, MONTHS) if i not in passed_months])


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
    """Same fixed finite guards as159; no outcome-dependent thresholds."""
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
    paths = [Path(__file__), root/'work/test_vvix160_runner.py',
             root/'work/vvix160_kernel.py', root/'work/test_vvix160_kernel.py',
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
