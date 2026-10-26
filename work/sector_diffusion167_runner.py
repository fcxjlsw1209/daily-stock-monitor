"""Fixed three-arm sector diffusion wiring; the registration host owns all file I/O.

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
NAMES = ('incumbent_local67', 'sector_volume_diffusion', 'sector_additive_control')
HANDS = EXTRA = ('delta_followthrough', 'delta_control')
MAX_ABS_DELTA = .02
INITIAL_PARTIAL = '2016-08-17'
OUT = 'outputs/monthly_sector_diffusion167'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json',
               'PRICE_PARITY.json', 'FEATURE_AUDIT.json', 'STATE_VALUES.json', 'SUPPORT_AUDIT.json')
require = base.require


def measurement_contract():
    return dict(
        arms=list(NAMES), promotable_arm=NAMES[1], nonpromotable_control=NAMES[2],
        adjusted_price_offsets=[-253, -42], volume_offsets=[-273, -43],
        paired_session_offsets=[-252, -43], adjusted_window_prices=212,
        volume_window_values=231, activity_baseline_sessions=21,
        window_returns=210, window_pairs=210, skip_sessions=42,
        r='r_s=diff(log(stock adjusted_close)); b_s=diff(log(sector adjusted_close)) for s=t-252..t-42',
        a='a_s=log(sector V_s)-mean(log(sector V_u),u=s-21..s-1) for s=t-252..t-43',
        y='y_s=r_(s+1) for s=t-252..t-43',
        q='product of centered unit-L2 b_s and centered unit-L2 a_s; then center and normalize q to unit L2',
        centering='for each b,a,r,next_b,y and then q: scale by max(abs(v)); center by arithmetic mean; reject zero scale or centered L2 <=1e-12 times scaled uncentered L2; normalize centered vector to unit L2',
        residualization='SVD of unit-centered columns, reject smallest singular <=1e-12*largest; full nuisance order b,a,r,next_b and additive order a,r,next_b; subtract U@(U.T@unit_vector); reject residual L2 <=1e-12 and normalize; either-arm failure makes both unavailable',
        raw_followthrough='partial_Pearson(q,y | b_s,a_s,r_s,b_(s+1)), with intercept through centering',
        raw_control='partial_Pearson(b_s,y | a_s,r_s,b_(s+1)), additive control with intercept; not price-only',
        correlation_roundoff='clip only excess no greater than 64*float64_machine_epsilon to [-1,1]; larger excess makes both unavailable',
        relative_degeneracy_epsilon=1e-12,
        rank='separate average ascending percentile minus .5 on common-valid direct stocks within original49 before trading eligibility; ETF and unknown mapping neutral',
        followthrough_delta='0.04*ranks_followthrough', control_delta='0.04*ranks_control',
        missing='both raw values unavailable and both deltas exactly zero for any missing/nonfinite/nonpositive consumed source, near-constant vector, dependent controls or residual degeneracy; no fill',
        max_abs_delta=MAX_ABS_DELTA, score_inputs='both deltas already include the complete dose',
        control_matching='same raw sources, paired sessions, common support, rank transform, positive dose and missing handling; control is never promotable',
        population='original49 unchanged; mapping from strict_study.benchmark_map at signal date; only non-None mapped direct stocks measured, ETFs and unknown mapping neutral; eligibility and original27 arrays unchanged',
        prices='panel has exactly aligned adj and volume DataFrames plus benchmark_map date->complete symbol->sector-or-None; only stock/sector adjusted close and sector volume windows read; no raw close or stock volume',
        placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',
        original_arrays=27, extra_arrays=list(EXTRA), prepared_arrays=29,
        initial_partial='exact_incumbent; measurable raw_followthrough/raw_control and ranks may remain in audit; common support false',
        zero_delta='exact_incumbent', core='unchanged_long_momentum_skip42',
        portfolio='original_EW3_two_half_less_than_two_cash; fixed_intramonth_units',
        strict_pit_vintage=False, strict_pit_membership=False,
        historical_series='inherited price vintages and estimated Top32 plus assumed theme17 historical membership proxy',
        new_predictive_model_fits=0, nuisance_estimation='within-window intercept and four/three-control partial correlations only')


def selection_contract():
    return dict(
        selection_count=1, first88_only=True, early53=True, recent35=True,
        later34_diagnostic_only=True, prefix_reselection=False,
        source_support='all structural and missing-neutral validation passes; common support divided by noninitial eligible mapped direct-stock cells; no coverage threshold; report full49 counts separately',
        minimum_common_support_cells=None, initial_partial_exempt=True,
        minimum_close_cagr_gain=.005, cagr_epsilon=1e-12, open_cost_noninferior=True,
        sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
        guards_apply_to='followthrough vs incumbent: same_close requires .005 CAGR gain and next_open_5bp is noninferior; followthrough vs sector_additive_control: both same_close and next_open_5bp are noninferior; every comparison uses the same Sharpe/drawdown/era guards',
        minimum_changed_held_set_months=3, minimum_changed_each_era=1,
        choice='sector_volume_diffusion only if every incumbent and matched-control guard passes; otherwise incumbent_local67; sector_additive_control is never selected',
        promotable_arm=NAMES[1], nonpromotable_control=NAMES[2],
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
                'Finite score delta within the fixed two-percent bound required')
        for i, day in enumerate(meta['dates'][:n]):
            etf_columns = [j for j, symbol in enumerate(meta['symbols'][i]) if symbol in meta['etfs']]
            require(np.all(delta[i, etf_columns] == 0.), 'ETF deltas must remain exactly neutral')
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
    require(len(arrays) == 29 and len(set(arrays) - set(EXTRA)) == 27,
            'Exactly original27 arrays plus two followthrough arrays required')
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    choices, scores, audit = choices_all(arrays, meta, MONTHS, backend, policy)
    for n in range(MONTHS + 1):
        cc, ss, _ = choices_all(arrays, meta, n, backend, policy)
        require(all(cc[k] == choices[k][:n] and ss[k].tobytes() == scores[k][:n].tobytes() for k in NAMES),
                'Prefix choices/scores changed')
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays mutated')
    return {k: backend.make_orders(choices[k], scores[k], meta) for k in NAMES}, scores, audit


def source_support(audit, meta, deltas=None):
    """Validate shared followthrough/control support; coverage has no threshold."""
    require(len(meta['dates']) == len(meta['symbols']) == MONTHS and meta['dates'][0] == INITIAL_PARTIAL and
            meta['dates'].count(INITIAL_PARTIAL) == 1, 'Original122 dates with one initial partial required')
    require(all(type(audit.get(key)) is int and audit[key] == value for key, value in
                (('events', MONTHS), ('symbols_per_event', WIDTH), ('window_returns', 210), ('skip_sessions', 42),
                 ('window_pairs', 210), ('adjusted_window_prices', 212), ('volume_window_values', 231), ('activity_baseline_sessions', 21))),
            'Exact122x49 and210-pair skip42 audit required')
    require(audit.get('relative_degeneracy_epsilon') == 1e-12, 'Fixed numerical degeneracy epsilon required')
    require(audit.get('trading_eligibility_modified') is False and audit.get('outcomes_used') is False,
            'Followthrough features must not alter eligibility or use outcomes')
    rows = audit.get('event_history')
    require(isinstance(rows, list) and len(rows) == MONTHS, 'All122 source records required')
    if deltas is not None:
        require(set(deltas) == set(EXTRA), 'Exact two score delta matrices required')
        for key in EXTRA:
            a = np.asarray(deltas[key])
            require(a.shape == (MONTHS, WIDTH) and a.dtype == np.float64 and np.isfinite(a).all() and
                    np.all(np.abs(a) <= MAX_ABS_DELTA), 'Finite bounded float64 deltas required')
    masks = []
    direct_masks = []
    for i, row in enumerate(rows):
        require(isinstance(row, dict) and row.get('date') == meta['dates'][i] and
                row.get('symbols') == meta['symbols'][i] and len(meta['symbols'][i]) == WIDTH and
                len(set(meta['symbols'][i])) == WIDTH, 'Source monthly axes/order changed')
        initial = i == 0
        require(type(row.get('initial_forced_neutral')) is bool and row['initial_forced_neutral'] == initial,
                'Explicit original initial-partial status required')
        direct, sectors = row.get('eligible_direct_stock'), row.get('sector_symbols')
        require(isinstance(direct, list) and len(direct) == WIDTH and all(type(x) is bool for x in direct),
                'Exact boolean mapped direct-stock eligibility required')
        sector_etfs = {'XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY'}
        require(isinstance(sectors, list) and len(sectors) == WIDTH and
                all(s is None or (type(s) is str and s in sector_etfs) for s in sectors),
                '49 mapped sector identities or explicit unknowns required')
        require(direct == [s is not None for s in sectors], 'Direct eligibility must equal mapped source identities')
        require(all(not direct[j] for j, s in enumerate(meta['symbols'][i]) if s in meta['etfs']),
                'ETF candidates must remain neutral')
        eligible_direct = np.asarray(direct, dtype=bool)
        start, end = row.get('window_start'), row.get('window_end')
        volume_start, volume_end, pair_start, pair_end = (row.get(key) for key in
            ('volume_window_start', 'volume_window_end', 'pair_start', 'pair_end'))
        for value in (start, end, volume_start, volume_end, pair_start, pair_end):
            require(value is None or (type(value) is str and len(value) == 10 and
                    pd.Timestamp(value).strftime('%Y-%m-%d') == value and value < row['date']),
                    'Window dates must be canonical past dates or missing')
        require(start is None or (end is not None and start < end), 'Ordered window endpoints required')
        require(volume_end == pair_end and (volume_end is None or (end is not None and volume_end < end)),
                'Volume window and pair end must precede adjusted-window end')
        require(start is None or (pair_start is not None and pair_end is not None and start < pair_start <= pair_end),
                'Pair dates must be ordered after the adjusted history start')
        require(volume_start is None or (start is not None and volume_start < start),
                'Volume history must start before adjusted history')
        raw_support = row.get('common_support')
        require(isinstance(raw_support, list) and len(raw_support) == WIDTH and
                all(type(x) is bool for x in raw_support), 'Exact boolean common support required')
        support = np.asarray(raw_support, dtype=bool)
        available_both = []
        for suffix, delta_key in zip(('followthrough', 'control'), EXTRA):
            raw, ranks = row.get('raw_' + suffix), row.get('ranks_' + suffix)
            require(isinstance(raw, list) and isinstance(ranks, list) and len(raw) == len(ranks) == WIDTH,
                    '49 raw correlation values and ranks required for each arm')
            available = np.zeros(WIDTH, dtype=bool)
            for j, (value, rank) in enumerate(zip(raw, ranks)):
                require(value is None or (type(value) in (int, float) and math.isfinite(value) and -1. <= value <= 1.),
                        'Raw correlation must be finite in [-1,1] or explicitly missing')
                available[j] = value is not None
                require((rank is None and value is None) or
                        (value is not None and type(rank) in (int, float) and math.isfinite(rank) and -.5 <= rank <= .5),
                        'Ranks must have exactly the finite raw-value support')
            available_both.append(available)
            require(not np.any(available & ~eligible_direct), 'Unmapped candidates must not have raw measurements')
            require(not available.any() or all(x is not None for x in (start, end, volume_start, volume_end, pair_start, pair_end)),
                    'Measured correlations require all window endpoints')
            require(np.array_equal(support, available & (not initial)),
                    'Common support must match both finite correlations except the initial partial')
            expected_ranks = np.zeros(WIDTH)
            if available.any():
                expected_ranks[available] = pd.Series([raw[j] for j in np.flatnonzero(available)]).rank(
                    method='average', ascending=True, pct=True).to_numpy() - .5
                require(np.allclose(np.asarray([ranks[j] for j in np.flatnonzero(available)]),
                                    expected_ranks[available], atol=1e-12, rtol=0),
                        'Ranks differ from shared-pool ascending average percentiles')
            if deltas is not None:
                actual = np.asarray(deltas[delta_key])[i]
                expected = np.zeros(WIDTH)
                expected[support] = .04 * expected_ranks[support]
                require(np.all(actual[~support] == 0.), 'Missing support and initial rows must have exact zero deltas')
                require(np.allclose(actual, expected, atol=1e-12, rtol=0),
                        'Deltas must be the single fixed positive percentile dose')
        require(np.array_equal(*available_both), 'Both raw correlations require identical availability')
        masks.append(support)
        direct_masks.append(eligible_direct & (not initial))
    common = np.asarray(masks)
    direct = np.asarray(direct_masks)
    supported = int(common.sum())
    eligible_total = int(direct.sum())
    require(type(audit.get('eligible_direct_stock_cells')) is int and
            audit['eligible_direct_stock_cells'] == eligible_total, 'Eligible direct-stock denominator must match rows')
    require(type(audit.get('common_support_cells')) is int and type(audit.get('neutral_cells')) is int and
            audit['common_support_cells'] == supported and audit['neutral_cells'] == MONTHS*WIDTH-supported,
            'Audit aggregate counts must match row evidence')
    coverage = {}
    for name, lo, hi in (('full122', 0, MONTHS), ('first88', 0, TRAIN), ('early53', 0, 53),
                         ('recent35', 53, TRAIN), ('later34', TRAIN, MONTHS)):
        total, count, eligible = (hi-lo)*WIDTH, int(common[lo:hi].sum()), int(direct[lo:hi].sum())
        coverage[name] = dict(candidate_cells=total, common_support_cells=count,
                              eligible_direct_stock_cells=eligible,
                              neutral_cells=total-count, coverage=count/eligible if eligible else None)
    return dict(passed=True, candidate_cells=MONTHS*WIDTH, common_support_cells=supported,
                eligible_direct_stock_cells=eligible_total,
                neutral_cells=MONTHS*WIDTH-supported, initial_partial_exempt=True,
                coverage_threshold=None, coverage=coverage)


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
    followthrough = dict(source_support=support['passed'], changed_holdings=changed[NAMES[1]].get('passed') is True,
                    close=metric_guard(rows[1], rows[0], .005), cost_open=metric_guard(opened[1], opened[0], 0.),
                    close_vs_control=metric_guard(rows[1], rows[2], 0.),
                    cost_open_vs_control=metric_guard(opened[1], opened[2], 0.))
    guards = {NAMES[1]: followthrough, NAMES[2]: dict(promotable=False)}
    winner = 1 if all(followthrough.values()) else 0
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
    paths = [Path(__file__), root/'work/test_sector_diffusion167_runner.py',
             root/'work/sector_diffusion167_kernel.py', root/'work/test_sector_diffusion167_kernel.py',
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
