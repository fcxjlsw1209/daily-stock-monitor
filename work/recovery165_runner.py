"""Fixed three-arm stock-sector recovery wiring; the registration host owns all file I/O.

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
NAMES = ('incumbent_local67', 'recovery_race', 'impact_control')
HANDS = EXTRA = ('delta_recovery', 'delta_control')
MAX_ABS_DELTA = .02
INITIAL_PARTIAL = '2016-08-17'
OUT = 'outputs/monthly_recovery165'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json',
               'PRICE_PARITY.json', 'FEATURE_AUDIT.json', 'STATE_VALUES.json', 'SUPPORT_AUDIT.json')
require = base.require
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
from recovery165_kernel import SECTOR_ETFS


def measurement_contract():
    return dict(
        arms=list(NAMES), promotable_arm=NAMES[1], nonpromotable_control=NAMES[2],
        adjusted_price_offsets=[-252, -42], window_prices=211, window_returns=210,
        skip_sessions=42, start_indices=[1, 189], start_count=189,
        recovery_horizon=21, censor_value=22, minimum_joint_down=20,
        event='same start s=1..189 where both stock and own sector ETF have strictly negative simple returns; detect with exact price comparisons',
        recovery='T is first k=1..21 with P[s+k]>=P[s-1], else22; levels are separate for stock and own sector ETF',
        raw_recovery='sum(T_sector-T_stock)/(21*count) over identical joint-down starts; one final division preserves discrete-mean ties',
        raw_control='fsum((r_stock-r_sector)/(abs(r_stock)+abs(r_sector)))/count over identical joint-down starts; r=(P[s]-P[s-1])/P[s-1]',
        rank='separate average ascending percentile minus .5 for recovery/control on shared finite direct-stock support within original49 before trading eligibility',
        recovery_delta='0.04*ranks_recovery', control_delta='0.04*ranks_control',
        missing='both raw values unavailable and both deltas exactly zero for any missing/nonfinite/nonpositive value among all211 paired stock/sector prices, fewer than20 joint-down starts, ETF candidate or unknown sector mapping; no fill',
        max_abs_delta=MAX_ABS_DELTA, score_inputs='both deltas already include the complete dose',
        control_matching='same stock-sector pair, full price window, identical joint-down starts, common support, positive dose and missing handling; control is never promotable',
        population='current original49 pool; only mapped direct stocks ranked; all ETFs neutral; current trading eligibility and all original27 arrays unchanged',
        mapping='frozen strict_study.benchmark_map at signal date with original17 ETF identities; its SECTOR and FOREIGN assumptions retained; only its11 allowed sector ETF values',
        mapping_source_sha256=MAPPING_SHA,
        prices='adjusted close panel from pinned local inputs; slice official window before numeric conversion; no prices after t-42 are read',
        placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',
        original_arrays=27, extra_arrays=list(EXTRA), prepared_arrays=29,
        initial_partial='exact_incumbent; measurable raw_recovery/raw_control and ranks may remain in audit; common support false',
        zero_delta='exact_incumbent', core='unchanged_long_momentum_skip42',
        portfolio='original_EW3_two_half_less_than_two_cash; fixed_intramonth_units',
        strict_pit_vintage=False, strict_pit_membership=False,
        historical_series='inherited price vintages, sector classifications/foreign proxy assumptions, estimated Top32 plus assumed theme17 historical membership proxy',
        new_predictive_model_fits=0, nuisance_estimation='none')


def selection_contract():
    return dict(
        selection_count=1, first88_only=True, early53=True, recent35=True,
        later34_diagnostic_only=True, prefix_reselection=False,
        source_support='all structural and missing-neutral validation passes; coverage is reported without a threshold',
        minimum_common_support_cells=None, initial_partial_exempt=True,
        minimum_close_cagr_gain=.005, cagr_epsilon=1e-12, open_cost_noninferior=True,
        sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
        guards_apply_to='recovery vs incumbent: same_close requires .005 CAGR gain and next_open_5bp is noninferior; recovery vs impact_control: both same_close and next_open_5bp are noninferior; every comparison uses the same Sharpe/drawdown/era guards',
        minimum_changed_held_set_months=3, minimum_changed_each_era=1,
        choice='recovery_race only if every incumbent and matched-control guard passes; otherwise incumbent_local67; impact_control is never selected',
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
            'Exactly original27 arrays plus two recovery arrays required')
    before = {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}
    choices, scores, audit = choices_all(arrays, meta, MONTHS, backend, policy)
    for n in range(MONTHS + 1):
        cc, ss, _ = choices_all(arrays, meta, n, backend, policy)
        require(all(cc[k] == choices[k][:n] and ss[k].tobytes() == scores[k][:n].tobytes() for k in NAMES),
                'Prefix choices/scores changed')
    require(before == {k: (v.dtype.str, v.shape, v.tobytes()) for k, v in arrays.items()}, 'Original arrays mutated')
    return {k: backend.make_orders(choices[k], scores[k], meta) for k in NAMES}, scores, audit


def source_support(audit, meta, deltas=None):
    """Verify stock-only matched support; report coverage without a threshold."""
    require(len(meta['dates']) == len(meta['symbols']) == MONTHS and meta['dates'][0] == INITIAL_PARTIAL and
            meta['dates'].count(INITIAL_PARTIAL) == 1, 'Original122 dates with one initial partial required')
    etfs = meta['etfs']
    require(isinstance(etfs, list) and len(etfs) == len(set(etfs)) == 17, 'Original17 ETF identities required')
    require(all(type(audit.get(key)) is int and audit[key] == value for key, value in
                (('events', MONTHS), ('symbols_per_event', WIDTH), ('window_returns', 210), ('window_prices', 211),
                 ('skip_sessions', 42), ('start_count', 189), ('recovery_horizon', 21), ('censor_value', 22),
                 ('minimum_joint_down', 20))), 'Exact122x49 and211-price skip42 recovery audit required')
    require(audit.get('trading_eligibility_modified') is False and audit.get('outcomes_used') is False,
            'Recovery features must not alter eligibility or use outcomes')
    rows = audit.get('event_history')
    require(isinstance(rows, list) and len(rows) == MONTHS, 'All122 source records required')
    if deltas is not None:
        require(set(deltas) == set(EXTRA), 'Exact two score delta matrices required')
        for key in EXTRA:
            a = np.asarray(deltas[key])
            require(a.shape == (MONTHS, WIDTH) and a.dtype == np.float64 and np.isfinite(a).all() and
                    np.all(np.abs(a) <= MAX_ABS_DELTA), 'Finite bounded float64 deltas required')
    masks = []
    for i, row in enumerate(rows):
        names = meta['symbols'][i]
        require(isinstance(row, dict) and row.get('date') == meta['dates'][i] and row.get('symbols') == names and
                len(names) == len(set(names)) == WIDTH and set(etfs) <= set(names), 'Source monthly axes/order changed')
        initial = i == 0
        require(type(row.get('initial_forced_neutral')) is bool and row['initial_forced_neutral'] == initial,
                'Explicit original initial-partial status required')
        start, end = row.get('window_start'), row.get('window_end')
        for value in (start, end):
            require(value is None or (type(value) is str and len(value) == 10 and
                    pd.Timestamp(value).strftime('%Y-%m-%d') == value and value < row['date']),
                    'Window dates must be canonical past dates or missing')
        require(start is None or (end is not None and start < end), 'Ordered window endpoints required')
        support_values, benchmarks, counts, reasons = (row.get(key) for key in
                ('common_support', 'sector_etfs', 'joint_down_counts', 'source_status'))
        require(all(isinstance(v, list) and len(v) == WIDTH for v in (support_values, benchmarks, counts, reasons)),
                'Exact support, mapping, count and status vectors required')
        require(all(type(v) is bool for v in support_values), 'Exact boolean common support required')
        support = np.asarray(support_values)
        expected_available = np.zeros(WIDTH, dtype=bool)
        for j, (name, benchmark, count, reason) in enumerate(zip(names, benchmarks, counts, reasons)):
            require(benchmark is None or (type(benchmark) is str and benchmark in SECTOR_ETFS),
                    'Only frozen sector ETF benchmarks permitted')
            is_etf = name in etfs or name in SECTOR_ETFS
            require(not is_etf or benchmark is None, 'ETF candidates must remain neutral')
            require(count is None or (type(count) is int and 0 <= count <= 189), 'Joint-down counts must be 0..189 or missing')
            if is_etf:
                require(reason == 'etf_not_applicable' and count is None, 'ETF source status must be explicit')
            elif benchmark is None:
                require(reason == 'missing_sector_mapping' and count is None, 'Unknown sector mapping must remain neutral')
            elif reason in ('invalid_stock_path', 'invalid_sector_path', 'insufficient_history'):
                require(count is None, 'Invalid paired paths cannot supply event counts')
            elif reason == 'insufficient_joint_down':
                require(count is not None and count < 20, 'Insufficient-event status requires fewer than20 events')
            elif reason == 'available':
                require(count is not None and count >= 20, 'Available paired source needs at least20 events')
                expected_available[j] = True
            else:
                raise ValueError('Unrecognized source status')
        require(not expected_available.any() or (start is not None and end is not None),
                'Measured recovery requires both window endpoints')
        require(np.array_equal(support, expected_available & (not initial)),
                'Common support must match paired source availability except initial partial')
        for suffix, delta_key in zip(('recovery', 'control'), EXTRA):
            raw, ranks = row.get('raw_' + suffix), row.get('ranks_' + suffix)
            require(isinstance(raw, list) and isinstance(ranks, list) and len(raw) == len(ranks) == WIDTH,
                    '49 raw values and ranks required for each arm')
            available = np.zeros(WIDTH, dtype=bool)
            for j, (value, rank) in enumerate(zip(raw, ranks)):
                require(value is None or (type(value) in (int, float) and math.isfinite(value) and -1. <= value <= 1.),
                        'Raw value must be finite in [-1,1] or explicitly missing')
                available[j] = value is not None
                require((rank is None and value is None) or
                        (value is not None and type(rank) in (int, float) and math.isfinite(rank) and -.5 <= rank <= .5),
                        'Ranks must have exactly the finite raw-value support')
            require(np.array_equal(available, expected_available), 'Both raw values must have identical paired stock support')
            expected_ranks = np.zeros(WIDTH)
            if available.any():
                expected_ranks[available] = pd.Series([raw[j] for j in np.flatnonzero(available)]).rank(
                    method='average', ascending=True, pct=True).to_numpy() - .5
                require(np.allclose(np.asarray([ranks[j] for j in np.flatnonzero(available)]),
                                    expected_ranks[available], atol=1e-12, rtol=0),
                        'Ranks differ from shared-stock ascending average percentiles')
            if deltas is not None:
                actual = np.asarray(deltas[delta_key])[i]
                expected = np.zeros(WIDTH)
                expected[support] = .04 * expected_ranks[support]
                require(np.all(actual[~support] == 0.), 'Missing support and initial rows require exact zero deltas')
                require(np.allclose(actual, expected, atol=1e-12, rtol=0), 'Deltas require the single fixed positive percentile dose')
        masks.append(support)
    common = np.asarray(masks)
    supported = int(common.sum())
    require(type(audit.get('common_support_cells')) is int and type(audit.get('neutral_cells')) is int and
            audit['common_support_cells'] == supported and audit['neutral_cells'] == MONTHS*WIDTH-supported,
            'Audit aggregate counts must match row evidence')
    coverage = {}
    for name, lo, hi in (('full122', 0, MONTHS), ('first88', 0, TRAIN), ('early53', 0, 53),
                         ('recent35', 53, TRAIN), ('later34', TRAIN, MONTHS)):
        total, count = (hi-lo)*WIDTH, int(common[lo:hi].sum())
        coverage[name] = dict(candidate_cells=total, common_support_cells=count,
                              neutral_cells=total-count, coverage=count/total)
    return dict(passed=True, candidate_cells=MONTHS*WIDTH, common_support_cells=supported,
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
    recovery = dict(source_support=support['passed'], changed_holdings=changed[NAMES[1]].get('passed') is True,
                    close=metric_guard(rows[1], rows[0], .005), cost_open=metric_guard(opened[1], opened[0], 0.),
                    close_vs_control=metric_guard(rows[1], rows[2], 0.),
                    cost_open_vs_control=metric_guard(opened[1], opened[2], 0.))
    guards = {NAMES[1]: recovery, NAMES[2]: dict(promotable=False)}
    winner = 1 if all(recovery.values()) else 0
    return dict(primary=NAMES[winner], admitted=winner != 0, selection_count=1,
                guards=guards, later34_used_for_selection=False)


class Backend(base.Local67Backend):
    price_parity = previous.Backend.price_parity

    def sector_mapping(self, meta):
        """Use the already loaded frozen mapper and registered raw sector panel."""
        expected = self.root/'outputs/monthly_strict_relative/strict_study.py'
        mapper = self.local.ex.t.strict
        require(Path(mapper.__file__).resolve() == expected.resolve() and base.sha(expected) == MAPPING_SHA,
                'Frozen sector mapper changed')
        require(set(mapper.SECTOR.values()) == SECTOR_ETFS and set(mapper.FOREIGN.values()) <= SECTOR_ETFS,
                'Frozen allowed sector ETF set changed')
        etfs, sectors = meta['etfs'], self.panel['sectors']
        require(isinstance(etfs, list) and len(etfs) == len(set(etfs)) == 17,
                'Original17 ETF identities required')
        require(isinstance(sectors, pd.DataFrame) and isinstance(sectors.index, pd.DatetimeIndex) and
                sectors.index.is_unique and sectors.columns.is_unique, 'Unique registered sector panel required')
        require(len(meta['dates']) == len(meta['symbols']) and len(set(meta['dates'])) == len(meta['dates']),
                'Exact unique original event axes required')
        rows = {}
        for day, names in zip(meta['dates'], meta['symbols']):
            require(len(names) == len(set(names)) == WIDTH and set(etfs) <= set(names),
                    'Original32 stocks plus17 ETFs required')
            row = mapper.benchmark_map(day, names, sectors, etfs)
            require(isinstance(row, dict) and set(row) == set(names), 'Exact frozen benchmark map required')
            require(all(value is None or value in SECTOR_ETFS for value in row.values()),
                    'Unregistered sector ETF returned by mapper')
            require(all(row[name] is None for name in names if name in etfs or name in SECTOR_ETFS),
                    'ETF candidates must have neutral mappings')
            rows[day] = row
        return dict(etfs=list(etfs), by_date=rows)

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
    paths = [Path(__file__), root/'work/test_recovery165_runner.py',
             root/'work/recovery165_kernel.py', root/'work/test_recovery165_kernel.py',
             root/'work/basic_shares159_host.py',
             root/'outputs/monthly_strict_relative/strict_study.py',
             root/'outputs/index_union/alias_resolved_inputs/sectors.csv.gz']
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
