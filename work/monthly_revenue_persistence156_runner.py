"""Four fixed revenue-persistence arms; actual work requires a registered PLAN.

Source collection, source-grid review and strategy registration are separate.
This module reuses the frozen scoring/accounting helpers without changing them.
"""
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import monthly_forward_guidance132_runner as base
import monthly_stock_comp149_runner as prior
import after155_revenue_pilot_factor as factor
from after155_revenue_coverage_gate import coverage_gate, USABLE
from after155_multi_issuer_chain import evaluate as evaluate_multi
from after155_revenue_chain import evaluate as evaluate_msft

require, sha, read = base.require, base.sha, base.read
NAMES = ('incumbent_local67', 'matched_latest_revenue_growth',
         'matched_annualized_total_revenue_growth', 'weakest_annual_revenue_growth')
HANDS = factor.NAMES
TRAIN, MONTHS, WIDTH, DOSE = 88, 122, 49, .01
OUT = 'outputs/monthly_revenue_persistence156'
SOURCE_OUT = 'outputs/monthly_revenue_pilot_source156'
CACHE_FILES = base.CACHE_FILES
FROZEN_FILES = prior.FROZEN_FILES
TOLERANCES = dict(prior.TOLERANCES)
CHECKS = (*base.CHECKS, 'revenue_source_grid_verified',
          'all_three_revenue_features_common_mask', 'original_source_cutoffs_verified')
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'sessions',
          'protected_manifest', 'source_manifest', 'source_grid_review', 'source_plan', 'source_calendar')
FIXED = ['MSFT', 'AAPL', 'ABBV']
FIXED_INPUTS = {
    'meta': ('outputs/monthly_local_linear70/META.json',
             '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'),
    'source_plan': ('work/after155_revenue_pilot_source_plan.json',
                    '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10'),
    'source_calendar': ('outputs/monthly_paired_target143/CALENDAR.json',
                       '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'),
}
SOURCE_FILES = ('ROWS.json', 'REPORTS.json', 'EDGES.json', 'COVERAGE_GATE.json', 'SUMMARY.json')
selection_contract = prior.selection_contract
corrected = prior.corrected


def measurement_contract():
    return dict(levels='four_consecutive_own_year_originals_oldest_first',
        latest='h(R3/R2-1)', annualized='h((R3/R0)^(1/3)-1)',
        weakest='h(min(R1/R0-1,R2/R1-1,R3/R2-1))', bounded='h(x)=x/(1+abs(x))',
        common_mask='identical_independently_reviewed_four_original_chain_for_all_three_features',
        unavailable='NaN_with_reason_and_exact_incumbent_score_fallback',
        placement='add_0.01_h_to_original_pre_score_then_one_original_episode_selector',
        source_scope=list(FIXED), original_candidate_axis=[122, 49], unchanged_original_arrays=27,
        invalid_latest_blocks_fallback=True, maximum_annual_age_days=550,
        no_parameter_search=True, new_model_fits=0, etf_financial_features=False,
        monthly_equal_weight_positions=3, intramonth_trades=False)


def validate_proposal(proposal):
    require(proposal.get('round') == 156 and proposal.get('arms') == list(NAMES)
            and proposal.get('primary_challenger') == NAMES[3] and proposal.get('dose') == DOSE,
            'Wrong prospective revenue experiment')
    require(proposal.get('selection_contract') == selection_contract(), 'Selection contract changed')
    require(proposal.get('measurement') == measurement_contract(), 'Measurement contract changed')
    require(proposal.get('status') == 'FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
            'Final prospective proposal required')
    for key in ('actual_new_factors', 'actual_new_orders', 'actual_new_returns'):
        require(type(proposal.get(key)) is int and proposal[key] == 0, 'No preregistration calculation')


def verify_cutoffs(meta, rows, cutoffs):
    """Validate every clock before any factor is computed."""
    require(set(meta['dates']) <= set(cutoffs), 'Missing original decision close')
    for row in rows:
        day = row['decision_date']
        timestamp = pd.Timestamp(row['decision_cutoff'])
        require(timestamp.tzinfo is not None and type(row['decision_cutoff_ns']) is int
                and row['decision_cutoff_ns'] == int(timestamp.value) == cutoffs[day]
                and str(timestamp.tz_convert('America/New_York').date()) == day,
                'Source row is not bound to the original exchange close')


def validated_members(meta, rows, reports, edges, cutoffs, source_plan):
    """Recheck the entire supplied grid, then return only its admitted originals."""
    require(source_plan['fixed_issuer_scope'] == FIXED, 'Fixed issuer scope changed')
    coverage_gate(meta, rows, source_plan)  # Shape and slot identity, including held/nonpilot rows.
    verify_cutoffs(meta, rows, cutoffs)
    require(set(reports) == set(edges) == set(FIXED), 'Exact source issuer catalogs required')
    indexes = {}
    for issuer in FIXED:
        catalog = reports[issuer]
        indexes[issuer] = {item['report_id']: item for item in catalog}
        require(len(indexes[issuer]) == len(catalog), 'Duplicate original report identity')
    selected = []
    for row in rows:
        if row['status'] != USABLE:
            continue
        issuer = row['symbol']
        args = (reports[issuer], edges[issuer], row['decision_cutoff_ns'], row['decision_date'])
        check = evaluate_msft(*args) if issuer == 'MSFT' else evaluate_multi(issuer, *args)
        require(check['status'] == USABLE and all(check[key] == row[key]
                for key in ('report_ids', 'edge_ids', 'latest_report_id')),
                'Prepared source chain no longer matches its reviewed original identities')
        ids = row['report_ids']
        require(len(ids) == len(set(ids)) == 4, 'Exactly four unique originals required')
        originals = [indexes[issuer][identifier] for identifier in ids]
        years = [item['fiscal_year'] for item in originals]
        require(all(type(year) is int for year in years)
                and years == list(range(years[-1] - 3, years[-1] + 1)), 'Oldest-first annual order')
        require(len({Decimal(item['scale']) for item in originals}) == 1,
                'All four original levels must use one scale')
        require(row.get('source_refs') == [item['source']['raw'] for item in originals],
                'Source row raw-file references do not match selected originals')
        selected.append((row, originals))
    return selected


def build_features(meta, rows, reports, edges, cutoffs, source_plan):
    """Pure synthetic-testable builder; the actual caller must verify byte closure first."""
    selected = validated_members(meta, rows, reports, edges, cutoffs, source_plan)
    arrays = {key: np.full((MONTHS, WIDTH), np.nan) for key in HANDS}
    arrays['common_available'] = np.zeros((MONTHS, WIDTH), dtype=bool)
    cells = {(row['signal_index'], row['candidate_index']): dict(
        signal_index=row['signal_index'], candidate_index=row['candidate_index'],
        symbol=row['symbol'], decision_date=row['decision_date'], status=row['status'],
        reasons=list(row.get('reasons', [])), available=False) for row in rows}
    for row, originals in selected:
        i, j = row['signal_index'], row['candidate_index']
        measured = factor.measure_levels([item['current'] for item in originals])
        for key in HANDS:
            arrays[key][i, j] = measured[key]
        arrays['common_available'][i, j] = True
        cells[(i, j)].update(available=True, report_ids=list(row['report_ids']),
            edge_ids=list(row['edge_ids']), latest_report_id=row['latest_report_id'],
            decision_cutoff_ns=row['decision_cutoff_ns'], features=measured)
    return dict(arrays=arrays, audit=dict(measurement=measurement_contract(), cells=[
        cells[(i, j)] for i in range(MONTHS) for j in range(WIDTH)],
        rows=MONTHS * WIDTH, usable_rows=len(selected), source_holds_preserved=True,
        all_three_revenue_features_common_mask=True, original_source_cutoffs_verified=True,
        financial_source_scope=FIXED, new_model_fits=0))


def select(rows, opened):
    for group in (rows, opened):
        require([row['name'] for row in group] == list(NAMES), 'Four exact ordered arms required')
    closed = {NAMES[i]: prior.previous.guard(rows[3], rows[i], strict=True) for i in range(3)}
    op = {NAMES[i]: prior.previous.guard(opened[3], opened[i], strict=i != 0) for i in range(3)}
    passed = all(closed.values()) and all(op.values())
    return dict(primary=NAMES[3] if passed else NAMES[0], admitted=passed,
                guards_against=closed, next_open_guards_against=op, mechanism_confirmed=all(op.values()))


def choose(name, arrays, meta, n, backend, policy):
    require(name in NAMES, 'Undeclared arm')
    if name == NAMES[0]:
        return backend.baseline(arrays, meta, n, policy)
    return corrected(backend.pre(arrays, n), arrays[HANDS[NAMES.index(name) - 1]],
                     arrays, meta, n, backend, policy)


def panels(arrays, meta, backend, policy):
    before = {key: (value.dtype.str, value.shape, value.tobytes()) for key, value in arrays.items()}
    common = arrays['common_available']
    require(common.dtype == bool and common.shape == (MONTHS, WIDTH), 'Full common source mask')
    for key in HANDS:
        require(np.array_equal(np.isfinite(arrays[key]), common), 'Identical three-feature source mask')
    orders, scores = {}, {}
    for name in NAMES:
        choices, score = choose(name, arrays, meta, MONTHS, backend, policy)
        orders[name], scores[name] = backend.make_orders(choices, score, meta), score
        for n in range(MONTHS + 1):
            pc, ps = choose(name, arrays, meta, n, backend, policy)
            require(pc == choices[:n] and ps.tobytes() == score[:n].tobytes(), 'Future-prefix dependence')
    for value in (0., np.nan):
        null = dict(arrays)
        for key in HANDS:
            null[key] = np.full((MONTHS, WIDTH), np.nan)
            for i, symbols in enumerate(meta['symbols']):
                for j, symbol in enumerate(symbols):
                    if symbol not in meta['etfs']:
                        null[key][i, j] = value
        for name in NAMES[1:]:
            choices, score = choose(name, null, meta, MONTHS, backend, policy)
            require(backend.make_orders(choices, score, meta) == orders[NAMES[0]]
                    and score.tobytes() == scores[NAMES[0]].tobytes(), 'Zero/missing incumbent fallback')
    require(before == {key: (value.dtype.str, value.shape, value.tobytes())
                       for key, value in arrays.items()}, 'Original inputs changed')
    return orders, scores


class Backend(prior.Backend):
    """Keep the original backend, prefix fix, replay engine and open-path accounting."""
    def training(self, orders, meta, px, op):
        rp = self.local.rp
        blocks, _, bounds, valid = rp.paths(px, meta, TRAIN)
        returns, rows = [], []
        for name in NAMES:
            selected = [[meta['symbols'][i].index(symbol) for symbol in row['symbols']]
                        for i, row in enumerate(orders[name][:TRAIN])]
            rr = rp.fast_returns(selected, blocks, valid)
            returns.append(rr)
            rows.append(dict(name=name, **self.local.old.session.measure(rr, meta, bounds)))
        opened = self.open_training(orders, meta, px, op, TRAIN, NAMES)
        folds, stitched, control = [], [], []
        for n, end in ((36, 60), (60, 76), (76, 88)):
            prefix = [dict(name=name, **self.local.old.session.measure(rr, meta, bounds, n))
                      for name, rr in zip(NAMES, returns)]
            oprefix = self.open_training(orders, meta, px, op, n, NAMES)
            pick = select(prefix, oprefix)
            folds.append(dict(n=n, end=end, selection=pick, open_training=oprefix))
            stitched.extend(returns[NAMES.index(pick['primary'])][bounds[n]:bounds[end]])
            control.extend(returns[0][bounds[n]:bounds[end]])
        walk = dict(folds=folds, pristine_oos=False,
            metrics=rp.stats_rr(np.asarray(stitched), meta['dates'][36], meta['ends'][87]),
            incumbent=rp.stats_rr(np.asarray(control), meta['dates'][36], meta['ends'][87]))
        return rows, opened, walk

    def diagnostics(self, orders, meta, px, op, writer):
        results = {}
        for name in NAMES:
            results[name] = {}
            for mode, cost, key in (('close', 0., 'same_close'), ('open', 0., 'next_open_zero'),
                                    ('open', .0005, 'next_open_cost')):
                met, eq, trades, holdings = self.local.ex.e.engine.replay(px, op, orders[name],
                    cost=cost, execution=mode, start=pd.Timestamp(meta['dates'][0]),
                    end=pd.Timestamp(meta['ends'][-1]))
                for row in met['annual_returns']:
                    if row['year'] in (2016, 2026):
                        row['partial'] = True
                results[name][key] = dict(metrics=met,
                    slices=self.local.arch.old.prior.old.previous.cuts(eq, meta))
                writer(name + '_' + key + '_DAILY.json',
                       [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
                writer(name + '_' + key + '_TRADES.json', trades)
                writer(name + '_' + key + '_HOLDINGS.json', holdings)
        return results


class Runner(base.Runner):
    def __init__(self, root, plan_path, plan_sha, backend_factory=Backend):
        self.root, self.plan_path = Path(root).resolve(), Path(plan_path).resolve()
        self.plan_sha = plan_sha
        require(sha(self.plan_path) == plan_sha, 'Frozen PLAN changed')
        self.plan = plan = read(self.plan_path)
        require(plan['round'] == 156 and plan['status'] == 'REGISTERED'
                and plan['arms'] == list(NAMES) and plan['primary_challenger'] == NAMES[3],
                'Round156 registration required before actual calculation')
        require(plan['dose'] == DOSE and plan['tolerances'] == TOLERANCES, 'Declared numerical policy')
        require(plan['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                'Pinned runtime changed')
        for key in ('actual_feature_calculations_before_registration',
                    'actual_returns_before_registration', 'actual_model_fits_before_registration'):
            require(type(plan.get(key)) is int and plan[key] == 0, 'Unexpected preregistration activity')
        self.pins = self.verify(plan['hashes'])
        require(set(plan['inputs']) == set(INPUTS), 'Exact named inputs required')
        self.inputs = {key: self.bound_path(ref['path']) for key, ref in plan['inputs'].items()}
        require(all(sha(self.inputs[key]) == ref['sha256'] for key, ref in plan['inputs'].items()),
                'Input binding changed')
        for key, (path, expected) in FIXED_INPUTS.items():
            require(self.inputs[key] == self.root / path and sha(self.inputs[key]) == expected,
                    'Fixed source plan, original axis or calendar changed')
        validate_proposal(read(self.inputs['proposal']))
        for path, expected in source_files(self.root).items():
            require(self.pins.get(path) == expected, 'Current implementation not registered')
        self.out = self.root / OUT
        require(self.plan_path == self.out / 'PLAN.json' and Path(plan['output_directory']) == self.out,
                'Dedicated new experiment directory required')
        self.backend_factory = backend_factory
        self.require_source_grid()

    def path(self, path):
        raw = Path(path)
        raw = raw if raw.is_absolute() else self.root / raw
        runtime = {Path(sys.executable).resolve(), Path(np.__file__).resolve(), Path(pd.__file__).resolve()}
        require((raw.is_relative_to(self.root) or raw in runtime) and raw.resolve() == raw
                and '..' not in raw.parts and 'CloudStorage' not in raw.parts,
                'Nonlocal or symlink dependency')
        return raw

    def require_source_grid(self):
        """Verify independent closure and gate before parsing any source amounts."""
        manifest_path = self.inputs['source_manifest']
        require(manifest_path == self.root / SOURCE_OUT / 'MANIFEST.json', 'Exact pilot source manifest')
        manifest = read(manifest_path)
        require(manifest.get('source_only') is True and manifest.get('strategy_registration') is False
                and set(manifest['outputs']) == set(SOURCE_FILES), 'Source-only complete manifest')
        source_pins = self.verify(manifest['input_hashes'])
        require(all(self.pins.get(path) == expected for path, expected in source_pins.items()),
                'Unregistered source input')
        files, required = {}, dict(source_pins)
        required[str(manifest_path)] = sha(manifest_path)
        for name, ref in manifest['outputs'].items():
            path = self.bound_path(self.root / SOURCE_OUT / name)
            require(sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes'], 'Changed source output')
            files[name], required[str(path)] = path, ref['sha256']
        review = read(self.inputs['source_grid_review'])
        require(review.get('passed') is True and review.get('actual_grid_verified') is True,
                'Independent actual source-grid review required')
        reviewed = self.verify(review['input_hashes'])
        require(all(reviewed.get(path) == expected for path, expected in required.items()),
                'Independent source review omitted manifest, output or underlying evidence')
        require(all(self.pins.get(path) == expected for path, expected in reviewed.items()),
                'Unregistered independent source review dependency')
        meta, source_plan = read(self.inputs['meta']), read(self.inputs['source_plan'])
        rows = read(files['ROWS.json'])
        gate = coverage_gate(meta, rows, source_plan)
        saved_gate = read(files['COVERAGE_GATE.json'])
        require(gate['source_coverage_floor_passed'] is True
                and all(saved_gate.get(key) == value for key, value in gate.items())
                and saved_gate.get('source_reviews_verified_by_preparation_driver') is True,
                'Unchanged prospective coverage floor must pass before factor calculation')
        calendar = read(self.inputs['source_calendar'])
        implementation = self.bound_path(self.root / 'outputs/monthly_earnings_repair/research_calendar.py')
        module = base.module('revenue156_source_calendar', implementation)
        schedule = module.market_calendar(calendar['start'], calendar['end']).schedule
        dates = [str(day.date()) for day in schedule.index]
        require(dates == calendar['sessions'], 'Frozen calendar session axis changed')
        cutoffs = {day: int(close.value) for day, close in zip(dates, schedule['close'])}
        verify_cutoffs(meta, rows, cutoffs)
        reports, edges = read(files['REPORTS.json']), read(files['EDGES.json'])
        for catalog in reports.values():
            for report in catalog:
                for role in ('raw', 'receipt'):
                    ref = report['source'][role]
                    path = self.bound_path(ref['path'])
                    require(source_pins.get(str(path)) == ref['sha256']
                            and sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes'],
                            'Report source bytes are not in the independently verified closure')
        self.refresh()
        return meta, rows, reports, edges, cutoffs, source_plan

    def prepare(self):
        self.refresh()
        require(not any((self.out / name).exists() for name in (*CACHE_FILES, 'CACHE.json')),
                'Preparation already started; preserve partial outputs')
        source = self.require_source_grid()
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        base.validate_meta(meta)
        require(len(original) == 27 and meta == source[0], 'Exact original cache required')
        parity = backend.price_parity(meta, px, op, self.plan['raw_source_hashes'])
        result = build_features(*source)
        require(not (set(original) & set(result['arrays'])), 'Revenue features may not replace original arrays')
        arrays = dict(original, **result['arrays'])
        orders, scores = panels(arrays, meta, backend, read(self.inputs['policy']))
        require(orders[NAMES[0]] == read(self.inputs['original_orders']), 'Incumbent changed')
        result['audit'].update(revenue_source_grid_verified=True,
            source_manifest_sha256=sha(self.inputs['source_manifest']),
            source_grid_review_sha256=sha(self.inputs['source_grid_review']))
        self.refresh()
        for name, values in (('DATA.npz', arrays), ('SCORES.npz', scores)):
            with (self.out / name).open('xb') as handle:
                np.savez_compressed(handle, **values)
        for name, value in (('META.json', meta), ('PRETRAIN_ORDERS.json', orders),
                            ('PRICE_PARITY.json', parity), ('FEATURE_AUDIT.json', result['audit'])):
            self.write(name, value)
        self.write('CACHE.json', {name: sha(self.out / name) for name in CACHE_FILES})

    def loaded(self):
        self.cache()
        backend = self.backend_factory(self.root, self)
        meta, original, px, op = backend.load()
        require(meta == read(self.out / 'META.json'), 'Original metadata changed')
        with np.load(self.out / 'DATA.npz', allow_pickle=False) as data:
            require(set(data.files) == set(original) | set(HANDS) | {'common_available'}, 'Changed array names')
            require(all(data[key].tobytes() == value.tobytes() and data[key].dtype == value.dtype
                        and data[key].shape == value.shape for key, value in original.items()), 'Original27 changed')
        return backend, meta, px, op

    def require_prepare_review(self):
        self.cache()
        report = read(self.out / 'INDEPENDENT_PREPARE.json')
        require(report.get('passed') is True and report.get('stage') == 'prepare'
                and all(report.get(key) is True for key in CHECKS),
                'Independent preparation checks incomplete')
        require(report.get('producer_modules_imported') is False
                and report.get('returns_or_nav_computed') is False, 'Independent preparation isolation')
        for key, expected in (('orders_checked', 488), ('candidate_cells_checked', 5978),
                              ('months_checked', 122), ('prefixes_checked', 123),
                              ('revenue_feature_matrices_verified', 3), ('new_model_fits', 0)):
            require(type(report.get(key)) is int and report[key] == expected,
                    'Independent preparation scope: ' + key)
        require(report['plan_sha256'] == self.plan_sha
                and report['cache_sha256'] == sha(self.out / 'CACHE.json')
                and report['orders_sha256'] == sha(self.out / 'PRETRAIN_ORDERS.json'), 'Review binding changed')
        reviewed = self.verify(report['artifact_hashes'])
        require(all(reviewed.get(str(self.out / name)) == sha(self.out / name)
                    for name in ('PLAN.json', 'CACHE.json', *CACHE_FILES)), 'Unreviewed prepared artifact')
        require(self.verify(report['source_hashes']) == self.verify(self.plan['independent_prepare_source_hashes']),
                'Independent preparation reviewer changed')

    def select_first88(self):
        self.refresh()
        self.require_prepare_review()
        require(not any((self.out / name).exists() for name in
                ('TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'FROZEN.json')),
                'Selection already started')
        backend, meta, px, op = self.loaded()
        rows, opened, walk = backend.training(read(self.out / 'PRETRAIN_ORDERS.json'), meta, px, op)
        self.write('TRAIN_TRIALS.json', rows)
        self.write('TRAIN_OPEN_TRIALS.json', opened)
        self.write('WALKFORWARD.json', walk)
        self.write('FROZEN.json', dict(selection=select(rows, opened), pristine_oos=False,
            all_diagnostic_orders_frozen=True, hashes={name: sha(self.out / name) for name in FROZEN_FILES}))

    def diagnose_later34(self):
        self.refresh()
        self.require_prepare_review()
        frozen = read(self.out / 'FROZEN.json')
        require(set(frozen['hashes']) == set(FROZEN_FILES), 'Frozen artifact list changed')
        self.verify({str(self.out / name): expected for name, expected in frozen['hashes'].items()})
        require(frozen['selection'] == select(read(self.out / 'TRAIN_TRIALS.json'),
                read(self.out / 'TRAIN_OPEN_TRIALS.json')), 'First88 selection changed')
        require(not (self.out / 'DIAGNOSE_STARTED.json').exists(), 'Diagnostics already started')
        self.write('DIAGNOSE_STARTED.json', dict(plan_sha256=self.plan_sha,
                   frozen_sha256=sha(self.out / 'FROZEN.json'),
                   started_utc=datetime.now(timezone.utc).isoformat()))
        backend, meta, px, op = self.loaded()
        results = backend.diagnostics(read(self.out / 'PRETRAIN_ORDERS.json'), meta, px, op, self.write)
        for filename, mode in (('TRAIN_TRIALS.json', 'same_close'), ('TRAIN_OPEN_TRIALS.json', 'next_open_cost')):
            for row in read(self.out / filename):
                require(abs(results[row['name']][mode]['slices']['training2016_2023']['cagr'] -
                            row['training']['cagr']) < TOLERANCES['training_cagr'], 'Training replay parity')
        self.write('RESULTS.json', results)
        self.write('VALIDATION.json', dict(selection=frozen['selection'], no_observer_writes=True,
            no_intramonth_trades=True, new_model_fits=0, independent_outcome_review_pending=True,
            pristine_oos=False))


def source_files(root):
    root = Path(root)
    paths = [Path(__file__), Path(factor.__file__),
        root / 'work/test_monthly_revenue_persistence156_runner.py',
        root / 'work/test_after155_revenue_pilot_factor.py',
        root / 'work/after155_revenue_coverage_gate.py', root / 'work/after155_multi_issuer_chain.py',
        root / 'work/after155_revenue_chain.py',
        root / 'outputs/monthly_earnings_repair/research_calendar.py']
    return dict(prior.source_files(root), **{str(path.resolve()): sha(path) for path in paths})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'select_first88', 'diagnose_later34'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    getattr(Runner(args.root, args.plan, args.plan_sha), args.phase)()
