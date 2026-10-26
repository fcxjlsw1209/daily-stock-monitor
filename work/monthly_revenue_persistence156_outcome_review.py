"""Independent revenue-pilot accounting audit, using frozen accounting helpers.

No producer factor, scoring runner, or lifecycle module is imported. Actual
source/price arrays are read only after registered preparation and diagnostics.
"""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import unittest
from io import StringIO
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'work/monthly_same_report145_outcome_review.py'
if hashlib.sha256(OLD.read_bytes()).hexdigest() != '968038244802fa5ea18253adbc99786979bd5e40890c96d04f859721ed728d8e':
    raise ValueError('Frozen independent accounting implementation changed')
import monthly_same_report145_outcome_review as old

require, h, Reader = old.require, old.h, old.Reader
replay, compare_stats, normalize, rr_statistics = old.replay, old.compare_stats, old.normalize, old.rr_statistics
NAMES = ('incumbent_local67', 'matched_latest_revenue_growth',
         'matched_annualized_total_revenue_growth', 'weakest_annual_revenue_growth')
HANDS = ('latest_revenue_growth', 'annualized_total_revenue_growth', 'weakest_annual_revenue_growth')
STUDY = 'outputs/monthly_revenue_persistence156'
SOURCE = 'outputs/monthly_revenue_pilot_source156'
PRICE, PATHS, CACHE_FILES = old.PRICE, old.PATHS, old.CACHE_FILES
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'sessions',
          'protected_manifest', 'source_manifest', 'source_grid_review', 'source_plan', 'source_calendar')
SOURCE_FILES = ('ROWS.json', 'REPORTS.json', 'EDGES.json', 'COVERAGE_GATE.json', 'SUMMARY.json')
USABLE = 'checked_reported_total_chain_with_declared_source_scope'
SOURCE_CHECKS = ('revenue_source_grid_verified', 'all_three_revenue_features_common_mask', 'original_source_cutoffs_verified')
PREPARE_CHECKS = ('original27_arrays_byte_identical', 'all_features_independently_reconstructed',
    'all_four_score_matrices_and_ordered_choices_independently_verified',
    'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback',
    'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
    'all_signal_prefixes_verified', *SOURCE_CHECKS)


def source_files(root=ROOT):
    root = Path(root).resolve()
    return [root/'work/monthly_revenue_persistence156_outcome_review.py',
            root/'work/test_monthly_revenue_persistence156_outcome_review.py',
            root/'work/monthly_same_report145_outcome_review.py',
            root/'work/monthly_ridge_support70_result_review.py']


def count(doc, key, expected):
    require(type(doc.get(key)) is int and doc[key] == expected, 'Exact count required: '+key)


def select(rows, opened):
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    def check(group, at_open):
        require(type(group) is list and [r.get('name') for r in group] == list(NAMES), 'Four exact selection arms')
        values = [[r.get(a, {}).get(b) for a, b in fields] for r in group]
        answer = {}
        for i in range(3):
            a, b = values[3], values[i]
            if not all(type(x) in (int, float) and math.isfinite(x) for x in a+b):
                answer[NAMES[i]] = False
            else:
                growth = a[0] >= b[0]-1e-12 if at_open and i == 0 else a[0] > b[0]+1e-12
                answer[NAMES[i]] = bool(growth and all(x >= y-t for x, y, t in zip(a[1:], b[1:], (.05, .03, .025, .025))))
        return answer
    close, opening = check(rows, False), check(opened, True)
    admitted = all(close.values()) and all(opening.values())
    return dict(primary=NAMES[3] if admitted else NAMES[0], admitted=admitted,
                guards_against=close, next_open_guards_against=opening, mechanism_confirmed=all(opening.values()))


def validate_orders(close, opening, orders, meta):
    dates = old.daily_axis(close.index, 'prices')
    require(opening.index.equals(dates) and opening.columns.equals(close.columns) and close.columns.is_unique,
            'Same price axes required')
    days, ends = old.daily_axis(meta['dates'], 'signals'), pd.DatetimeIndex(meta['ends'])
    require(len(days) == len(ends) == len(meta['symbols']) == 122 and days.isin(dates).all() and ends.isin(dates).all()
            and (ends > days).all() and days[1:].equals(ends[:-1]), '122 adjacent holding periods required')
    require(set(orders) == set(NAMES), 'Four exact order arms required')
    for name in NAMES:
        require(len(orders[name]) == 122, '122 orders per arm required')
        for i, row in enumerate(orders[name]):
            require(set(row) == {'date', 'end', 'signal_date', 'symbols', 'scores'}, 'Exact order fields required')
            require(row['date'] == row['signal_date'] == meta['dates'][i] and row['end'] == meta['ends'][i], 'Order period')
            symbols = row['symbols']
            require(type(symbols) is list and len(symbols) == len(set(symbols)) and len(symbols) in (0, 2, 3), 'Cash or two/three positions')
            require(all(type(s) is str and s in meta['symbols'][i] and s in close.columns for s in symbols), 'Original securities')
            require(len(row['scores']) == len(symbols) and all(type(x) in (int, float) and math.isfinite(x) for x in row['scores']),
                    'Aligned finite order scores')
            require(dates.get_loc(days[i])+1 < len(dates), 'Next session exists')
    return dates[(dates >= days[0]) & (dates <= ends[-1])]


def walkforward(dates, navs, opens, meta):
    folds, chosen, controls = [], [], []
    for n, end in ((36, 60), (60, 76), (76, 88)):
        opening = [dict(name=name, **h.measures(dates, opens[name], meta, n)) for name in NAMES]
        rows = [dict(name=name, **h.measures(dates, navs[name], meta, n)) for name in NAMES]
        pick = select(rows, opening)
        folds.append(dict(n=n, end=end, selection=pick, open_training=opening))
        lo, hi = dates.get_loc(pd.Timestamp(meta['dates'][n])), dates.get_loc(pd.Timestamp(meta['ends'][end-1]))
        for target, name in ((chosen, pick['primary']), (controls, NAMES[0])):
            target.extend((navs[name][1:]/navs[name][:-1]-1.)[lo:hi])
    return dict(folds=folds, pristine_oos=False,
                metrics=rr_statistics(chosen, meta['dates'][36], meta['ends'][87]),
                incumbent=rr_statistics(controls, meta['dates'][36], meta['ends'][87]))


def independent_features(levels):
    require(len(levels) == 4 and all(type(v) is str for v in levels), 'Four exact original level strings')
    values = [Decimal(v) for v in levels]
    require(all(v.is_finite() and v > 0 for v in values), 'Positive original levels')
    fractions = [Fraction(v) for v in values]
    rates = [b/a-1 for a, b in zip(fractions, fractions[1:])]
    with localcontext() as ctx:
        ctx.prec = 80
        ratio = values[3]/values[0]
        # Newton iteration is independent of the producer's fractional power.
        root = Decimal(10) ** (ratio.adjusted()//3)
        for _ in range(128):
            if abs(root**3-ratio) <= Decimal('1e-65')*ratio:
                break
            root = (2*root+ratio/(root*root))/3
        else:
            raise ValueError('Independent cube-root iteration did not converge within128 steps')
        growth = root-1
        geometric = float(growth/(1+abs(growth)))
    bounded = lambda v: float(v/(1+abs(v)))
    return dict(zip(HANDS, (bounded(rates[-1]), geometric, bounded(min(rates)))))


def verify_features(arrays, feature, meta, source_rows, reports):
    require(len(source_rows) == len(feature['cells']) == 5978, 'Complete source and feature grids required')
    count(feature, 'new_model_fits', 0)
    require(all(feature.get(k) is True for k in (*SOURCE_CHECKS, 'source_holds_preserved')), 'Feature source checks missing')
    require(feature.get('financial_source_scope') == ['MSFT', 'AAPL', 'ABBV'], 'Fixed financial scope')
    indexes = {s: {r['report_id']: r for r in rows} for s, rows in reports.items()}
    want = {k: np.full((122, 49), np.nan) for k in HANDS}
    want['common_available'] = np.zeros((122, 49), dtype=bool)
    available = 0
    for n, (source, cell) in enumerate(zip(source_rows, feature['cells'])):
        i, j = divmod(n, 49)
        identity = (i, j, meta['dates'][i], meta['symbols'][i][j])
        keys = ('signal_index', 'candidate_index', 'decision_date', 'symbol')
        require(tuple(source[k] for k in keys) == tuple(cell[k] for k in keys) == identity, 'Complete original slot identities')
        usable = source['status'] == USABLE
        require(cell['available'] is usable and cell['status'] == source['status'] and cell['reasons'] == source['reasons'],
                'Feature source status changed')
        if not usable:
            require('features' not in cell and 'report_ids' not in cell, 'Unavailable row must retain missing features')
            continue
        available += 1
        require(source['symbol'] not in meta['etfs'] and source['symbol'] in ['MSFT', 'ABBV'], 'Financial scope/ETF hold')
        ids = source['report_ids']
        require(len(ids) == len(set(ids)) == 4 and cell['report_ids'] == ids and cell['edge_ids'] == source['edge_ids']
                and cell['latest_report_id'] == source['latest_report_id']
                and cell['decision_cutoff_ns'] == source['decision_cutoff_ns'], 'Original chain provenance')
        originals = [indexes[source['symbol']][rid] for rid in ids]
        years = [r['fiscal_year'] for r in originals]
        require(years == list(range(years[0], years[0]+4)) and len({r['scale'] for r in originals}) == 1
                and all(r['known_ns'] <= source['decision_cutoff_ns'] for r in originals), 'Consecutive known same-scale originals')
        values = independent_features([r['current'] for r in originals])
        require(set(cell['features']) == set(HANDS), 'Exactly three financial features')
        for key in HANDS:
            require(type(cell['features'][key]) in (int, float) and math.isfinite(cell['features'][key]), 'Finite feature')
            require(abs(cell['features'][key]-values[key]) <= 1e-12, 'Independent original-level factor mismatch')
            want[key][i, j] = values[key]
        want['common_available'][i, j] = True
    count(feature, 'usable_rows', available); count(feature, 'rows', 5978)
    for key, values in want.items():
        require(arrays[key].shape == values.shape and arrays[key].dtype == values.dtype, 'Exact feature matrix schema')
        if key == 'common_available':
            np.testing.assert_array_equal(arrays[key], values)
        else:
            np.testing.assert_allclose(arrays[key], values, atol=1e-12, rtol=0, equal_nan=True)
    return available


def check_outcomes(close, opening, orders, meta, read):
    dates = validate_orders(close, opening, orders, meta)
    results = read('RESULTS.json'); require(set(results) == set(NAMES), 'Four result arms')
    bounds = dict(early2016_2020=(meta['dates'][0], meta['ends'][52]), original2021_2026=(meta['dates'][53], meta['ends'][-1]),
                  training2016_2023=(meta['dates'][0], meta['ends'][87]), later34=(meta['dates'][88], meta['ends'][-1]))
    errors, navs, opens, metrics, details, maxnav = [], {}, {}, {}, {}, 0.
    for name in NAMES:
        require(set(results[name]) == {x[2] for x in PATHS}, 'Three execution paths per arm')
        for execution, cost, key in PATHS:
            met, nav, trades, holdings = replay(close, opening, orders[name], dates, execution, cost)
            prefix = name+'_'+key
            compare_stats(results[name][key], dict(metrics=met, slices={k: h.slice_stat(dates, nav, *span) for k, span in bounds.items()}), errors, prefix)
            daily = read(prefix+'_DAILY.json')
            require(all(set(x) == {'date', 'equity'} for x in daily) and [x['date'] for x in daily] == [str(d.date()) for d in dates], 'Daily NAV axis')
            actual = np.asarray([x['equity'] for x in daily], dtype=float)
            require(np.isfinite(actual).all() and (actual > 0).all(), 'Finite positive NAV')
            np.testing.assert_allclose(actual, nav, atol=1e-5, rtol=2e-11)
            maxnav = max(maxnav, float(np.max(np.abs(actual-nav))))
            actual_trades, actual_holdings = read(prefix+'_TRADES.json'), read(prefix+'_HOLDINGS.json')
            h.compare(normalize(actual_trades), normalize(trades), errors, prefix+'/trades')
            h.compare(normalize(actual_holdings), normalize(holdings), errors, prefix+'/holdings')
            allowed = {o['date'] if execution == 'close' else str(close.index[close.index.get_loc(pd.Timestamp(o['date']))+1].date()) for o in orders[name]}
            require(all(t['date'] in allowed for t in actual_trades), 'Intramonth trade')
            require(all(r['partial'] is (r['year'] in (2016, 2026)) for r in met['annual_returns']), 'Annual partial flags')
            details[prefix] = dict(trades=len(trades), holding_periods=len(holdings), fees_paid=met['fees_paid'])
            if execution == 'close': navs[name], metrics[name] = nav, met
            if key == 'next_open_cost': opens[name] = nav
        prefix_dates = dates[dates <= pd.Timestamp(meta['ends'][87])]
        for execution, cost, full in (('close', 0., navs[name]), ('open', .0005, opens[name])):
            _, short, _, _ = replay(close.loc[:prefix_dates[-1]], opening.loc[:prefix_dates[-1]], orders[name][:88], prefix_dates, execution, cost)
            np.testing.assert_allclose(short, full[:len(prefix_dates)], atol=1e-5, rtol=2e-11)
    rows = [dict(name=name, **h.measures(dates, navs[name], meta)) for name in NAMES]
    opened = [dict(name=name, **h.measures(dates, opens[name], meta)) for name in NAMES]
    compare_stats(read('TRAIN_TRIALS.json'), rows, errors, 'training')
    compare_stats(read('TRAIN_OPEN_TRIALS.json'), opened, errors, 'opening_training')
    pick = select(rows, opened); h.assert_exact_tree(read('FROZEN.json')['selection'], pick)
    compare_stats(read('WALKFORWARD.json'), walkforward(dates, navs, opens, meta), errors, 'walkforward')
    validation = read('VALIDATION.json'); h.assert_exact_tree(validation['selection'], pick)
    require(validation.get('pristine_oos') is False and validation.get('independent_outcome_review_pending') is True
            and validation.get('no_observer_writes') is True and validation.get('no_intramonth_trades') is True, 'Scope claims')
    count(validation, 'new_model_fits', 0)
    monthly = []
    for i, day in enumerate(meta['dates']):
        held = {n: orders[n][i]['symbols'] for n in NAMES}
        returns = {n: math.fsum(float(close.at[pd.Timestamp(meta['ends'][i]), s])/float(close.at[pd.Timestamp(day), s])-1.
                    for s in held[n])/len(held[n]) if held[n] else 0. for n in NAMES}
        monthly.append(dict(date=day, end=meta['ends'][i], split='early53' if i < 53 else 'recent35' if i < 88 else 'later34',
                            holdings=held, returns=returns, changed_arms=[n for n in NAMES[1:] if held[n] != held[NAMES[0]]]))
    return dict(passed=True, stage='evaluate', execution_paths_verified=12, total_monthly_orders_verified=488,
        daily_nav_trades_holdings_metrics_slices=True, first88_and_three_fold_selection_verified=True,
        first88_causal_replay_verified=True, annual_partial_flags_verified=True, all_changed_months_reported=True,
        zero_model_fit_count_verified=True, models_fit=0, selected_name=pick['primary'],
        primary_cagr=metrics[pick['primary']]['cagr'], primary_target_reached=metrics[pick['primary']]['cagr'] >= .7,
        independent_monthly_comparison=monthly, replay_details=details,
        max_compared_absolute_difference=max((x for _, x in errors), default=0.), max_nav_absolute_difference=maxnav,
        producer_modules_imported=False, protected_member_scan_repeated=False)


def proposal_contract():
    return dict(total_months=122, training_months=88, early_months=53, recent_months=35, later_months=34,
        walkforward=[[36, 60], [60, 76], [76, 88]],
        execution_paths=[dict(execution=x, cost=c, key=k) for x, c, k in PATHS],
        same_close_cagr='strict_greater_than_all_three_controls',
        next_open_cost_cagr='noninferior_incumbent_strict_greater_than_two_controls',
        strict_cagr_epsilon=1e-12, max_sharpe_degradation=.05, max_drawdown_degradation=.03,
        max_early_cagr_degradation=.025, max_recent_cagr_degradation=.025,
        new_model_fits=0, intramonth_trades=False, later34_for_selection=False)


def verify_source_grid(reader, plan, pins):
    inputs = plan['inputs']
    require(reader.path(inputs['source_manifest']['path']) == reader.root/SOURCE/'MANIFEST.json', 'Canonical source package')
    manifest = reader.json(inputs['source_manifest']['path'])
    require(manifest.get('source_only') is True and manifest.get('strategy_registration') is False and
            set(manifest['outputs']) == set(SOURCE_FILES), 'Complete source-only manifest')
    required = reader.verify(manifest['input_hashes'])
    required[str(reader.root/SOURCE/'MANIFEST.json')] = reader.sha(reader.root/SOURCE/'MANIFEST.json')
    for name, ref in manifest['outputs'].items():
        p = reader.root/SOURCE/name
        require(reader.sha(p) == ref['sha256'] and p.stat().st_size == ref['bytes'], 'Source output digest/size')
        required[str(p)] = ref['sha256']
    review = reader.json(inputs['source_grid_review']['path'])
    require(review.get('passed') is True and review.get('actual_grid_verified') is True, 'Independent actual grid review')
    checked = reader.verify(review['input_hashes'])
    require(all(checked.get(p) == v and pins.get(p) == v for p, v in required.items()), 'Source evidence closure not registered/reviewed')
    require(all(pins.get(p) == v for p, v in checked.items()), 'Source review dependency not registered')
    gate = reader.json(reader.root/SOURCE/'COVERAGE_GATE.json')
    require(gate.get('source_coverage_floor_passed') is True and gate.get('source_reviews_verified_by_preparation_driver') is True,
            'Predeclared source floor required')
    return reader.json(reader.root/SOURCE/'ROWS.json'), reader.json(reader.root/SOURCE/'REPORTS.json')


def verify_diagnostic_start(reader, out, plan, plan_sha, started):
    marker = reader.json(out/'DIAGNOSE_STARTED.json')
    require(set(marker) == {'plan_sha256', 'frozen_sha256', 'started_utc'} and marker['plan_sha256'] == plan_sha
            and marker['frozen_sha256'] == reader.sha(out/'FROZEN.json'), 'Diagnostic marker binding')
    def utc(value):
        require(type(value) is str and value.endswith(('Z', '+00:00')), 'UTC timestamp required')
        return datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    require(utc(plan['registered_utc']) < utc(marker['started_utc']) < started, 'Registration/diagnostics/audit chronology')


def bindings(reader, plan_sha, started):
    require(type(plan_sha) is str and len(plan_sha) == 64 and all(c in '0123456789abcdef' for c in plan_sha), 'Exact PLAN SHA required')
    out = reader.root/STUDY
    require(reader.sha(out/'PLAN.json') == plan_sha, 'Registered PLAN changed')
    plan = reader.json(out/'PLAN.json')
    require(plan.get('round') == 156 and plan.get('status') == 'REGISTERED' and plan.get('arms') == list(NAMES)
            and plan.get('primary_challenger') == NAMES[3] and plan.get('dose') == .01, 'Fixed registered four-arm experiment')
    require(reader.path(plan['output_directory']) == out and plan['runtime'] ==
            dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Output/runtime changed')
    for key in ('actual_feature_calculations_before_registration', 'actual_returns_before_registration',
                'actual_model_fits_before_registration', 'new_market_downloads'):
        count(plan, key, 0)
    require(plan.get('no_observer_writes') is True, 'Observer protection required')
    pins = reader.verify(plan['hashes'])
    sources = {str(p): reader.sha(p) for p in source_files(reader.root)}
    require(reader.verify(plan['independent_outcome_source_hashes']) == sources, 'Registered independent audit implementation')
    require(all(pins.get(p) == v for p, v in sources.items()), 'Audit source not registered')
    require(set(plan['inputs']) == set(INPUTS), 'Exact named input set')
    for ref in plan['inputs'].values():
        require(reader.sha(ref['path']) == ref['sha256'] == pins.get(str(reader.path(ref['path']))), 'Named input changed')
    proposal = reader.json(plan['inputs']['proposal']['path'])
    require(proposal.get('round') == 156 and proposal.get('arms') == list(NAMES) and
            proposal.get('primary_challenger') == NAMES[3] and proposal.get('dose') == .01 and
            proposal.get('selection_contract') == proposal_contract(), 'Registered proposal contract')
    for key in ('actual_new_factors', 'actual_new_orders', 'actual_new_returns'):
        count(proposal, key, 0)
    source_rows, reports = verify_source_grid(reader, plan, pins)
    require(reader.path(plan['inputs']['prices']['path']) == reader.root/PRICE and
            reader.path(plan['inputs']['original_orders']['path']) == reader.root/'outputs/monthly_local_linear70/local_linear_ORDERS.json',
            'Original price/order inputs required')
    require(len(plan['raw_source_hashes']) == 64, 'Original64 raw price sources')
    require(all(pins.get(p) == v for p, v in reader.verify(plan['raw_source_hashes']).items()), 'Raw price source closure')
    cache = reader.json(out/'CACHE.json'); require(set(cache) == set(CACHE_FILES), 'Exact preparation cache')
    cached = reader.verify(cache, out)
    prep = reader.json(out/'INDEPENDENT_PREPARE.json')
    require(prep.get('passed') is True and prep.get('stage') == 'prepare' and all(prep.get(k) is True for k in PREPARE_CHECKS),
            'Completed independent preparation required')
    for key, value in [('orders_checked', 488), ('candidate_cells_checked', 5978), ('months_checked', 122),
                       ('prefixes_checked', 123), ('revenue_feature_matrices_verified', 3), ('new_model_fits', 0)]:
        count(prep, key, value)
    require(prep.get('producer_modules_imported') is False and prep.get('returns_or_nav_computed') is False, 'Preparation independence')
    for key, name in [('plan_sha256', 'PLAN.json'), ('cache_sha256', 'CACHE.json'), ('orders_sha256', 'PRETRAIN_ORDERS.json')]:
        require(prep.get(key) == reader.sha(out/name), 'Preparation receipt binding')
    require(reader.verify(prep['source_hashes']) == reader.verify(plan['independent_prepare_source_hashes']), 'Preparation audit code')
    artifacts = reader.verify(prep['artifact_hashes'])
    needed = dict(cached, **{str(out/name): reader.sha(out/name) for name in ('PLAN.json', 'CACHE.json')})
    require(all(artifacts.get(p) == v for p, v in needed.items()), 'Complete independently prepared artifact closure')
    frozen = reader.json(out/'FROZEN.json')
    require(set(frozen['hashes']) == set(FROZEN_FILES), 'Exact first88 frozen scope')
    reader.verify(frozen['hashes'], out)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True, 'Diagnostic orders frozen before replay')
    verify_diagnostic_start(reader, out, plan, plan_sha, started)
    orders = reader.json(out/'PRETRAIN_ORDERS.json')
    h.assert_exact_tree(orders[NAMES[0]], reader.json(plan['inputs']['original_orders']['path']))
    return out, plan, source_rows, reports


def audit(root, plan_sha):
    started = datetime.now(timezone.utc); reader = Reader(root)
    out, plan, source_rows, reports = bindings(reader, plan_sha, started)
    meta, orders = reader.json(out/'META.json'), reader.json(out/'PRETRAIN_ORDERS.json')
    require(meta == reader.json(plan['inputs']['meta']['path']) and meta['dates'][0] == '2016-08-17'
            and meta['ends'][-1] == '2026-09-24', 'Original122 metadata/horizon')
    feature = reader.json(out/'FEATURE_AUDIT.json')
    for field, key in [('source_manifest_sha256', 'source_manifest'), ('source_grid_review_sha256', 'source_grid_review')]:
        require(feature.get(field) == plan['inputs'][key]['sha256'], 'Feature source audit binding')
    arrays = reader.npz(out/'DATA.npz')
    available = verify_features(arrays, feature, meta, source_rows, reports)
    for i in range(122):
        for name, key in zip(NAMES[1:], HANDS):
            if not np.any(np.isfinite(arrays[key][i]) & (arrays[key][i] != 0.)):
                h.assert_exact_tree(orders[name][i], orders[NAMES[0]][i])
    data = reader.npz(PRICE); require(set(data) == {'dates', 'symbols', 'close', 'opening'}, 'Original price payload')
    dates = pd.to_datetime(data['dates'])
    close = pd.DataFrame(data['close'], index=dates, columns=data['symbols'])
    opening = pd.DataFrame(data['opening'], index=dates, columns=data['symbols'])
    report = check_outcomes(close, opening, orders, meta, lambda name: reader.json(out/name))
    report.update(schema_version=1, plan_sha256=plan_sha, proposal_sha256=plan['inputs']['proposal']['sha256'],
        cache_sha256=reader.sha(out/'CACHE.json'), orders_sha256=reader.sha(out/'PRETRAIN_ORDERS.json'),
        frozen_sha256=reader.sha(out/'FROZEN.json'), diagnose_started_sha256=reader.sha(out/'DIAGNOSE_STARTED.json'),
        audit_started_utc=started.isoformat(), diagnostic_start_marker_verified=True,
        baseline_local67_full_orders_unchanged=True, revenue_source_grid_verified=True,
        all_three_revenue_features_common_mask=True, original_source_cutoffs_verified=True,
        revenue_feature_matrices_verified=3, available_feature_cells=available,
        source_hashes={str(p): reader.sha(p) for p in source_files(reader.root)},
        limitations=['Repeated historical sample; not pristine out of sample.',
                     'Bounded source-grid admission and declared reported-dollar/archival assumptions remain.',
                     'Protected inventory verification is performed separately by formal closure.'])
    for path, expected in list(reader.seen.items()):
        require(reader.sha(path) == expected, 'Post-audit source/artifact change')
    report['artifact_hashes'] = dict(reader.seen)
    return report


def prereview(root=ROOT):
    """Run only generated-input tests and bind the final prospective contract."""
    root = Path(root).resolve()
    import test_monthly_revenue_persistence156_outcome_review as test_module
    log = StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    require(result.wasSuccessful() and result.testsRun > 0 and not result.skipped, 'Synthetic outcome tests failed: '+log.getvalue())
    reader = Reader(root)
    proposal = 'work/monthly_revenue_persistence156_proposal.json'
    proposal_doc = reader.json(proposal)
    require(proposal_doc.get('round') == 156 and proposal_doc.get('arms') == list(NAMES)
            and proposal_doc.get('selection_contract') == proposal_contract(), 'Final prospective contract mismatch')
    implementations = ['work/'+name for name in ('after155_revenue_pilot_factor.py',
        'monthly_revenue_persistence156_runner.py', 'monthly_revenue_persistence156_lifecycle.py',
        'prepare_revenue_persistence156_proposal.py')]
    inputs = [proposal, 'work/after155_revenue_experiment_path.json', 'work/after155_revenue_pilot_source_plan.json',
              SOURCE+'/MANIFEST.json']
    report = dict(passed=True, round=156, stage='preregistration_outcome_review', synthetic_only=True,
        proposal_sha256=reader.sha(proposal), actual_factors_calculated=0, actual_returns_calculated=0,
        actual_orders_calculated=0, actual_model_fits=0, producer_modules_imported=False,
        tests=result.testsRun, test_log=log.getvalue(),
        source_hashes={str(p): reader.sha(p) for p in source_files(root)},
        implementation_hashes={str(reader.path(p)): reader.sha(p) for p in implementations},
        input_hashes={str(reader.path(p)): reader.sha(p) for p in inputs},
        expected_execution_paths=12, expected_monthly_orders=488,
        reviewer_scope='Independent accounting/original-level formulas and generated-input tamper tests; root manually reviewed this module before this receipt.',
        actual_audit_pending=True)
    for p, expected in list(reader.seen.items()): require(reader.sha(p) == expected, 'Post-prereview binding')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--evaluate-complete', action='store_true'); choice.add_argument('--prereview', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = prereview(args.root) if args.prereview else audit(args.root, args.plan_sha)
    destination = (args.root/'work/monthly_revenue_persistence156_outcome_prereview.json' if args.prereview
                   else args.root/STUDY/'INDEPENDENT_OUTCOME.json')
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(destination), execution_paths=12)))


if __name__ == '__main__':
    main()
