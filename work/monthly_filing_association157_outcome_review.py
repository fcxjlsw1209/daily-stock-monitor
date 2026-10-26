"""Independent filing-association accounting audit, using frozen accounting helpers.

No producer factor, scoring runner, or lifecycle module is imported. Actual
source/price arrays are read only after registered preparation and diagnostics.
"""
from datetime import datetime, timezone
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
import monthly_filing_association157_prepare_review as preparation
from collections import Counter

require, h, Reader = old.require, old.h, old.Reader
replay, compare_stats, normalize, rr_statistics = old.replay, old.compare_stats, old.normalize, old.rr_statistics
HANDS = ('whole_formation_beta_residual_momentum', 'filing_window_density',
         'filing_vs_nonfiling_beta_residual_contrast')
NAMES = ('incumbent_local67', *HANDS)
STUDY = 'outputs/monthly_filing_association157'
SOURCE = 'outputs/monthly_filing_association157_source'
PRICE, PATHS, CACHE_FILES = old.PRICE, old.PATHS, old.CACHE_FILES
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'INDEPENDENT_PREPARE.json')
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices',
          'protected_manifest', 'source_manifest', 'source_grid_review', 'source_plan', 'source_calendar')
CLOCK = 'second_XNYS_session_strictly_after_filingDate_v1'
COUNTERS = ('nuisance_beta_estimates_attempted', 'nuisance_beta_estimates_completed', 'new_predictive_model_fits')
PREPARE_CHECKS = (*preparation.CHECKS, 'raw_prices_independently_verified',
                  'calendar_overlay_independently_verified', 'all_signal_prefixes_verified')
FORBIDDEN = ('monthly_filing_association157_runner', 'monthly_filing_association157_lifecycle',
             'after156_association_measure')


def source_files(root=ROOT):
    root = Path(root).resolve()
    return sorted(set(preparation.source_files(root) + [root/'work'/name for name in (
        'monthly_filing_association157_outcome_review.py',
        'test_monthly_filing_association157_outcome_review.py',
        'monthly_same_report145_outcome_review.py', 'monthly_ridge_support70_result_review.py')]))


def independent_imports():
    require(not any(name in sys.modules for name in FORBIDDEN), 'New producer modules must not be imported')


def counters(doc):
    require('new_model_fits' not in doc, 'Unqualified fit count conceals nuisance estimation')
    values = {key: doc.get(key) for key in COUNTERS}
    require(all(type(value) is int and value >= 0 for value in values.values()), 'Exact nonnegative estimator counts')
    require(values[COUNTERS[2]] == 0 and values[COUNTERS[1]] <= values[COUNTERS[0]] <= 3904, 'Estimator count bounds')
    return values

def count(doc, key, expected):
    require(type(doc.get(key)) is int and doc[key] == expected, 'Exact count required: '+key)


def check_prepare_scope(prep, feature):
    require(prep.get('passed') is True and prep.get('stage') == 'prepare'
            and all(prep.get(k) is True for k in PREPARE_CHECKS), 'Completed independent preparation required')
    for key, value in [('orders_checked', 488), ('candidate_cells_checked', 5978), ('months_checked', 122),
                       ('prefixes_checked', 123), ('filing_feature_matrices_verified', 3), ('new_predictive_model_fits', 0)]:
        count(prep, key, value)
    require(prep.get('producer_modules_imported') is False and prep.get('strategy_returns_or_nav_computed') is False,
            'Preparation independence')
    require(counters(prep) == counters(feature), 'Prepared nuisance counts changed')
    require(prep.get('exact_acceptance_certified') is False, 'Preparation cannot certify exact acceptance')


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


def verify_features(arrays, feature, meta, source_rows, sessions, prices, cutoffs):
    require(len(meta['dates']) == 122 and len(source_rows) == len(feature['cells']) == 5978,
            'Complete original source and feature grids required')
    expected = preparation.features_oracle(meta, source_rows, sessions, prices, cutoffs, numeric_cache={})
    for key in (*HANDS, 'common_available'):
        wanted = expected['arrays'][key]
        require(arrays[key].shape == wanted.shape and arrays[key].dtype == wanted.dtype, 'Feature array schema')
        if key == 'common_available':
            np.testing.assert_array_equal(arrays[key], wanted)
        else:
            np.testing.assert_allclose(arrays[key], wanted, atol=1e-12, rtol=0, equal_nan=True)
    require(counters(feature) == counters(expected), 'Independent beta estimate counts')
    preparation.old.compare(feature['cells'], expected['cells'], 'filing-cells')
    available = int(expected['arrays']['common_available'].sum())
    count(feature, 'rows', 5978); count(feature, 'usable_rows', available)
    require(feature.get('clock_policy_id') == CLOCK and feature.get('exact_acceptance_certified') is False,
            'Conditional filing-date interpretation required')
    require(feature.get('strategy_returns_or_nav_computed') is False
            and feature.get('original_eligibility_unchanged') is True
            and feature.get('filing_source_grid_verified') is True, 'Feature scope checks')
    gate = preparation.coverage_oracle(meta, source_rows, expected['arrays']['common_available'])
    require(gate['coverage_floor_passed'], 'Post-price common-mask coverage floor failed')
    h.assert_exact_tree(feature['common_coverage'], gate)
    require(feature['missing_reason_counts'] ==
            dict(Counter(c['reason'] for c in expected['cells'] if not c['available'])), 'All missing reasons counted')
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
    estimate_counts = counters(read('FEATURE_AUDIT.json'))
    require(counters(validation) == estimate_counts, 'Validation nuisance counts changed')
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
        nuisance_estimate_counts_verified=True, **estimate_counts, selected_name=pick['primary'],
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
        new_predictive_model_fits=0,
        nuisance_estimates='preformation_beta_attempts_and_completions_counted_separately',
        intramonth_trades=False, later34_for_selection=False)


def verify_source_grid(reader, plan, pins):
    return preparation.source_inputs(reader, plan, pins)

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
    require(plan.get('round') == 157 and plan.get('status') == 'REGISTERED' and plan.get('arms') == list(NAMES)
            and plan.get('primary_challenger') == NAMES[3] and plan.get('dose') == .01, 'Fixed registered four-arm experiment')
    require(reader.path(plan['output_directory']) == out and plan['runtime'] ==
            dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Output/runtime changed')
    for key in ('actual_feature_calculations_before_registration', 'actual_returns_before_registration',
                'actual_model_fits_before_registration', 'actual_nuisance_estimates_before_registration', 'new_market_downloads'):
        count(plan, key, 0)
    require(plan.get('no_observer_writes') is True and plan.get('exact_acceptance_certified') is False,
            'Observer protection and conditional source interpretation required')
    pins = reader.verify(plan['hashes'])
    sources = {str(p): reader.sha(p) for p in source_files(reader.root)}
    require(reader.verify(plan['independent_outcome_source_hashes']) == sources, 'Registered independent audit implementation')
    require(all(pins.get(p) == v for p, v in sources.items()), 'Audit source not registered')
    require(set(plan['inputs']) == set(INPUTS), 'Exact named input set')
    for ref in plan['inputs'].values():
        require(reader.sha(ref['path']) == ref['sha256'] == pins.get(str(reader.path(ref['path']))), 'Named input changed')
    proposal = reader.json(plan['inputs']['proposal']['path'])
    require(proposal.get('round') == 157 and proposal.get('arms') == list(NAMES) and
            proposal.get('primary_challenger') == NAMES[3] and proposal.get('dose') == .01 and
            proposal.get('selection_contract') == proposal_contract(), 'Registered proposal contract')
    for key in ('actual_new_factors', 'actual_new_orders', 'actual_new_returns',
                'actual_new_nuisance_estimates', 'actual_new_predictive_model_fits'):
        count(proposal, key, 0)
    source_rows, sessions, cutoffs = verify_source_grid(reader, plan, pins)
    require(reader.path(plan['inputs']['prices']['path']) == reader.root/PRICE and
            reader.path(plan['inputs']['original_orders']['path']) == reader.root/'outputs/monthly_local_linear70/local_linear_ORDERS.json',
            'Original price/order inputs required')
    require(len(plan['raw_source_hashes']) == 64, 'Original64 raw price sources')
    require(all(pins.get(p) == v for p, v in reader.verify(plan['raw_source_hashes']).items()), 'Raw price source closure')
    cache = reader.json(out/'CACHE.json'); require(set(cache) == set(CACHE_FILES), 'Exact preparation cache')
    cached = reader.verify(cache, out)
    prep = reader.json(out/'INDEPENDENT_PREPARE.json')
    check_prepare_scope(prep, reader.json(out/'FEATURE_AUDIT.json'))
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
    return out, plan, source_rows, sessions, cutoffs


def audit(root, plan_sha):
    independent_imports()
    started = datetime.now(timezone.utc); reader = Reader(root)
    out, plan, source_rows, sessions, cutoffs = bindings(reader, plan_sha, started)
    meta, orders = reader.json(out/'META.json'), reader.json(out/'PRETRAIN_ORDERS.json')
    require(meta == reader.json(plan['inputs']['meta']['path']) and meta['dates'][0] == '2016-08-17'
            and meta['ends'][-1] == '2026-09-24', 'Original122 metadata/horizon')
    feature = reader.json(out/'FEATURE_AUDIT.json')
    for field, key in [('source_manifest_sha256', 'source_manifest'), ('source_grid_review_sha256', 'source_grid_review')]:
        require(feature.get(field) == plan['inputs'][key]['sha256'], 'Feature source audit binding')
    # All source, preparation and chronology gates precede numeric price loading.
    arrays = reader.npz(out/'DATA.npz')
    data = reader.npz(PRICE); require(set(data) == {'dates', 'symbols', 'close', 'opening'}, 'Original price payload')
    dates = pd.to_datetime(data['dates'])
    close = pd.DataFrame(data['close'], index=dates, columns=data['symbols'])
    opening = pd.DataFrame(data['opening'], index=dates, columns=data['symbols'])
    available = verify_features(arrays, feature, meta, source_rows, sessions, close, cutoffs)
    for i in range(122):
        for name, key in zip(NAMES[1:], HANDS):
            if not np.any(np.isfinite(arrays[key][i]) & (arrays[key][i] != 0.)):
                h.assert_exact_tree(orders[name][i], orders[NAMES[0]][i])
    report = check_outcomes(close, opening, orders, meta, lambda name: reader.json(out/name))
    require(counters(report) == counters(feature), 'Replayed nuisance counts changed')
    report.update(schema_version=1, plan_sha256=plan_sha, proposal_sha256=plan['inputs']['proposal']['sha256'],
        cache_sha256=reader.sha(out/'CACHE.json'), orders_sha256=reader.sha(out/'PRETRAIN_ORDERS.json'),
        frozen_sha256=reader.sha(out/'FROZEN.json'), diagnose_started_sha256=reader.sha(out/'DIAGNOSE_STARTED.json'),
        audit_started_utc=started.isoformat(), diagnostic_start_marker_verified=True,
        baseline_local67_full_orders_unchanged=True, filing_source_grid_verified=True,
        all_three_filing_features_common_mask=True, original_source_cutoffs_verified=True,
        filing_feature_matrices_verified=3, available_feature_cells=available,
        clock_policy_id=CLOCK, exact_acceptance_certified=False,
        source_hashes={str(p): reader.sha(p) for p in source_files(reader.root)},
        limitations=['Repeated historical sample; not pristine out of sample.',
                     'Filing-date and issuer continuity use declared retrospective reconstruction assumptions; no exact acceptance or earnings-first certification.',
                     'Beta is a counted nuisance estimate; no new predictive model is fitted.',
                     'Protected inventory verification is performed separately by formal closure.'])
    independent_imports()
    for path, expected in list(reader.seen.items()):
        require(reader.sha(path) == expected, 'Post-audit source/artifact change')
    report['artifact_hashes'] = dict(reader.seen)
    return report

def prereview(root=ROOT):
    """Run only generated-input tests and bind the final prospective contract."""
    independent_imports()
    root = Path(root).resolve()
    import test_monthly_filing_association157_outcome_review as test_module
    log = StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    require(result.wasSuccessful() and result.testsRun > 0 and not result.skipped, 'Synthetic outcome tests failed: '+log.getvalue())
    reader = Reader(root)
    proposal = 'work/monthly_filing_association157_proposal.json'
    proposal_doc = reader.json(proposal)
    require(proposal_doc.get('round') == 157 and proposal_doc.get('arms') == list(NAMES)
            and proposal_doc.get('selection_contract') == proposal_contract(), 'Final prospective contract mismatch')
    implementations = ['work/'+name for name in ('after156_association_measure.py',
        'monthly_filing_association157_runner.py', 'monthly_filing_association157_lifecycle.py',
        'prepare_filing_association157_proposal.py')]
    inputs = [proposal, 'work/after156_association_protocol.json',
              SOURCE+'/MANIFEST.json', 'work/after156_association_package_review.json']
    report = dict(passed=True, round=157, stage='preregistration_outcome_review', synthetic_only=True,
        proposal_sha256=reader.sha(proposal), actual_factors_calculated=0, actual_returns_calculated=0,
        actual_orders_calculated=0, actual_predictive_model_fits=0, actual_nuisance_estimates=0, producer_modules_imported=False,
        tests=result.testsRun, test_log=log.getvalue(),
        source_hashes={str(p): reader.sha(p) for p in source_files(root)},
        implementation_hashes={str(reader.path(p)): reader.sha(p) for p in implementations},
        input_hashes={str(reader.path(p)): reader.sha(p) for p in inputs},
        expected_execution_paths=12, expected_monthly_orders=488,
        reviewer_scope='Independent accounting, independent Decimal beta/features, and generated-input tamper tests; actual audit remains pending.',
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
    destination = (args.root/'work/monthly_filing_association157_outcome_prereview.json' if args.prereview
                   else args.root/STUDY/'INDEPENDENT_OUTCOME.json')
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(destination), execution_paths=12)))


if __name__ == '__main__':
    main()

