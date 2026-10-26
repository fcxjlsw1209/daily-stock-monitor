"""Independent fixed KO/WMT annual-lag pilot outcome review.

Only the frozen independent accounting helper used by round155 is reused.
No producer model, runner, selector or lifecycle is imported. Pure functions
accept supplied objects; actual I/O is gated by a registered completed run.
"""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import sys
import unittest

import numpy as np
import pandas as pd
import monthly_paired_target143_outcome_review as accounting

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'outputs/monthly_annual_lag158'
NAMES = ('incumbent_local67', 'negative_current_annual_lag', 'negative_annual_lag_change')
EXTRA = ('negative_current_lag', 'negative_lag_change', 'current_lag_rank',
         'lag_change_rank', 'common_available')
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'PRETRAIN_ORDERS.json',
               'PRICE_PARITY.json', 'FEATURE_AUDIT.json', 'CALENDAR.json', 'RANK_AUDIT.json')
FROZEN_FILES = ('PLAN.json', 'CACHE.json', 'PRETRAIN_ORDERS.json', 'TRAIN_TRIALS.json',
                'TRAIN_OPEN_TRIALS.json', 'INDEPENDENT_PREPARE.json')
PATHS = accounting.PATHS
OUTCOME_CHECKS = ('daily_nav_trades_holdings_metrics_slices', 'first88_once_selection_verified',
    'annual_partial_flags_verified', 'monthly_difference_and_attribution_verified',
    'first88_causal_replay_verified', 'zero_model_fit_counts_verified',
    'all_changed_months_reported', 'diagnostic_start_marker_verified',
    'baseline_local67_full_orders_unchanged', 'annual_lag_source_grid_verified',
    'original_source_cutoffs_verified', 'both_lag_features_common_mask',
    'centered_rank_and_score_bound_verified', 'incremental_evidence_counts_verified')
h, require, Reader = accounting.h, accounting.require, accounting.Reader
daily_axis, replay = accounting.daily_axis, accounting.replay
compare_stats, normalize = accounting.compare_stats, accounting.normalize


def preparation():
    return importlib.import_module('annual_lag158_independent_prepare')


def independent_imports():
    forbidden = ('monthly_annual_lag158_model', 'monthly_annual_lag158_runner',
                 'annual_lag158_lifecycle', 'monthly_annual_lag158_lifecycle')
    require(not any(name in sys.modules for name in forbidden), 'No producer or lifecycle imports')


def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    paths = [root/'work'/name for name in ('annual_lag158_independent_outcome.py',
        'test_annual_lag158_independent_outcome.py', 'monthly_paired_target143_outcome_review.py',
        'test_monthly_paired_target143_outcome_review.py', 'monthly_ridge_support70_result_review.py')]
    return sorted(set(preparation().source_files(root) + paths))


def count(doc, key, expected):
    require(type(doc.get(key)) is int and doc[key] == expected, 'Exact count required: '+key)


def validate_orders(close, opening, orders, meta):
    require(isinstance(close, pd.DataFrame) and isinstance(opening, pd.DataFrame), 'Price frames')
    dates = daily_axis(close.index, 'prices')
    require(opening.index.equals(dates) and opening.columns.equals(close.columns)
            and close.columns.is_unique, 'Same unique price axes')
    days, ends = daily_axis(meta['dates'], 'signals'), pd.DatetimeIndex(meta['ends'])
    require(len(days) == len(ends) == len(meta['symbols']) == 122 and ends.isin(dates).all()
            and days.isin(dates).all() and (ends > days).all(), '122 complete signal periods')
    require(days[1:].equals(ends[:-1]), 'Adjacent frozen holding periods')
    require(set(orders) == set(NAMES), 'Three exact order arms')
    for name in NAMES:
        require(type(orders[name]) is list and len(orders[name]) == 122, '122 orders per arm')
        for i, row in enumerate(orders[name]):
            require(set(row) == {'date', 'end', 'signal_date', 'symbols', 'scores'}, 'Exact order fields')
            require(row['date'] == row['signal_date'] == meta['dates'][i]
                    and row['end'] == meta['ends'][i], 'Order date and period')
            names = row['symbols']
            require(type(names) is list and len(names) == len(set(names)) and len(names) in (0, 2, 3),
                    'Cash or two/three equal positions')
            require(all(type(s) is str and s in meta['symbols'][i] and s in close.columns for s in names),
                    'Original order security identifiers')
            require(len(row['scores']) == len(names) and
                    all(type(v) in (int, float) and math.isfinite(v) for v in row['scores']), 'Aligned finite scores')
            require(dates.get_loc(days[i])+1 < len(dates), 'Next price session exists')
    return dates[(dates >= days[0]) & (dates <= ends[-1])]


def select(rows, open_rows):
    """Reconstruct the sole first88 decision. No fold or later-period selection."""
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    def checks(group, opening):
        require(type(group) is list and len(group) == 3 and
                [r.get('name') for r in group] == list(NAMES), 'Three exact selection arms')
        values = [[row.get(a, {}).get(b) for a, b in fields] for row in group]
        def guard(reference):
            a, b = values[2], values[reference]
            if not all(type(v) in (int, float) and math.isfinite(v) for pair in (a, b) for v in pair):
                return False
            cagr = a[0] >= b[0]-1e-12 if opening and reference == 0 else a[0] > b[0]+1e-12
            return bool(cagr and all(x >= y-t for x, y, t in zip(a[1:], b[1:], (.05, .03, .025, .025))))
        return {NAMES[i]: guard(i) for i in (0, 1)}
    close_checks, open_checks = checks(rows, False), checks(open_rows, True)
    admitted = all(close_checks.values()) and all(open_checks.values())
    return dict(primary=NAMES[2] if admitted else NAMES[0], admitted=admitted, guards=admitted,
        guards_against=close_checks, next_open_guards_against=open_checks,
        mechanism_confirmed=all(open_checks.values()))


def validate_months(months, meta):
    require(type(months) is list and len(months) == len(meta['dates']) == 122, 'All122 rank rows')
    keys = {'signal_index', 'date', 'shared_query_population', 'informative_pair',
            'initial_partial_fallback', 'unchanged_current_correction', 'unchanged_change_correction'}
    for i, row in enumerate(months):
        require(set(row) == keys and type(row['signal_index']) is int and row['signal_index'] == i
                and row['date'] == meta['dates'][i], 'Ordered rank audit')
        n = row['shared_query_population']
        require(type(n) is int and 0 <= n <= 2, 'Fixed two-issuer common population')
        require(all(type(row[k]) is bool for k in keys-{'signal_index', 'date', 'shared_query_population'}),
                'Exact rank booleans')
        require(row['informative_pair'] == (n == 2), 'Two-issuer availability flag')
        require(row['initial_partial_fallback'] == (row['date'] == '2016-08-17'), 'Initial partial flag')
        if row['initial_partial_fallback']:
            require(n == 0, 'Initial partial unavailable')
        if n <= 1:
            require(row['unchanged_current_correction'] and row['unchanged_change_correction'],
                    'Singleton and unavailable months have zero correction')
    return months


def monthly(orders, close, meta, months):
    validate_months(months, meta)
    rows = []
    for i, day in enumerate(meta['dates']):
        held, returns = {}, {}
        for name in NAMES:
            names = orders[name][i]['symbols']; held[name] = names
            returns[name] = math.fsum(float(close.at[pd.Timestamp(meta['ends'][i]), s]) /
                float(close.at[pd.Timestamp(day), s])-1. for s in names)/len(names) if names else 0.
        for name, key in zip(NAMES[1:], ('unchanged_current_correction', 'unchanged_change_correction')):
            if months[i][key]:
                h.assert_exact_tree(orders[name][i], orders[NAMES[0]][i])
        rows.append(dict(date=day, end=meta['ends'][i], split='early53' if i < 53 else
            'recent35' if i < 88 else 'later34', holdings=held, returns=returns, annual_lag=months[i]))
    return rows


def attribution(rows, metrics):
    comparisons = {}
    for label, name, reference in (('change_vs_incumbent', NAMES[2], NAMES[0]),
            ('current_vs_incumbent', NAMES[1], NAMES[0]), ('change_vs_current', NAMES[2], NAMES[1])):
        changes, ordered = [], 0
        for row in rows:
            ordered += int(row['holdings'][reference] != row['holdings'][name])
            if set(row['holdings'][reference]) != set(row['holdings'][name]):
                changes.append(dict(date=row['date'], split=row['split'], old=row['holdings'][reference],
                    new=row['holdings'][name], guidance_events=[], annual_lag=row['annual_lag'],
                    log_contribution=math.log1p(row['returns'][name])-math.log1p(row['returns'][reference])))
        positive = sorted((x['log_contribution'] for x in changes if x['log_contribution'] > 0.), reverse=True)
        metric = metrics[name]
        comparisons[label] = dict(changed_months=len(changes), ordered_changed_months=ordered, changes=changes,
            undo_best1_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:1])))**(1/metric['years'])-1,
            undo_best3_cagr=((1+metric['total_return'])*math.exp(-math.fsum(positive[:3])))**(1/metric['years'])-1)
    return dict(comparisons=comparisons)


def incremental_evidence(orders, months):
    correction = {name:sum(not row[key] for row in months) for name,key in
        zip(NAMES[1:], ('unchanged_current_correction','unchanged_change_correction'))}
    changed = {name:sum(orders[name][i]['symbols'] != orders[NAMES[0]][i]['symbols']
        for i in range(122)) for name in NAMES[1:]}
    return dict(nonzero_correction_months=correction, ordered_holdings_changed_months=changed,
        primary_no_incremental_evidence=correction[NAMES[2]] == 0 or changed[NAMES[2]] == 0)


def check_outcomes(close, opening, orders, meta, read):
    dates = validate_orders(close, opening, orders, meta)
    results = read('RESULTS.json'); require(set(results) == set(NAMES), 'Three result arms')
    bounds = dict(early2016_2020=(meta['dates'][0], meta['ends'][52]),
        original2021_2026=(meta['dates'][53], meta['ends'][-1]),
        training2016_2023=(meta['dates'][0], meta['ends'][87]), later34=(meta['dates'][88], meta['ends'][-1]))
    errors, open_navs, navs, metrics, details, maxnav = [], {}, {}, {}, {}, 0.
    for name in NAMES:
        require(set(results[name]) == {p[2] for p in PATHS}, 'Three execution paths per arm')
        for execution, cost, key in PATHS:
            met, nav, trades, hold = replay(close, opening, orders[name], dates, execution, cost)
            prefix = name+'_'+key
            compare_stats(results[name][key]['metrics'], met, errors, prefix+'/metrics')
            compare_stats(results[name][key]['slices'],
                {k: h.slice_stat(dates, nav, *span) for k, span in bounds.items()}, errors, prefix+'/slices')
            daily = read(prefix+'_DAILY.json')
            require(all(set(x) == {'date', 'equity'} for x in daily) and
                    [x['date'] for x in daily] == [str(x.date()) for x in dates], 'Exact daily NAV dates')
            actual = np.asarray([x['equity'] for x in daily], float)
            require(np.isfinite(actual).all() and (actual > 0).all(), 'Finite positive NAV')
            np.testing.assert_allclose(actual, nav, atol=1e-5, rtol=2e-11)
            maxnav = max(maxnav, float(np.max(np.abs(actual-nav))))
            actual_trades, actual_hold = read(prefix+'_TRADES.json'), read(prefix+'_HOLDINGS.json')
            h.compare(normalize(actual_trades), normalize(trades), errors, prefix+'/trades')
            h.compare(normalize(actual_hold), normalize(hold), errors, prefix+'/holdings')
            allowed = {r['date'] if execution == 'close' else
                str(close.index[close.index.get_loc(pd.Timestamp(r['date']))+1].date()) for r in orders[name]}
            require(all(t['date'] in allowed for t in actual_trades), 'No intramonth trades')
            require([x['year'] for x in met['annual_returns']] == sorted({int(x.year) for x in dates}) and
                all(x['partial'] is (x['year'] in (2016, 2026)) for x in met['annual_returns']), 'Annual partial flags')
            details[prefix] = dict(trades=len(trades), holding_periods=len(hold), fees_paid=met['fees_paid'])
            if execution == 'close':
                navs[name], metrics[name] = nav, met
            if key == 'next_open_cost':
                open_navs[name] = nav
        td = dates[dates <= pd.Timestamp(meta['ends'][87])]
        for execution, cost, full in (('close', 0., navs[name]), ('open', .0005, open_navs[name])):
            _, tn, _, _ = replay(close.loc[:td[-1]], opening.loc[:td[-1]], orders[name][:88], td, execution, cost)
            np.testing.assert_allclose(tn, full[:len(td)], atol=1e-5, rtol=2e-11)
    measured = [dict(name=name, **h.measures(dates, navs[name], meta)) for name in NAMES]
    opened = [dict(name=name, **h.measures(dates, open_navs[name], meta)) for name in NAMES]
    compare_stats(read('TRAIN_TRIALS.json'), measured, errors, 'training')
    compare_stats(read('TRAIN_OPEN_TRIALS.json'), opened, errors, 'open_training')
    frozen = read('FROZEN.json'); count(frozen, 'selection_count', 1)
    selected = select(measured, opened)
    h.assert_exact_tree(frozen['selection'], selected)
    rank = read('RANK_AUDIT.json'); count(rank, 'prefixes_checked', 123)
    months = monthly(orders, close, meta, rank['months'])
    compare_stats(read('MONTH_COMPARISON.json'), months, errors, 'monthly')
    compare_stats(read('ATTRIBUTION.json'), attribution(months, metrics), errors, 'attribution')
    validation = read('VALIDATION.json')
    require(validation.get('primary') == selected['primary'] and validation.get('pristine_oos') is False,
            'Validation selected result')
    h.assert_exact_tree(validation['selection'], selected)
    feature = read('FEATURE_AUDIT.json')
    for doc in (validation, feature):
        count(doc, 'new_predictive_model_fits', 0); count(doc, 'nuisance_estimates', 0)
    incremental = incremental_evidence(orders, rank['months'])
    for key, value in incremental.items():
        h.assert_exact_tree(feature.get(key), value)
    require(type(validation.get('primary_no_incremental_evidence')) is bool and
        validation['primary_no_incremental_evidence'] == incremental['primary_no_incremental_evidence'],
        'No-incremental-evidence disclosure')
    count(validation, 'selection_count', 1)
    require(all(validation.get(k) is True for k in ('original_arrays_unchanged', 'no_observer_writes',
        'no_intramonth_trades', 'independent_outcome_review_pending')), 'Validation invariants')
    return dict(passed=True, stage='evaluate', execution_paths_verified=9,
        total_monthly_orders_verified=366,
        daily_nav_trades_holdings_metrics_slices=True, first88_once_selection_verified=True,
        annual_partial_flags_verified=True, monthly_difference_and_attribution_verified=True,
        first88_causal_replay_verified=True, zero_model_fit_counts_verified=True,
        all_changed_months_reported=True, selection_count=1, new_predictive_model_fits=0,
        nuisance_estimates=0, models_fit=0, selected_name=selected['primary'],
        incremental_evidence_counts_verified=True, **incremental,
        primary_cagr=metrics[selected['primary']]['cagr'],
        primary_target_reached=metrics[selected['primary']]['cagr'] >= .7,
        max_compared_absolute_difference=max((x for _, x in errors), default=0.),
        max_nav_absolute_difference=maxnav, replay_details=details, producer_modules_imported=False,
        protected_member_scan_repeated=False)


def verify_features(arrays, feature, rank, meta, rows, events, source_mask, calendar, input_digests):
    """Recompute dates/ranks with the separate ordinal/Fraction oracle."""
    prep = preparation()
    require(all(k in arrays and arrays[k].shape == (122,49) and
        arrays[k].dtype == (np.dtype(bool) if k == 'common_available' else np.dtype('float64'))
        for k in EXTRA), 'Five exact feature array axes/types')
    require(arrays['eligible'].shape == (122,49) and arrays['eligible'].dtype == bool and
            not (arrays['common_available'] & ~arrays['eligible']).any(), 'Original eligible subset')
    expected = prep.features_oracle(meta, rows, events, source_mask, arrays['eligible'], calendar)
    for key in EXTRA:
        np.testing.assert_array_equal(arrays[key], expected['arrays'][key], err_msg=key)
    prep.verify_feature_audit(feature, expected, rows, input_digests)
    h.assert_exact_tree(rank, prep.rank_audit_oracle(meta, expected))
    validate_months(rank['months'], meta)
    return int(expected['arrays']['common_available'].sum())


def verify_prepare_receipt(reader, out, plan, plan_sha, pins):
    cache = reader.json(out/'CACHE.json')
    require(set(cache) == set(CACHE_FILES), 'Exact preparation cache')
    cached = reader.verify(cache, out)
    doc = reader.json(out/'INDEPENDENT_PREPARE.json')
    require(doc.get('passed') is True and doc.get('stage') == 'prepare' and
            doc.get('producer_modules_imported') is False and doc.get('returns_or_nav_computed') is False and
            all(doc.get(k) is True for k in preparation().CHECKS), 'Independent prepare proof')
    for key, value in (('orders_checked',366), ('candidate_cells_checked',5978), ('months_checked',122),
        ('prefixes_checked',123), ('lag_feature_matrices_verified',2), ('new_predictive_model_fits',0),
        ('nuisance_estimates',0)):
        count(doc, key, value)
    for key, name in (('plan_sha256','PLAN.json'), ('cache_sha256','CACHE.json'), ('orders_sha256','PRETRAIN_ORDERS.json')):
        require(doc.get(key) == reader.sha(out/name), 'Prepare binding: '+key)
    require(doc['plan_sha256'] == plan_sha, 'Prepare PLAN binding')
    require(reader.verify(doc['source_hashes']) == reader.verify(plan['independent_prepare_source_hashes']),
            'Prepare source map changed')
    artifacts = reader.verify(doc['artifact_hashes'])
    required = dict(cached, **{str(out/name):reader.sha(out/name) for name in ('PLAN.json','CACHE.json')})
    allowed = dict(pins, **required)
    require(all(artifacts.get(k) == v for k,v in required.items()) and
            all(allowed.get(k) == v for k,v in artifacts.items()), 'Complete bound prepare artifacts')
    return doc


def verify_diagnostic_start(reader, out, plan, plan_sha, started):
    marker = reader.json(out/'DIAGNOSE_STARTED.json')
    require(set(marker) == {'plan_sha256','frozen_sha256','started_utc'} and
            marker['plan_sha256'] == plan_sha and marker['frozen_sha256'] == reader.sha(out/'FROZEN.json'),
            'Diagnostic marker binding')
    def utc(value):
        require(type(value) is str and value.endswith(('Z','+00:00')), 'Explicit UTC chronology')
        parsed = datetime.fromisoformat(value.replace('Z','+00:00'))
        require(parsed.tzinfo is not None, 'Timezone chronology')
        return parsed
    require(utc(plan['registered_utc']) < utc(marker['started_utc']) < started <= datetime.now(timezone.utc),
            'Registration/diagnostic/audit chronology')


def verify_frozen(reader, out, plan, plan_sha, started):
    frozen = reader.json(out/'FROZEN.json')
    require(set(frozen['hashes']) == set(FROZEN_FILES), 'Exact once-only frozen scope')
    require(not (out/'WALKFORWARD.json').exists(), 'No fold-selection artifact is permitted')
    reader.verify(frozen['hashes'], out)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,
            'Frozen diagnostic orders and reused-history disclosure')
    count(frozen, 'selection_count', 1)
    verify_diagnostic_start(reader, out, plan, plan_sha, started)


def bindings(reader, plan_sha, started=None):
    """All source/code/phase checks precede numerical price loading."""
    independent_imports()
    started = started or datetime.now(timezone.utc)
    prep = preparation(); out = reader.path(STUDY)
    plan, pins, inputs = prep.validate_plan(reader, out/'PLAN.json', plan_sha, stage='evaluate')
    sources = {str(p):reader.sha(p) for p in source_files(reader.root)}
    require(reader.verify(plan['independent_outcome_source_hashes']) == sources and
            all(pins.get(p) == v for p,v in sources.items()), 'Independent outcome implementation closure')
    for key in ('actual_nuisance_estimates_before_registration',):
        count(plan, key, 0)
    raw = reader.verify(plan['raw_source_hashes'])
    require(len(raw) == 64 and all(pins.get(p) == v for p,v in raw.items()), 'Original64 raw source closure')
    meta, rows, events, source, calendar = prep.source_inputs(reader, plan, pins)
    require(meta == reader.json(out/'META.json') and meta['dates'][0] == '2016-08-17'
            and meta['ends'][-1] == '2026-09-24', 'Original prepared metadata and horizon')
    require(prep.canonical(reader.json(out/'CALENDAR.json')) == prep.canonical(calendar), 'Prepared full calendar')
    verify_prepare_receipt(reader, out, plan, plan_sha, pins)
    verify_frozen(reader, out, plan, plan_sha, started)
    orders = reader.json(out/'PRETRAIN_ORDERS.json')
    h.assert_exact_tree(orders[NAMES[0]], reader.json(inputs['original_orders']))
    for name in NAMES[1:]:
        h.assert_exact_tree(orders[name][0], orders[NAMES[0]][0])
    return out, plan, inputs, meta, rows, events, source, calendar


def audit(root, plan_sha):
    independent_imports()
    prep = preparation(); reader = prep.old.Reader(root); started = datetime.now(timezone.utc)
    require(not (reader.path(STUDY)/'INDEPENDENT_OUTCOME.json').exists(), 'Outcome already reviewed; do not replay')
    out, plan, inputs, meta, rows, events, source, calendar = bindings(reader, plan_sha, started)
    arrays = reader.npz(out/'DATA.npz')
    prep.verify_original_arrays(reader.npz(prep.old.LOCAL+'/DATA.npz'), arrays)
    feature, rank = reader.json(out/'FEATURE_AUDIT.json'), reader.json(out/'RANK_AUDIT.json')
    usable = verify_features(arrays, feature, rank, meta, rows, events, source, calendar,
        {k:plan['inputs'][k]['sha256'] for k in ('source_overlay','source_review','source_prior')})
    data = reader.npz(inputs['prices'])
    require(set(data) == {'dates','symbols','close','opening'}, 'Original price payload')
    dates = pd.to_datetime(data['dates'])
    close = pd.DataFrame(data['close'], index=dates, columns=data['symbols'])
    opening = pd.DataFrame(data['opening'], index=dates, columns=close.columns)
    report = check_outcomes(close, opening, reader.json(out/'PRETRAIN_ORDERS.json'), meta,
                            lambda name:reader.json(out/name))
    report.update(schema_version=1, round=158, plan_sha256=plan_sha,
        proposal_sha256=plan['inputs']['proposal']['sha256'],
        cache_sha256=reader.sha(out/'CACHE.json'), orders_sha256=reader.sha(out/'PRETRAIN_ORDERS.json'),
        frozen_sha256=reader.sha(out/'FROZEN.json'), diagnose_started_sha256=reader.sha(out/'DIAGNOSE_STARTED.json'),
        audit_started_utc=started.isoformat(), diagnostic_start_marker_verified=True,
        baseline_local67_full_orders_unchanged=True, annual_lag_source_grid_verified=True,
        original_source_cutoffs_verified=True, both_lag_features_common_mask=True,
        centered_rank_and_score_bound_verified=True, lag_feature_matrices_verified=2,
        available_feature_cells=usable, exact_acceptance_certified=False,
        historical_price_identity_certified=False, source_admission_changed=False,
        source_hashes={str(p):reader.sha(p) for p in source_files(reader.root)},
        limitations=['Fixed KO/WMT pilot; sparse repeated annual pairs are not independent cross-sectional observations.',
            'Conditional reconstructed filing-date availability and issuer continuity retain their original limitations.',
            'Repeated historical sample; later34 is locked diagnostic evidence, not pristine out of sample.',
            'Protected inventory verification occurs separately at formal closure.'])
    independent_imports()
    for path, expected in list(reader.seen.items()):
        require(reader.sha(path) == expected, 'Post-audit source or artifact changed')
    report['artifact_hashes'] = dict(reader.seen)
    return report


def prereview(root=ROOT):
    """Run only this module's synthetic suite and bind the final proposal."""
    independent_imports()
    root = Path(root).resolve()
    suite = importlib.import_module('test_annual_lag158_independent_outcome')
    log = StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(suite))
    require(result.wasSuccessful() and result.testsRun > 0 and not result.skipped,
            'Synthetic outcome review failed: '+log.getvalue())
    prep = preparation(); reader = prep.old.Reader(root)
    proposal_path = 'work/monthly_annual_lag158_proposal.json'
    proposal = reader.json(proposal_path)
    require(proposal.get('round') == 158 and proposal.get('arms') == list(NAMES) and
        proposal.get('selection_contract') == prep.selection_contract() and
        proposal.get('measurement') == prep.measurement_contract(), 'Final prospective contracts')
    implementation = ['work/'+name for name in ('monthly_annual_lag158_model.py',
        'monthly_annual_lag158_runner.py', 'monthly_annual_lag158_lifecycle.py')]
    inputs = [proposal_path]+[ref[0] for ref in prep.FIXED.values()]
    report = dict(round=158, stage='preregistration_outcome_review', passed=True, synthetic_only=True,
        proposal_sha256=reader.sha(proposal_path), actual_features_or_returns_computed=False,
        actual_factors_calculated=0, actual_lag_values_calculated=0, actual_orders_calculated=0,
        actual_returns_calculated=0, actual_predictive_model_fits=0, actual_nuisance_estimates=0,
        producer_modules_imported=False, tests=result.testsRun, test_log=log.getvalue(),
        source_hashes={str(p):reader.sha(p) for p in source_files(root)},
        implementation_hashes={str(reader.path(p)):reader.sha(p) for p in implementation},
        input_hashes={str(reader.path(p)):reader.sha(p) for p in inputs},
        expected_execution_paths=9, expected_monthly_orders=366, actual_audit_pending=True)
    independent_imports()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--plan-sha')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--evaluate-complete', action='store_true'); choice.add_argument('--prereview', action='store_true')
    args = parser.parse_args()
    destination = (args.root/'work/annual_lag158_outcome_prereview.json' if args.prereview else
                   args.root/STUDY/'INDEPENDENT_OUTCOME.json')
    require(not destination.exists(), 'Existing receipt must be preserved')
    report = prereview(args.root) if args.prereview else audit(args.root, args.plan_sha)
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(destination),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), execution_paths=9)))


if __name__ == '__main__':
    main()
