"""Independent159 verifier. Real I/O requires exact registered PLAN and phase.

No new producer, kernel or host is imported. Preparation never calls accounting
or reads outcomes. Outcome audit requires frozen selection and diagnostic start.
"""
from pathlib import Path
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import re
import sys

import numpy as np
import pandas as pd

import basic_shares159_independent_prepare as prepare
import basic_shares159_independent_outcome as outcome

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_basic_shares159'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json',
               'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json', 'SUPPORT_AUDIT.json')
PREPARE_CHECKS = ('original27_arrays_byte_identical', 'original5978_rows_preserved',
    'two_clock_tminus42_latest_invalid_no_fallback', 'original_eligible_and_distinct_issuer_before_transform',
    'exact_fraction_raw_and_rank', 'centered_five_percent_magnitude_and_rank_hands',
    'once_point01_delta_then_original_episode', 'all123_feature_and_score_prefixes',
    'initial_and_zero_delta_exact_incumbent', 'baseline_full_orders_unchanged',
    'source_support_independently_verified', 'raw_price_parity_independently_verified', 'sourcegate_v2_review_bound')
OUTCOME_CHECKS = ('all9_daily_nav_trade_holdings_metrics', 'first88_once_selection',
    'causal_first88_replay', 'source_and_changed_support_verified', 'diagnostic_start_bound',
    'baseline_full_orders_unchanged', 'all_original_inputs_and_frozen_artifacts_unchanged')
SOURCE_CHECKS = ('all_input_output_and_implementation_hashes_verified','independent_earliest_cached_accession_selection',
    'all_88_original_counts_and_native_intervals_verified','all_44_original_periods_and_holds_verified',
    'max_of_two_second_session_close_clocks','v1_failure_preserved_and_only_v2_unit_guard_changed',
    'all_34_general_units_have_independent_original_table_evidence','all_5978_grid_rows_and_original_nested_records_preserved',
    'latest_known_before_validity_no_fallback','original_t_minus_42_and_identity_constraints',
    'six_original_rejections_not_lifted','initial_partial_unchanged','all_five_coverage_splits_recomputed',
    'strict_admission_flags_remain_false')
require = prepare.require


def source_files(root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    files = [root/'work'/name for name in (
        'basic_shares159_independent.py', 'basic_shares159_independent_prepare.py',
        'basic_shares159_independent_outcome.py', 'basic_shares159_independent_tests.py',
        'monthly_paired_target143_outcome_review.py', 'monthly_ridge_support70_result_review.py',
        'monthly_latent_state134_prepare_review.py')]
    files.append(root/'outputs/monthly_stability_monitor/inputs.py')
    modules = ('exchange_calendars.exchange_calendar_xnys', 'exchange_calendars.exchange_calendar',
               'exchange_calendars.us_holidays', 'exchange_calendars.calendar_helpers', 'pandas.tseries.holiday')
    files.extend(Path(importlib.import_module(name).__file__).resolve() for name in modules)
    return sorted(set(files))


def measurement_contract():
    return dict(arms=list(prepare.NAMES),source_scope=list(prepare.ISSUERS),
        raw='1-current_basic_annual/same_report_prior_basic_annual',arithmetic='exact_decimal_to_Fraction; bounded score conversion only',
        magnitude='(x/(0.05+abs(x))-mean_over_common_eligible_population)/2',
        rank='2*((average_ascending_rank-1)/(n-1)-0.5)',dose=.01,max_abs_delta=.01,
        common_mask='admitted_latest_same_report_pair_intersect_original_eligibility; no ETF',
        known_at='second_XNYS_session_strictly_after_release_date_close',source_cutoff='original_signal_session_minus42_close',
        max_age_days=550,initial_partial='exact_incumbent',zero_delta='exact_incumbent',
        placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',original_arrays=27,
        extra_arrays=list(prepare.EXTRA),new_predictive_model_fits=0,nuisance_estimates=0,
        usable_at='max(second_XNYS_close_after_release,second_XNYS_close_after_selected_corroborating_filing)',
        identity='exact source symbol/security and distinct issuer in each original 49-name row, under preserved historical identity limitations',
        conditional_source_policy='after158_basic_shares_admission_v2')


def selection_contract():
    return dict(selection_count=1,first88_only=True,early53=True,recent35=True,later34_diagnostic_only=True,
        prefix_reselection=False,minimum_source_events=12,minimum_source_issuers=3,minimum_multi_source_months=24,
        minimum_multi_source_months_each_era=8,minimum_close_cagr_gain=.005,cagr_epsilon=1e-12,
        open_cost_noninferior=True,sharpe_tolerance=.05,drawdown_tolerance=.03,era_cagr_tolerance=.025,
        minimum_changed_held_set_months=3,minimum_changed_each_era=1,minimum_changed_source_events=3,
        minimum_changed_source_issuers=2,choice='highest_same_close_first88_CAGR; ties_within1e-12_lower_fixed_arm_index',execution_paths=9)


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.seen = {}
    def path(self, path):
        value = Path(path)
        value = (value if value.is_absolute() else self.root/value).resolve()
        require(not any(x in str(value) for x in ('CloudStorage', 'OneDrive', 'stock-observer-runtime')), 'Forbidden path')
        return value
    def sha(self, path):
        path = self.path(path)
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        require(str(path) not in self.seen or self.seen[str(path)] == value, 'Input changed during audit')
        self.seen[str(path)] = value
        return value
    def json(self, path):
        self.sha(path)
        def unique(items):
            result = {}
            for key, value in items:
                require(key not in result, 'Duplicate JSON key')
                result[key] = value
            return result
        def nonfinite(value):
            raise ValueError('Nonfinite JSON token: '+value)
        return json.loads(self.path(path).read_text(), object_pairs_hook=unique, parse_constant=nonfinite)
    def npz(self, path):
        self.sha(path)
        with np.load(self.path(path), allow_pickle=False) as values:
            return {key: values[key].copy() for key in values.files}
    def verify(self, pins):
        result = {}
        require(type(pins) is dict and pins, 'Nonempty hash map')
        for name, digest in pins.items():
            require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'SHA256 required')
            path = str(self.path(name))
            require(path not in result or result[path] == digest, 'Conflicting normalized digest')
            require(self.sha(path) == digest, 'Hash mismatch: '+path)
            result[path] = digest
        return result


def bind(reader, plan_sha, stage):
    prepare.independent_imports()
    require(stage in ('prepare', 'outcome'), 'Known independent phase')
    require(type(plan_sha) is str and re.fullmatch('[0-9a-f]{64}', plan_sha), 'Explicit PLAN SHA')
    directory = reader.path(OUT)
    require(reader.sha(directory/'PLAN.json') == plan_sha, 'Exact registered PLAN')
    plan = reader.json(directory/'PLAN.json')
    require(plan['round'] == 159 and plan['status'] == 'REGISTERED' and plan['arms'] == list(prepare.NAMES),
            'Fixed registered round159')
    require(reader.path(plan['output_directory']) == directory, 'Exact study output directory')
    require(plan['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Pinned runtime')
    require(plan['measurement'] == measurement_contract() and plan['selection_contract'] == selection_contract(),
            'Fixed prospective formulas and single selection')
    for key in ('actual_features_before_registration', 'actual_orders_before_registration', 'actual_returns_before_registration'):
        require(type(plan[key]) is int and plan[key] == 0, 'No preregistration numerical work')
    require(plan['no_observer_writes'] is True, 'Research scope')
    if stage == 'prepare':
        require(not any((directory/name).exists() for name in ('SELECT_STARTED.json', 'TRAIN_TRIALS.json',
            'TRAIN_OPEN_TRIALS.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json', 'RESULTS.json')), 'Prepare must precede all outcomes')
    pins = reader.verify(plan['hashes'])
    for path in source_files(reader.root):
        require(pins.get(str(path)) == reader.sha(path), 'Unregistered independent implementation: '+str(path))
    paths = {}
    for key, ref in plan['inputs'].items():
        path = reader.path(ref['path'])
        require(reader.sha(path) == ref['sha256'] == pins.get(str(path)), 'Named input binding: '+key)
        paths[key] = path
    fixed = dict(meta='outputs/monthly_local_linear70/META.json', policy='outputs/monthly_local_linear70/INCUMBENT_POLICY.json',
        original_orders='outputs/monthly_local_linear70/local_linear_ORDERS.json', prices='outputs/monthly_2016_extension/PRICES.npz',
        design='work/after158_basic_shares_design_review.json', calendar='outputs/monthly_paired_target143/CALENDAR.json',
        source_events='outputs/after158_basic_shares_admission_v2/EVENTS.json', source_rows='outputs/after158_basic_shares_admission_v2/ROWS.json',
        source_manifest='outputs/after158_basic_shares_admission_v2/MANIFEST.json',
        source_review='work/after158_basic_shares_admission_independent.json')
    require(set(paths) == set(fixed) | {'protected_manifest'}, 'Exact named input schema')
    for key, value in fixed.items():
        require(paths[key] == reader.path(value), 'Original fixed input path: '+key)
    require(all(reader.json(paths['policy']).get(k) == v for k, v in prepare.old_helper().POLICY.items()), 'Original episode policy')
    cache = reader.json(directory/'CACHE.json')
    require(set(cache) == set(CACHE_FILES), 'Complete exact preparation cache')
    reader.verify({str(directory/name): value for name, value in cache.items()})
    start = reader.json(directory/'PREPARE_STARTED.json')
    require(start['plan_sha256'] == plan_sha, 'Preparation start PLAN binding')
    return plan, pins, paths


def source_inputs(reader, plan, pins, paths):
    manifest, review = reader.json(paths['source_manifest']), reader.json(paths['source_review'])
    require(review['passed'] is True and review['source_manifest_sha256'] == reader.sha(paths['source_manifest']) and
            review['plan_sha256'] == manifest['plan_sha256'] and all(review['checks'].get(k) is True for k in SOURCE_CHECKS),
            'Complete independent v2 source review bound to this manifest')
    for doc in (manifest, review):
        require(doc['strict_source_admitted'] is False, 'No strict source promotion')
        checked = reader.verify(doc['input_hashes'])
        require(all(pins.get(p) == h for p, h in checked.items()), 'Unregistered source closure')
    outputs = reader.verify(manifest['outputs'])
    require(all(pins.get(p) == h for p, h in outputs.items()), 'Unregistered source outputs')
    require(outputs.get(str(paths['source_events'])) == reader.sha(paths['source_events']) and
            outputs.get(str(paths['source_rows'])) == reader.sha(paths['source_rows']), 'Current source output binding')
    meta = reader.json(paths['meta'])
    require(meta == reader.json(reader.path(OUT)/'META.json') and len(meta['dates']) == len(meta['symbols']) == 122 and
            meta['train_months'] == 88 and meta['early_months'] == 53, 'Original full metadata')
    original_path = reader.path('outputs/monthly_revenue_pilot_source156/ROWS.json')
    require(reader.sha(original_path) == pins.get(str(original_path)), 'Original source rows registered')
    events = reader.json(paths['source_events'])['events']
    rows, originals = reader.json(paths['source_rows']), reader.json(original_path)
    saved = reader.json(paths['calendar'])
    closure = reader.verify(saved['provenance']['source_hashes'])
    require(all(pins.get(p) == h for p, h in closure.items()), 'Complete calendar runtime/source closure')
    exception_path = reader.path('outputs/index_union/CALENDAR_EXCEPTIONS.json')
    require(reader.sha(exception_path) == pins.get(str(exception_path)), 'Calendar exception pin')
    closed = set()
    for record in reader.json(exception_path)['closures']:
        if 'XNYS' in record['markets']:
            path = reader.path('outputs/index_union')/record['source_file']
            require(reader.sha(path) == record['source_sha256'] == pins.get(str(path)), 'Official calendar closure evidence')
            closed.add(record['date'])
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    schedule = XNYSExchangeCalendar(start=saved['start'], end=saved['end']).schedule
    sessions = [str(d.date()) for d in schedule.index if str(d.date()) not in closed]
    require(sessions == saved['sessions'], 'Independent XNYS sessions')
    closes = {str(d.date()): int(c.value) for d, c in zip(schedule.index, schedule['close']) if str(d.date()) not in closed}
    return meta, events, rows, originals, dict(sessions=sessions, closes=closes)


def prices(reader, plan, pins, paths, meta, parity_check=False):
    raw = reader.verify(plan['raw_source_hashes'])
    require(len(raw) == 64 and all(pins.get(p) == h for p, h in raw.items()), 'Original64 raw price closure')
    cached = reader.npz(paths['prices'])
    require(set(cached) == {'dates', 'symbols', 'close', 'opening'}, 'Original price cache schema')
    if parity_check:
        import monthly_latent_state134_prepare_review as old
        loader_path = reader.path(old.LOADER)
        require(reader.sha(loader_path) == old.LOADER_SHA == pins.get(str(loader_path)), 'Original price loader')
        spec = importlib.util.spec_from_file_location('basic159_independent_price_loader', loader_path)
        loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
        panel, info = loader.load_panel(meta['etfs'])
        used = {str(reader.path(p)): h for p, h in info['source_hashes'].items()}
        used[str(loader_path)] = old.LOADER_SHA
        require(used == raw, 'Actually consumed raw price sources')
        reader.verify(used)
        parity = old.price_parity(panel, cached)
        old.compare(reader.json(reader.path(OUT)/'PRICE_PARITY.json'), parity, 'PRICE_PARITY')
    close = pd.DataFrame(cached['close'], index=pd.to_datetime(cached['dates']), columns=cached['symbols'])
    opening = pd.DataFrame(cached['opening'], index=close.index, columns=close.columns)
    return close, opening


def finish(reader, plan, extra):
    allowed = {str(reader.path(p)): h for p, h in plan['hashes'].items()}
    allowed.update({str(reader.path(p)): h for p, h in extra.items()})
    for path, digest in list(reader.seen.items()):
        require(reader.sha(path) == digest == allowed.get(path), 'Unbound or changed audit input: '+path)
    prepare.independent_imports()


def preparation_audit(root, plan_sha):
    reader = Reader(root); directory = reader.path(OUT)
    plan, pins, paths = bind(reader, plan_sha, 'prepare')
    meta, events, rows, originals, calendar = source_inputs(reader, plan, pins, paths)
    original_path = reader.path('outputs/monthly_local_linear70/DATA.npz')
    require(reader.sha(original_path) == pins.get(str(original_path)), 'Original27 array cache bound')
    original, arrays = reader.npz(original_path), reader.npz(directory/'DATA.npz')
    prepare.verify_original_arrays(original, arrays)
    expected = prepare.features_oracle(meta, events, rows, originals, original['eligible'], calendar)
    feature = reader.json(directory/'FEATURE_AUDIT.json')
    prepare.verify_feature_output(expected, arrays, feature)
    support = outcome.support_oracle(expected['cells'])
    require(support['passed'] is True, 'Pre-outcome support gate failed; this pilot must stop')
    scored = prepare.score_oracle(original, meta, expected['arrays'])
    orders = prepare.order_oracle(meta, scored)
    saved_scores, saved_orders = reader.npz(directory/'SCORES.npz'), reader.json(directory/'PRETRAIN_ORDERS.json')
    require(set(saved_scores) == set(saved_orders) == set(prepare.NAMES), 'Three prepared score/order arms')
    accounting = outcome.accounting_helper()
    for name in prepare.NAMES:
        require(saved_scores[name].shape == (122,49) and saved_scores[name].dtype == scored['scores'][name].dtype, 'Score shape/dtype')
        np.testing.assert_allclose(saved_scores[name], scored['scores'][name], atol=1e-12, rtol=0)
        errors=[]; accounting.h.compare(saved_orders[name], orders[name], errors, name)
    require(saved_orders[prepare.NAMES[0]] == reader.json(paths['original_orders']), 'Exact original full incumbent orders')
    changed = {name: outcome.changed_support_oracle(orders, expected['cells'], name) for name in prepare.NAMES[1:]}
    require(reader.json(directory/'SUPPORT_AUDIT.json') == dict(source=support, changed=changed), 'Independent support counts')
    for n in range(123):
        prefix = prepare.features_oracle(meta, events, rows, originals, original['eligible'], calendar, n)
        for key in prepare.EXTRA:
            np.testing.assert_array_equal(prefix['arrays'][key], expected['arrays'][key][:n])
        prior = prepare.score_oracle(original, meta, prefix['arrays'], n)
        for name in prepare.NAMES:
            require(prior['choices'][name] == scored['choices'][name][:n] and
                    prior['scores'][name].tobytes() == scored['scores'][name][:n].tobytes(), 'Prefix score/order invariant')
        # Replace future values rather than merely truncate outputs. No actual
        # future source quantity may be needed to reconstruct an earlier prefix.
        last_cut = prepare.ns(rows[49*(n-1)]['cutoff']) if n else -1
        poisoned_events = [e if prepare.ns(e['known_at']) <= last_cut else
                           {k:e[k] for k in ('event_id','symbol','known_at')} for e in events]
        future = prepare.features_oracle(meta,poisoned_events,rows,originals,original['eligible'],calendar,n)
        for key in prepare.EXTRA:
            np.testing.assert_array_equal(future['arrays'][key],prefix['arrays'][key])
        poisoned_arrays = {k:v.copy() for k,v in original.items()}
        for value in poisoned_arrays.values():
            if value.ndim and value.shape[0] == 122:
                if value.dtype.kind == 'f':value[n:] = np.nan
                elif value.dtype.kind == 'b':value[n:] = ~value[n:]
        future_scores = prepare.score_oracle(poisoned_arrays,meta,prefix['arrays'],n)
        for name in prepare.NAMES:
            require(future_scores['choices'][name] == prior['choices'][name] and
                    future_scores['scores'][name].tobytes() == prior['scores'][name].tobytes(), 'Future-poison score invariant')
    for name, key in zip(prepare.NAMES[1:], prepare.HANDS):
        for i in range(122):
            if not expected['arrays'][key][i].any():
                require(scored['choices'][name][i] == scored['choices'][prepare.NAMES[0]][i] and
                        scored['scores'][name][i].tobytes() == scored['scores'][prepare.NAMES[0]][i].tobytes(), 'Exact neutral parity')
    prices(reader, plan, pins, paths, meta, parity_check=True)
    extra = {str(directory/name): reader.sha(directory/name) for name in (*CACHE_FILES, 'PLAN.json', 'CACHE.json', 'PREPARE_STARTED.json')}
    finish(reader, plan, extra)
    return dict(schema='basic_shares159_independent_prepare_v1', passed=True, stage='prepare', plan_sha256=plan_sha,
        created_utc=datetime.now(timezone.utc).isoformat(), cache_sha256=reader.sha(directory/'CACHE.json'),
        checks=dict.fromkeys(PREPARE_CHECKS, True), source_hashes={str(p):reader.sha(p) for p in source_files(reader.root)},
        artifact_hashes=dict(reader.seen), candidate_cells_checked=5978, orders_checked=366, prefixes_checked=123,
        feature_matrices_checked=2, original_array_count=27, prepared_array_count=30,
        max_score_error=max(float(np.max(abs(saved_scores[n]-scored['scores'][n]))) for n in prepare.NAMES),
        source_support=support, changed_support=changed, producer_modules_imported=False,
        returns_or_nav_computed=False, strict_source_admitted=False)


def outcome_audit(root, plan_sha):
    reader = Reader(root); directory = reader.path(OUT)
    plan, pins, paths = bind(reader, plan_sha, 'outcome')
    prepared = reader.json(directory/'INDEPENDENT_PREPARE.json')
    require(prepared['passed'] is True and prepared['plan_sha256'] == plan_sha and
            prepared['cache_sha256'] == reader.sha(directory/'CACHE.json') and
            all(prepared['checks'].get(k) is True for k in PREPARE_CHECKS), 'Current independent prepare receipt')
    require(reader.verify(prepared['source_hashes']) == {str(p):reader.sha(p) for p in source_files(reader.root)}, 'Prepared verifier closure')
    reader.verify(prepared['artifact_hashes'])
    frozen = reader.json(directory/'FROZEN.json')
    required_frozen = {'PLAN.json','CACHE.json','PRETRAIN_ORDERS.json','INDEPENDENT_PREPARE.json',
                       'TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','SUPPORT_AUDIT.json'}
    require(set(frozen['hashes']) == required_frozen and frozen['plan_sha256'] == plan_sha and
            frozen['selection_count'] == 1 and frozen['later34_diagnostic_only'] is True, 'Frozen one-time selection')
    reader.verify({str(directory/k):h for k,h in frozen['hashes'].items()})
    marker = reader.json(directory/'DIAGNOSE_STARTED.json')
    require(marker['plan_sha256'] == plan_sha and marker['frozen_sha256'] == reader.sha(directory/'FROZEN.json'), 'Diagnostic start binding')
    require(prepare.ns(marker['utc']) < prepare.ns(datetime.now(timezone.utc).isoformat()), 'Diagnostic started before audit')
    meta, events, rows, originals, calendar = source_inputs(reader, plan, pins, paths)
    orders = reader.json(directory/'PRETRAIN_ORDERS.json')
    require(orders[prepare.NAMES[0]] == reader.json(paths['original_orders']), 'Original full incumbent orders')
    feature = reader.json(directory/'FEATURE_AUDIT.json')
    support = outcome.support_oracle(feature['cells'])
    require(support['passed'] is True, 'Pre-outcome support gate')
    changed = {name: outcome.changed_support_oracle(orders,feature['cells'],name) for name in prepare.NAMES[1:]}
    require(reader.json(directory/'SUPPORT_AUDIT.json') == dict(source=support,changed=changed), 'Support receipt unchanged')
    close, opening = prices(reader, plan, pins, paths, meta)
    expected = outcome.nine_path_oracle(close, opening, orders, meta)
    names = ['RESULTS.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json']
    names += [prefix+'_'+suffix+'.json' for prefix in expected['details'] for suffix in ('DAILY','TRADES','HOLDINGS')]
    artifacts = {name:reader.json(directory/name) for name in names}
    require(artifacts['RESULTS.json']['plan_sha256'] == plan_sha and artifacts['RESULTS.json']['selection'] == frozen['selection'],
            'Results bind frozen selection')
    comparison = outcome.compare_nine_paths(expected, artifacts)
    selected = outcome.select_first88_oracle(expected['training'],expected['opened'],support,changed)
    require(frozen['selection'] == selected, 'Independent single first88 selection')
    extra_names = (*CACHE_FILES,'PLAN.json','CACHE.json','PREPARE_STARTED.json','SELECT_STARTED.json',
                   'INDEPENDENT_PREPARE.json','FROZEN.json','DIAGNOSE_STARTED.json',*names)
    selection_start = reader.json(directory/'SELECT_STARTED.json')
    require(selection_start['plan_sha256'] == plan_sha and prepare.ns(selection_start['utc']) <= prepare.ns(marker['utc']),
            'Selection start precedes diagnostics')
    extra = {str(directory/name):reader.sha(directory/name) for name in extra_names}
    finish(reader, plan, extra)
    return dict(schema='basic_shares159_independent_outcome_v1',passed=True,stage='outcome',plan_sha256=plan_sha,
        created_utc=datetime.now(timezone.utc).isoformat(),frozen_sha256=reader.sha(directory/'FROZEN.json'),
        results_sha256=reader.sha(directory/'RESULTS.json'), paths_verified=9, orders_verified=366,
        checks=dict.fromkeys(OUTCOME_CHECKS,True),selection=selected,comparison=comparison,
        source_hashes={str(p):reader.sha(p) for p in source_files(reader.root)},artifact_hashes=dict(reader.seen),
        paths_checked=9,orders_checked=366,selection_count=1,producer_modules_imported=False,strict_source_admitted=False)


def contract():
    return dict(status='REGISTERED_PHASE_GUARDED_VERIFIER', actual_financial_inputs_read=0,
        actual_factors_or_returns_computed=0,
        source_adapter='44 events, sourcegate v2 rows, original5978 rows, eligible122x49 and pinned sessions/integer close nanoseconds',
        checks=['original27_arrays_byte_identical', 'original5978_rows_preserved', 'two_clock_tminus42_latest_invalid_no_fallback',
                'original_eligible_and_distinct_issuer_before_transform', 'exact_fraction_raw_and_rank',
                'centered_five_percent_magnitude_and_rank_hands', 'once_point01_delta_then_original_episode',
                'all123_score_prefixes', 'first88_source_and_changed_holdings_support',
                'first88_once_selection', 'all9_daily_nav_trade_holdings_metrics', 'causal_first88_replay'],
        remaining=['actual_registered_preparation_and_outcome_execution', 'protected_archive_closure_by_host'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', choices=('prepare','outcome'))
    parser.add_argument('--plan-sha')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(contract(), indent=2))
    else:
        require(args.phase is not None and args.plan_sha is not None, 'Explicit phase and PLAN SHA required')
        destination=ROOT/OUT/('INDEPENDENT_PREPARE.json' if args.phase=='prepare' else 'INDEPENDENT_OUTCOME.json')
        require(not destination.exists(), 'Write-once independent receipt already exists')
        report=(preparation_audit if args.phase=='prepare' else outcome_audit)(ROOT,args.plan_sha)
        with destination.open('x') as handle:
            json.dump(report,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
        print(json.dumps(dict(passed=True,stage=args.phase,receipt=str(destination),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
