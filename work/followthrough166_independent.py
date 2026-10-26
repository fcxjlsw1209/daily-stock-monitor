"""Independent166 verifier. Real I/O requires exact registered PLAN and phase.

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

import followthrough166_independent_prepare as prepare
import followthrough166_independent_outcome as outcome

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_followthrough166'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CACHE_FILES = ('DATA.npz','SCORES.npz','META.json','FEATURE_AUDIT.json','STATE_VALUES.json',
               'PRETRAIN_ORDERS.json','PRICE_PARITY.json','SUPPORT_AUDIT.json')
PREPARE_CHECKS = ('original27_arrays_byte_identical','original122x49_axis_preserved',
    'source_review_and_calendar_closure','exact_official_212_adjusted_231_volume_210_pairs_skip42',
    'independent_partial_followthrough_and_return_persistence','independent_current_pool_midrank_and_common_mask',
    'matched_positive_two_percent_deltas','once_delta_then_original_episode',
    'all123_feature_and_score_prefixes','exact_adjusted_volume_window_and_original_array_poison_invariance',
    'initial_and_zero_delta_exact_incumbent','all366_ordered_holdings_and_scores','baseline_full_orders_unchanged',
    'source_support_independently_verified','raw_price_parity_independently_verified','inherited_source_limitations_preserved')
OUTCOME_CHECKS = ('all9_daily_nav_trade_holdings_metrics','first88_once_selection','causal_first88_replay',
    'source_and_changed_support_verified','diagnostic_start_bound','baseline_full_orders_unchanged',
    'all_original_inputs_and_frozen_artifacts_unchanged','inherited_source_limitations_preserved')
require = prepare.require


def source_files(root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    files = [root/'work'/name for name in (
        'followthrough166_independent.py', 'followthrough166_independent_prepare.py',
        'followthrough166_independent_outcome.py', 'test_followthrough166_independent.py',
        'monthly_paired_target143_outcome_review.py', 'monthly_ridge_support70_result_review.py',
        'monthly_latent_state134_prepare_review.py')]
    files.append(root/'outputs/monthly_stability_monitor/inputs.py')
    modules = ('exchange_calendars.exchange_calendar_xnys', 'exchange_calendars.exchange_calendar',
               'exchange_calendars.us_holidays', 'exchange_calendars.calendar_helpers', 'pandas.tseries.holiday')
    files.extend(Path(importlib.import_module(name).__file__).resolve() for name in modules)
    return sorted(set(files))


def measurement_contract():
    return dict(
        arms=list(prepare.NAMES), promotable_arm=prepare.NAMES[1], nonpromotable_control=prepare.NAMES[2],
        adjusted_price_offsets=[-253, -42], volume_offsets=[-273, -43],
        paired_session_offsets=[-252, -43], adjusted_window_prices=212,
        volume_window_values=231, activity_baseline_sessions=21,
        window_returns=210, window_pairs=210, skip_sessions=42,
        r='r_s=diff(log(adjusted_close)) for s=t-252..t-42',
        a='a_s=log(V_s)-mean(log(V_u),u=s-21..s-1) for s=t-252..t-43',
        y='y_s=r_(s+1) for s=t-252..t-43',
        q='product of centered unit-L2 r_s and centered unit-L2 a_s; then center and normalize q to unit L2',
        centering='for each r,a,y and then q: scale by max(abs(v)); center by arithmetic mean; reject zero scale or centered L2 <=1e-12 times scaled uncentered L2; normalize centered vector to unit L2',
        residualization='Gram-Schmidt: b=unit_a-dot(unit_a,unit_r)*unit_r; reject L2(b)<=1e-12; normalize b; for unit_q and unit_y subtract projection on unit_r, then projection of that residual on b; reject residual L2 <=1e-12 and normalize',
        raw_followthrough='dot(unit_residual_q,unit_residual_y); Pearson partial correlation with intercept controlling r and a',
        raw_control='dot(unit_r,unit_y); Pearson return persistence',
        correlation_roundoff='clip only excess no greater than 64*float64_machine_epsilon to [-1,1]; larger excess makes both unavailable',
        relative_degeneracy_epsilon=1e-12,
        rank='separate average ascending percentile minus .5 for followthrough/control on shared finite original49 pool including ETFs before trading eligibility',
        followthrough_delta='0.04*ranks_followthrough', control_delta='0.04*ranks_control',
        missing='both raw values unavailable and both deltas exactly zero for any missing/nonfinite/nonpositive consumed source, near-constant vector, dependent controls or residual degeneracy; no fill',
        max_abs_delta=.02, score_inputs='both deltas already include the complete dose',
        control_matching='same raw sources, paired sessions, common support, rank transform, positive dose and missing handling; control is never promotable',
        population='current original49 pool including ETFs; current trading eligibility and all original27 arrays remain unchanged',
        prices='exactly aligned adj and volume DataFrames from pinned local panel; no raw close; slice official windows before numeric conversion; no source outside consumed windows read',
        placement='unchanged_Local67_config4_pre_plus_one_delta_then_one_original_episode',
        original_arrays=27, extra_arrays=list(prepare.EXTRA), prepared_arrays=29,
        initial_partial='exact_incumbent; measurable raw_followthrough/raw_control and ranks may remain in audit; common support false',
        zero_delta='exact_incumbent', core='unchanged_long_momentum_skip42',
        portfolio='original_EW3_two_half_less_than_two_cash; fixed_intramonth_units',
        strict_pit_vintage=False, strict_pit_membership=False,
        historical_series='inherited price vintages and estimated Top32 plus assumed theme17 historical membership proxy',
        new_predictive_model_fits=0, nuisance_estimation='within-window intercept and two-control partial correlation only')


def selection_contract():
    return dict(
        selection_count=1, first88_only=True, early53=True, recent35=True,
        later34_diagnostic_only=True, prefix_reselection=False,
        source_support='all structural and missing-neutral validation passes; coverage is reported without a threshold',
        minimum_common_support_cells=None, initial_partial_exempt=True,
        minimum_close_cagr_gain=.005, cagr_epsilon=1e-12, open_cost_noninferior=True,
        sharpe_tolerance=.05, drawdown_tolerance=.03, era_cagr_tolerance=.025,
        guards_apply_to='followthrough vs incumbent: same_close requires .005 CAGR gain and next_open_5bp is noninferior; followthrough vs persistence_control: both same_close and next_open_5bp are noninferior; every comparison uses the same Sharpe/drawdown/era guards',
        minimum_changed_held_set_months=3, minimum_changed_each_era=1,
        choice='volume_followthrough only if every incumbent and matched-control guard passes; otherwise incumbent_local67; persistence_control is never selected',
        promotable_arm=prepare.NAMES[1], nonpromotable_control=prepare.NAMES[2],
        execution_paths=9)



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
    require(plan['round'] == 166 and plan['status'] == 'REGISTERED' and plan['arms'] == list(prepare.NAMES),
            'Fixed registered round166')
    require(reader.path(plan['output_directory']) == directory, 'Exact study output directory')
    require(plan['runtime'] == dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__), 'Pinned runtime')
    require(plan['measurement'] == measurement_contract() and plan['selection_contract'] == selection_contract(),
            'Fixed prospective formulas and single selection')
    for key in ('actual_features_before_registration', 'actual_orders_before_registration', 'actual_returns_before_registration'):
        require(type(plan[key]) is int and plan[key] == 0, 'No preregistration numerical work')
    require(plan['no_observer_writes'] is True, 'Research scope')
    if stage == 'prepare':
        require(not any((directory/name).exists() for name in ('SELECT_STARTED.json', 'TRAIN_TRIALS.json',
            'TRAIN_OPEN_TRIALS.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json', 'RESULTS.json', 'INDEPENDENT_OUTCOME.json')) and
                not any(directory.glob('*_DAILY.json')) and not any(directory.glob('*_TRADES.json')) and
                not any(directory.glob('*_HOLDINGS.json')), 'Prepare must precede all outcomes')
    registration(reader.root, plan, plan_sha)
    pins = reader.verify(plan['hashes'])
    for path in source_files(reader.root):
        require(pins.get(str(path)) == reader.sha(path), 'Unregistered independent implementation: '+str(path))
    paths = {}
    for key, ref in plan['inputs'].items():
        path = reader.path(ref['path'])
        require(reader.sha(path) == ref['sha256'] == pins.get(str(path)), 'Named input binding: '+key)
        paths[key] = path
    fixed = dict(meta='outputs/monthly_local_linear70/META.json',policy='outputs/monthly_local_linear70/INCUMBENT_POLICY.json',
        original_orders='outputs/monthly_local_linear70/local_linear_ORDERS.json',prices='outputs/monthly_2016_extension/PRICES.npz',
        training_meta='outputs/monthly_architecture_restart/META.json',
        proposal='work/followthrough166_proposal.json',calendar='outputs/monthly_paired_target143/CALENDAR.json',
        source_review='work/followthrough166_design_review.json',kernel_source='work/followthrough166_kernel.py',
        protected_manifest='work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json')
    require(set(paths)==set(fixed),'Exact named input schema')
    for key,value in fixed.items(): require(paths[key]==reader.path(value),'Original fixed input path: '+key)
    require(reader.sha(paths['protected_manifest'])=='89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb','Original protected manifest identity')
    require(plan['source_limitations']==reader.json(paths['proposal'])['source_limitations'],'Declared source limitations unchanged')
    require(all(reader.json(paths['policy']).get(k) == v for k, v in prepare.old_helper().POLICY.items()), 'Original episode policy')
    cache = reader.json(directory/'CACHE.json')
    require(set(cache) == set(CACHE_FILES), 'Complete exact preparation cache')
    reader.verify({str(directory/name): value for name, value in cache.items()})
    start = reader.json(directory/'PREPARE_STARTED.json')
    require(start['plan_sha256'] == plan_sha and prepare.ns(plan['registered_utc']) <= prepare.ns(start['utc']), 'Preparation start PLAN/time binding')
    return plan, pins, paths


def registration(root,plan,plan_sha):
    # Central state is mutable control metadata, not an immutable numerical input.
    control=Reader(root); directory=control.path(OUT)
    require(control.path(plan['registration_precondition']['path'])==control.path(STATE),'Fixed registration state path')
    before=control.json(directory/'STATE_BEFORE.json')
    require(control.sha(directory/'STATE_BEFORE.json')==plan['registration_precondition']['sha256'],'Exact preregistration state')
    require(len(before['completed'])==165 and sum(x['trials'] for x in before['completed'])==1010 and not before['registered_queue'],'Expected preceding165/1010 state')
    current=control.json(STATE)
    entry=dict(round=166,directory='monthly_followthrough166',plan=dict(path=str(directory/'PLAN.json'),sha256=plan_sha),registered_utc=plan['registered_utc'])
    require(current['registered_queue']==[entry] and current['completed']==before['completed'],'Current exact registered queue required')
    require(current.get('best_training_selected_candidate')==before.get('best_training_selected_candidate'),'Prior incumbent metadata must remain exact')


def source_inputs(reader, plan, pins, paths):
    review=reader.json(paths['source_review'])
    require(review['passed'] is True and review['conditional_research_sources_ready'] is True and
            review['strict_pit_vintage'] is False and review['strict_pit_membership'] is False and
            review['strategy_admitted'] is False and review['proposal_sha256']==reader.sha(paths['proposal']),
            'Conditional metadata review and exact proposal binding')
    for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed'):
        require(type(review[key]) is int and review[key]==0,'Source review was metadata-only')
    checked=reader.verify(review['input_hashes'])
    require(checked==reader.verify(review['source_hashes']) and all(pins.get(p)==h for p,h in checked.items()),'Registered complete source review closure')
    require(review['summary']['all_overlap_sets_equal'] is True and review['summary']['noninitial_dates_verified_monthend'] is True and
            review['summary']['historical_names_absent_from_price_panel']==[], 'Exact original pool metadata coverage')
    meta=reader.json(paths['meta'])
    require(meta==reader.json(reader.path(OUT)/'META.json') and
            reader.sha(paths['meta'])=='3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4','Original metadata bytes')
    require(len(meta['dates'])==len(meta['symbols'])==len(meta['ends'])==122 and all(len(x)==49 for x in meta['symbols']) and
            meta['train_months']==88 and meta['early_months']==53 and meta['dates'][0]=='2016-08-17' and
            meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24','Original full122x49 metadata')
    training=reader.json(paths['training_meta'])['training']
    require(len(training['dates'])==len(training['symbols'])==180,'Original historical monthly pool axis')
    saved=reader.json(paths['calendar']);closure=reader.verify(saved['provenance']['source_hashes'])
    require(len(closure)==9 and all(pins.get(p)==h for p,h in closure.items()),'Complete calendar9 closure')
    exception=reader.path('outputs/index_union/CALENDAR_EXCEPTIONS.json')
    require(reader.sha(exception)==pins.get(str(exception)),'Calendar exception pin')
    closed=set()
    for row in reader.json(exception)['closures']:
        if 'XNYS' in row['markets']:
            path=reader.path('outputs/index_union')/row['source_file']
            require(reader.sha(path)==row['source_sha256']==pins.get(str(path)),'Official calendar closure evidence')
            closed.add(row['date'])
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    schedule=XNYSExchangeCalendar(start=saved['start'],end=saved['end']).schedule
    sessions=[str(d.date()) for d in schedule.index if str(d.date()) not in closed]
    require(sessions==saved['sessions'],'Independent XNYS session reconstruction')
    return meta,sessions,training


def prices(reader, plan, pins, paths, meta, parity_check=False):
    raw = reader.verify(plan['raw_source_hashes'])
    require(len(raw) == 64 and all(pins.get(p) == h for p, h in raw.items()), 'Original64 raw price closure')
    cached = reader.npz(paths['prices'])
    require(set(cached) == {'dates', 'symbols', 'close', 'opening'}, 'Original price cache schema')
    if parity_check:
        import monthly_latent_state134_prepare_review as old
        loader_path = reader.path(old.LOADER)
        require(reader.sha(loader_path) == old.LOADER_SHA == pins.get(str(loader_path)), 'Original price loader')
        spec = importlib.util.spec_from_file_location('followthrough166_independent_price_loader', loader_path)
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
    if parity_check:
        aligned = dict(adj=close,volume=panel['volume'].loc[close.index,close.columns])
        return close, opening, aligned
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
    meta, calendar, training = source_inputs(reader,plan,pins,paths)
    original_path=reader.path('outputs/monthly_local_linear70/DATA.npz')
    require(reader.sha(original_path)==pins.get(str(original_path)),'Original27 cache bound')
    original,arrays=reader.npz(original_path),reader.npz(directory/'DATA.npz')
    prepare.verify_original_arrays(original,arrays)
    close,opening,panel=prices(reader,plan,pins,paths,meta,parity_check=True)
    expected=prepare.features_oracle(meta,calendar,training,panel)
    feature=reader.json(directory/'FEATURE_AUDIT.json')
    prepare.verify_feature_output(expected,arrays,feature,reader.json(directory/'STATE_VALUES.json'))
    support=outcome.support_oracle(expected['audit'],meta,expected['arrays'])
    require(support['passed'] is True,'Complete source support before any outcome')
    scored = prepare.score_oracle(original, meta, expected['arrays'])
    orders = prepare.order_oracle(meta, scored)
    saved_scores, saved_orders = reader.npz(directory/'SCORES.npz'), reader.json(directory/'PRETRAIN_ORDERS.json')
    require(set(saved_scores) == set(saved_orders) == set(prepare.NAMES), 'Three prepared score/order arms')
    for name in prepare.NAMES:
        require(saved_scores[name].shape == (122,49) and saved_scores[name].dtype == scored['scores'][name].dtype, 'Score shape/dtype')
        np.testing.assert_allclose(saved_scores[name], scored['scores'][name], atol=1e-12, rtol=0)
    prepare.verify_orders(saved_orders,orders)
    require(saved_orders[prepare.NAMES[0]] == reader.json(paths['original_orders']), 'Exact original full incumbent orders')
    changed = {name: outcome.changed_support_oracle(orders, name) for name in prepare.NAMES[1:]}
    require(reader.json(directory/'SUPPORT_AUDIT.json') == dict(source=support, changed=changed), 'Independent support counts')
    for n in range(123):
        prefix=prepare.features_oracle(meta,calendar,training,panel,n)
        for key in prepare.EXTRA:
            require(prefix['arrays'][key].tobytes()==expected['arrays'][key][:n].tobytes(),'Exact feature prefix')
        prior=prepare.score_oracle(original,meta,prefix['arrays'],n)
        for name in prepare.NAMES:
            require(prior['choices'][name]==scored['choices'][name][:n] and
                    prior['scores'][name].tobytes()==scored['scores'][name][:n].tobytes(),'Prefix score/order invariant')
        poisoned_prices=prepare.poison_outside_windows(panel,meta,calendar,n)
        future=prepare.features_oracle(meta,calendar,training,poisoned_prices,n)
        prepare.compare(future['audit']['event_history'],prefix['audit']['event_history'],'future-source audit prefix')
        for key in prepare.EXTRA:
            require(future['arrays'][key].tobytes()==prefix['arrays'][key].tobytes(),'Future price poison invariant')
        poisoned_arrays={k:v.copy() for k,v in original.items()}
        for value in poisoned_arrays.values():
            if value.ndim and value.shape[0]==122:
                if value.dtype.kind=='f':value[n:]=np.nan
                elif value.dtype.kind=='b':value[n:]=~value[n:]
        future_scores=prepare.score_oracle(poisoned_arrays,meta,prefix['arrays'],n)
        for name in prepare.NAMES:
            require(future_scores['choices'][name]==prior['choices'][name] and
                    future_scores['scores'][name].tobytes()==prior['scores'][name].tobytes(),'Future original-array poison invariant')
    for name, key in zip(prepare.NAMES[1:], prepare.HANDS):
        for i in range(122):
            if not expected['arrays'][key][i].any():
                require(scored['choices'][name][i] == scored['choices'][prepare.NAMES[0]][i] and
                        scored['scores'][name][i].tobytes() == scored['scores'][prepare.NAMES[0]][i].tobytes(), 'Exact neutral parity')
    extra = {str(directory/name): reader.sha(directory/name) for name in (*CACHE_FILES, 'PLAN.json', 'CACHE.json', 'PREPARE_STARTED.json')}
    finish(reader, plan, extra)
    registration(reader.root,plan,plan_sha)
    return dict(schema='followthrough166_independent_prepare_v1', passed=True, stage='prepare', plan_sha256=plan_sha,
        created_utc=datetime.now(timezone.utc).isoformat(), cache_sha256=reader.sha(directory/'CACHE.json'),
        checks=dict.fromkeys(PREPARE_CHECKS, True), source_hashes={str(p):reader.sha(p) for p in source_files(reader.root)},
        artifact_hashes=dict(reader.seen), candidate_cells_checked=5978, orders_checked=366, prefixes_checked=123,
        feature_matrices_checked=2, original_array_count=27, prepared_array_count=29,
        max_score_error=max(float(np.max(abs(saved_scores[n]-scored['scores'][n]))) for n in prepare.NAMES),
        source_support=support, changed_support=changed, producer_modules_imported=False,
        returns_or_nav_computed=False, strict_source_admitted=False, strict_pit_vintage=False,strict_pit_membership=False)


def outcome_audit(root, plan_sha):
    reader = Reader(root); directory = reader.path(OUT)
    plan, pins, paths = bind(reader, plan_sha, 'outcome')
    prepared = reader.json(directory/'INDEPENDENT_PREPARE.json')
    require(prepared['passed'] is True and prepared['plan_sha256'] == plan_sha and
            prepared['cache_sha256'] == reader.sha(directory/'CACHE.json') and
            all(prepared['checks'].get(k) is True for k in PREPARE_CHECKS), 'Current independent prepare receipt')
    require(reader.verify(prepared['source_hashes']) == {str(p):reader.sha(p) for p in source_files(reader.root)}, 'Prepared verifier closure')
    require(prepared.get('producer_modules_imported') is False and prepared.get('returns_or_nav_computed') is False, 'Independent preparation scope')
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
    selection_start=reader.json(directory/'SELECT_STARTED.json')
    prepare_start=reader.json(directory/'PREPARE_STARTED.json')
    require(selection_start['plan_sha256']==plan_sha and
            prepare.ns(prepare_start['utc']) <= prepare.ns(prepared['created_utc']) <= prepare.ns(selection_start['utc']) <= prepare.ns(marker['utc']),
            'Independent preparation must precede selection and diagnostics')
    require(prepared['orders_checked']==366 and prepared['prefixes_checked']==123 and
            prepared['original_array_count']==27 and prepared['prepared_array_count']==29 and
            prepared['strict_pit_vintage'] is False,'Complete independent preparation scope')
    meta,calendar,training=source_inputs(reader,plan,pins,paths)
    orders=reader.json(directory/'PRETRAIN_ORDERS.json')
    require(orders[prepare.NAMES[0]]==reader.json(paths['original_orders']),'Original full incumbent orders')
    feature=reader.json(directory/'FEATURE_AUDIT.json')
    cached=reader.npz(directory/'DATA.npz')
    support=outcome.support_oracle(feature,meta,{key:cached[key] for key in prepare.EXTRA})
    require(support['passed'] is True,'Pre-outcome support gate')
    changed={name:outcome.changed_support_oracle(orders,name) for name in prepare.NAMES[1:]}
    require(reader.json(directory/'SUPPORT_AUDIT.json')==dict(source=support,changed=changed),'Support receipt unchanged')
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
    extra = {str(directory/name):reader.sha(directory/name) for name in extra_names}
    finish(reader, plan, extra)
    registration(reader.root,plan,plan_sha)
    return dict(schema='followthrough166_independent_outcome_v1',passed=True,stage='outcome',plan_sha256=plan_sha,
        created_utc=datetime.now(timezone.utc).isoformat(),frozen_sha256=reader.sha(directory/'FROZEN.json'),
        results_sha256=reader.sha(directory/'RESULTS.json'), paths_verified=9, orders_verified=366,
        checks=dict.fromkeys(OUTCOME_CHECKS,True),selection=selected,comparison=comparison,
        source_hashes={str(p):reader.sha(p) for p in source_files(reader.root)},artifact_hashes=dict(reader.seen),
        paths_checked=9,orders_checked=366,selection_count=1,producer_modules_imported=False,strict_source_admitted=False, strict_pit_vintage=False,strict_pit_membership=False)


def contract():
    return dict(status='REGISTERED_PHASE_GUARDED_VERIFIER',actual_inputs_read=0,actual_factors_or_returns_computed=0,
                prepare_checks=list(PREPARE_CHECKS),outcome_checks=list(OUTCOME_CHECKS),
                strict_pit_vintage=False,strict_pit_membership=False,remaining=['registered_actual_prepare_and_outcome','protected_archive_closure_by_host'])


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

