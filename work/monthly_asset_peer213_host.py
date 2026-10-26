"""Round213: same-asset-type versus all-asset peer residual transfer.

No actual fits, forecasts, orders or returns before the registered PLAN gate.
The original local-quarter slot and one original episode are retained.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_asset_peer213'
LOCAL = ROOT/'outputs/monthly_local_linear70'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
PRIOR = ROOT/'outputs/monthly_monotone_path212'
SOURCE = ROOT/'outputs/monthly_identity_transfer206'
SOURCE_KEYS = {'raw_source_log','raw_local_linear','raw_local_constant','raw_candidate','raw_control','available','local_available'}
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67','same_asset_peer_transfer','all_asset_peer_control')
RAW = {NAMES[1]:'candidate', NAMES[2]:'control'}
RAW_KEYS = ('source_log','local_linear','local_constant','candidate','control')
TARGETS = ('simple_excess','log_excess','vol_scaled','loss_utility')
CORRECTION_ACCOUNTING = 'One pooled log Ridge refit per predicted annual cohort; own constant, all-peer and supported same-type corrections counted separately; cached four-target/local-linear fits are not repeated; prefix/oracle repeats excluded.'
TRAIN = 88
MODEL_PREFIXES = (0,1,53,88,122)
MODEL_COUNT_FIELDS = ('annual_pooled_log_refits','cached_forecast_rows','own_weight_evaluations','all_peer_query_evaluations','all_peer_month_evaluations','all_peer_row_observations','same_type_supported_queries','same_type_fallback_queries','same_type_month_evaluations','same_type_row_observations','actual_regression_solves')
CONTEXT = ROOT/'work/monthly_asset_peer213_context.json'
CONTEXT_SHA = '44b2225c237c6306f0cf531550e2202ce57ae6357af55ef244c3cc87ef6c6f8d'
COVERAGE = ROOT/'work/after212_asset_peer_coverage.json'
COVERAGE_SHA = '613df99fe63918c43d6175a2de72105e6f9ca174e5339fe604eee032cac3770b'
SOURCE_REVIEW = ROOT/'work/asset_peer213_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '7af413ea369b4d95a08a929798f65456b1e1a516c27b0ae260295894ca60c236'
TEST_RECEIPT = ROOT/'work/monthly_asset_peer213_TESTS.json'
EXTERNAL_RUNTIME = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED = {
 'PLAN.json':'f8d6f427f3b2e7e96dc550ced3f0e5daca9e4a6d439144c18e39d6a2e79abdd3',
 'COMPLETION.json':'3a80102475de0f488c081192d8e3f146f2290cd0a498811ade0f4cd8837201cb',
 'results/COMPLETION.json':'9272d87e463d61547eac92dd7933c3f015532adb6867a0e417ca139c82cfd01e',
 'RESEARCH_REVIEW.json':'328f1fbbc9e5ee999a9ca8a4fd9908392932135e80183d9c4b4310be5ab9d247'}
MODEL_FLAGS = ('passed','independent_augmented_least_squares','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked','four_target_common_cohort_checked','availability_unchanged','own_history_threshold_checked','matched_month_identity_checked','query_security_excluded','peer_kernel_checked','own_month_mass_preserved','common_shrinkage_penalty_checked','no_future_or_outcome_inputs','audit_checked','fixed_asset_types_checked','same_type_peer_domain_checked','whole_query_fallback_checked','cached_all_peer_control_exact','cached_source_and_local_reused','no_other_source_targets_or_local_refits')
SLOT_FLAGS = ('passed','full_original_universe','includes_eligible_etfs','no_local_history_filter',
 'replacement_not_additive_local_slot','paired_whole_row_readiness','initial_partial_unchanged','exact_rank_rebuild')
PREPARED_FILES = {'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json',
 'META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES = {arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS
 for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')} | {'RESULTS.json','INDEPENDENT_OUTCOME.json'}
sha,read,require,dump = ev.sha,ev.read,ev.require,ev.dump

def now():
    return datetime.now(timezone.utc).isoformat()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def local():
    return module('monthly_asset_peer213_original', LOCAL / 'study.py')

def contract():
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective context')
    require(sha(COVERAGE)==COVERAGE_SHA,'Changed prospective coverage')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),
        candidate=NAMES[1],control_nonpromotable=NAMES[2],period=context['evaluation']['period'],
        training_months=88,later_months=34,model_prefixes=list(MODEL_PREFIXES),all123_order_prefixes=True,
        model_fit_accounting=context['formation']['fit_accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,
        original_pool_preserved=True,no_parameter_search=True,no_new_score_budget=True)


def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_asset_peer213_host.py',
        'test_monthly_asset_peer213_host.py','monthly_asset_peer213_model.py',
        'test_monthly_asset_peer213_model.py','monthly_asset_peer213_independent.py',
        'test_monthly_asset_peer213_independent.py','monthly_asset_peer213_context.json',
        'after212_asset_peer_coverage.json','asset_peer213_SOURCE_REVIEW.json','asset_peer213_source_review.py',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_asset_peer213_model as model
    import monthly_asset_peer213_independent as independent
    return direct+sorted((set(model.dependency_files())|set(independent.dependency_files()))-set(direct))


def verify(pins):
    require(isinstance(pins,dict) and pins,'Nonempty source pins required')
    for name,digest in pins.items():
        path=Path(name);path=(path if path.is_absolute() else ROOT/path).resolve()
        require('CloudStorage' not in path.parts,'Cloud input forbidden')
        require(path.is_relative_to(ROOT) or EXTERNAL_RUNTIME.get(str(path))==digest,
                'Unapproved external dependency: '+str(path))
        require(path!=CENTRAL.resolve(),'Mutable central cannot be a persistent source input')
        require(path.is_file() and sha(path)==digest,'Changed or missing source: '+str(path))

def bind(pins, path, digest=None):
    key, actual = str(path), sha(path)
    require(digest is None or actual == digest, 'Source differs from declared digest: ' + key)
    require(key not in pins or pins[key] == actual, 'Conflicting inherited source digest: ' + key)
    pins[key] = actual

def predecessor_binding(state):
    pins = {}
    for name, digest in PRIOR_EXPECTED.items(): bind(pins, PRIOR / name, digest)
    reference = state['completed'][-1]['completion']
    require(Path(reference['path']).resolve() == (PRIOR/'COMPLETION.json').resolve()
            and reference['sha256'] == PRIOR_EXPECTED['COMPLETION.json'], 'Predecessor central receipt')
    closed = read(PRIOR/'COMPLETION.json')
    require(closed.get('status') == 'COMPLETED_INDEPENDENTLY_VERIFIED' and
            closed.get('plan_sha256') == PRIOR_EXPECTED['PLAN.json'] and closed.get('protected_files') == 21444 and
            closed.get('observers_modified') is False, 'Predecessor closure incomplete')
    bind(pins, PRIOR/'results/COMPLETION.json', closed['result_completion_sha256'])
    result = read(PRIOR/'results/COMPLETION.json')
    require(result['status']=='COMPLETE' and result['plan_sha256']==PRIOR_EXPECTED['PLAN.json'] and
            result['paths']==9 and result['slices']==45, 'Predecessor results incomplete')
    for name, digest in read(CONTEXT)['evidence_sha256'].items(): bind(pins, ROOT/name, digest)
    return dict(directory=PRIOR.name, input_sha256=pins)

def commit_state(before, after, tag):
    expected = hashlib.sha256(before).hexdigest()
    require(CENTRAL.read_bytes() == before, 'Concurrent central update')
    backup = CENTRAL.parent / ('PRE_MONTHLY_ASSET_PEER213_' + tag + '_' + expected + '.json')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(before)
    require(sha(backup) == expected, 'Central backup mismatch')
    with tempfile.NamedTemporaryFile(dir=CENTRAL.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    try:
        require(CENTRAL.read_bytes() == before, 'Concurrent central update before commit')
        os.replace(temporary, CENTRAL)
    finally:
        temporary.unlink(missing_ok=True)

def code_review_binding():
    path = ROOT / 'work/asset_peer213_CODE_REVIEW.json'
    review = read(path)
    expected = {str(p.relative_to(ROOT)): sha(p) for p in code_files()[:6]}
    require(review.get('passed') is True and review.get('unresolved_blockers') == [], 'Independent numerical code review incomplete')
    require(review.get('context_sha256')==CONTEXT_SHA,'Independent review context differs')
    require(review.get('code_sha256') == expected, 'Numerical files changed after independent review')
    for key in ('actual_features_computed', 'actual_model_fits_computed', 'actual_portfolio_returns_computed', 'production_files_modified', 'strategy_registered_by_reviewer'):
        require(review.get(key) is False, 'Independent review scope differs: '+key)
    require(isinstance(review.get('input_sha256'),dict) and review['input_sha256'],'Independent review source pins required')
    pins = {str(path): sha(path)}
    for name, digest in review['input_sha256'].items():
        require((Path(name) if Path(name).is_absolute() else ROOT/name).resolve()!=CENTRAL.resolve(),'Mutable central cannot be a persistent review input')
        bind(pins, Path(name) if Path(name).is_absolute() else ROOT/name, digest)
    verify(pins)
    return dict(path=str(path), sha256=sha(path), input_sha256=pins)

def gate(expected):
    require(sha(OUT / 'PLAN.json') == expected, 'Explicit PLAN SHA required')
    p = read(OUT / 'PLAN.json')
    require(p['contract'] == contract() and p['runtime'] == ev.runtime(), 'Frozen contract/runtime changed')
    verify(p['input_sha256'])
    require(p['model_source_binding'] == model_source_binding(p['raw_source_hashes']), 'Registered model source binding changed')
    require(p['numerical_code_review']==code_review_binding(),'Registered independent code review changed')
    require(sha(OUT / 'STATE_BEFORE.json') == p['prior_central_sha256'], 'Changed pre-registration state')
    state = read(CENTRAL)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=expected)], 'Central registration mismatch')
    state_check(state)
    require(p.get('predecessor_binding') == predecessor_binding(state), 'Registered predecessor bindings changed')
    return p

def require_outcome_audit(check):
    for key in ('passed', 'independent_accounts', 'independent_of_production_model_and_engine'):
        require(check.get(key) is True, 'Independent outcome audit missing ' + key)
    require(check.get('checked_paths') == 9 and check.get('checked_slices') == 45 and
            check.get('terminal_liquidation') is False, 'Independent outcome coverage')
    require(check.get('accounting_source_sha256') == sha(ROOT / 'work/ptr169_results_independent.py'), 'Accounting source changed')
    require(set(check.get('paths', {})) == set(NAMES), 'Missing independent arms')
    for paths in check['paths'].values():
        require(set(paths) == {p[0] for p in ev.PATHS}, 'Missing independent paths')
        for path in paths.values():
            require(path.get('passed') is True and path.get('monthly_executions') == 122 and
                    set(path.get('slices', {})) == {'full122', 'first88', 'early53', 'recent35', 'later34'}, 'Incomplete path audit')

def train_metrics(engine, eq, meta):
    answer = {}
    for name, lo, hi in (('training', 0, 87), ('early', 0, 52), ('recent', 53, 87)):
        part = eq.loc[meta['dates'][lo]:meta['ends'][hi]]
        answer[name] = engine.stats(part / part.iloc[0] * 100000.)
    return answer

def guard(candidate, reference, gain):
    return (candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12 and
        candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05 and
        candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03 and
        all(candidate[e]['cagr'] >= reference[e]['cagr'] - .025 for e in ('early', 'recent')))

def select_first88(metrics, orders):
    changed = [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[NAMES[0]][i]['symbols'])]
    guards = dict(close_vs_incumbent=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[0]]['same_close_zero'], .005),
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[0]]['next_open_cost'], 0.),
        close_vs_all_peer_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_all_peer_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
        changed_support=len(changed) >= 3 and any(i < 53 for i in changed) and any(i >= 53 for i in changed))
    changed_control = [i for i in range(TRAIN) if set(orders[NAMES[1]][i]['symbols']) != set(orders[NAMES[2]][i]['symbols'])]
    guards['changed_support_vs_control'] = len(changed_control)>=3 and any(i<53 for i in changed_control) and any(i>=53 for i in changed_control)
    return dict(primary=NAMES[1] if all(guards.values()) else NAMES[0], admitted=all(guards.values()),
        guards=guards, changed_months=changed, changed_months_vs_control=changed_control, first88_only=True, later34_used=False)

def select(expected):
    meta, orders = prepared(expected); require(not (OUT / 'selected').exists(), 'Selection is write-once')
    px, op = ev.prices(); engine = ev.engine(); metrics = {}
    for arm in NAMES:
        metrics[arm] = {}
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[2]):
            _, eq, _, _ = engine.replay(px, op, orders[arm][:TRAIN], cost=cost, execution=mode,
                start=pd.Timestamp(meta['dates'][0]), end=pd.Timestamp(meta['ends'][87]))
            require(str(eq.index[-1].date()) == meta['ends'][87], 'Selection escaped first88 cutoff')
            metrics[arm][path] = train_metrics(engine, eq, meta)
    result = select_first88(metrics, orders); gate(expected)
    dest = OUT / 'selected'; dest.mkdir(); dump(dest / 'TRAIN_RESULTS.json', metrics)
    dump(dest / 'FROZEN.json', dict(created_utc=now(), plan_sha256=expected,
        prepared_sha256=sha(OUT / 'prepared/COMPLETION.json'), orders_sha256=sha(OUT / 'prepared/ORDERS.json'),
        training_sha256=sha(dest / 'TRAIN_RESULTS.json'), selection=result))
    print(json.dumps(result), flush=True)

def frozen_selection(expected, orders):
    frozen = read(OUT / 'selected/FROZEN.json')
    require(frozen['plan_sha256'] == expected and frozen['orders_sha256'] == sha(OUT / 'prepared/ORDERS.json') and
        frozen['training_sha256'] == sha(OUT / 'selected/TRAIN_RESULTS.json') and
        frozen['prepared_sha256'] == sha(OUT / 'prepared/COMPLETION.json'), 'Frozen selection dependencies changed')
    require(select_first88(read(OUT / 'selected/TRAIN_RESULTS.json'), orders) == frozen['selection'], 'Selection mismatch')
    return frozen

def verify_training_against_audit(training, audit):
    import monthly_asset_peer213_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_cost'}, 'Training paths differ')
        for path, slices in training[arm].items():
            require(set(slices) == {'training', 'early', 'recent'}, 'Training slices differ')
            for name, full_name in (('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')):
                helper.check_metrics(audit['paths'][arm][path]['slices'][full_name], slices[name],
                                     'Frozen training/' + arm + '/' + path + '/' + name)

def diagnose(expected):
    meta, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(not (OUT / 'results').exists(), 'Diagnostics are write-once')
    px, op = ev.prices(); engine = ev.engine(); dest = OUT / 'results'; dest.mkdir(); results, exports = {}, {}
    for arm in NAMES:
        results[arm], exports[arm] = {}, {}
        for path, mode, cost in ev.PATHS:
            result, eq, trades, holds, positions = ev.run_path(engine, px, op, orders[arm], meta, mode, cost)
            result['slices'] = dict(full122=result['metrics'], **ev.split_stats(engine, eq, meta))
            results[arm][path] = result
            data = dict(DAILY=[dict(date=str(d.date()), equity=float(v)) for d, v in eq.items()],
                TRADES=trades, HOLDINGS=holds, DAILY_POSITIONS=positions)
            exports[arm][path] = data
            for kind, value in data.items(): dump(dest / (arm + '_' + path + '_' + kind + '.json'), value)
    import monthly_asset_peer213_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, asset_type_peer_transfer=True, same_support_both_arms=True, observer_promotion=False))
    gate(expected)
    dump(dest / 'COMPLETION.json', dict(status='COMPLETE', created_utc=now(), plan_sha256=expected,
        frozen_sha256=sha(OUT / 'selected/FROZEN.json'), paths=9, slices=45,
        artifacts_sha256={p.name: sha(p) for p in dest.iterdir()}))
    print(json.dumps(dict(status='COMPLETE', primary=frozen['selection']['primary'],
        full_cagr={n: results[n]['same_close_zero']['metrics']['cagr'] for n in NAMES},
        completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)

def require_result_manifest(manifest):
    require(isinstance(manifest,dict) and set(manifest)==RESULT_FILES,
            'Exact38 result artifacts required; completion receipt is excluded')

def state_check(state):
    require(len(state['completed'])==210 and sum(row['trials'] for row in state['completed'])==1147,
            'Expected210 completed entries /1147 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round212 must be the completed predecessor')


def model_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed model source review')
    review=read(SOURCE_REVIEW)
    require(review.get('passed') is True and review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    require(review.get('raw_source_hashes')==raw,'Original raw source lineage changed')
    require(review.get('evidence_sha256')==read(CONTEXT)['evidence_sha256'] and review.get('coverage_sha256')==COVERAGE_SHA,'Source declared evidence differs')
    for key in ('cached_forecasts_are_prior_annual_asof_replays','not_realtime_archival_vintages','numeric_type_support_not_yet_computed','unchanged_source_bytes_reverified'):
        require(review.get(key) is True,'Source metadata scope missing '+key)
    for key in ('actual_features_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed','actual_source_arrays_loaded','new_data_acquired'):
        require(review.get(key) is False,'Source review crossed research boundary: '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    require(isinstance(review.get('source_sha256'),dict) and review['source_sha256'],'Missing immutable model sources')
    for name,digest in review['source_sha256'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        require(path.resolve()!=CENTRAL.resolve(),'Mutable central cannot be a persistent source input')
        bind(pins,path,digest)
    require(review.get('immutable_source_files')==len(review['source_sha256']) and review.get('prior_saved_training_rows')==8820 and review.get('prior_saved_query_rows')==5978 and review.get('prior_saved_feature_count')==10 and review.get('prior_parity_receipt_reverified') is True and review.get('strict_pit') is False,'Original model source coverage metadata differs')
    cached=cached_source_binding(raw)
    for name,digest in cached['input_sha256'].items():bind(pins,Path(name),digest)
    verify(pins)
    return dict(cached_reference=cached,path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        original_cached_training_and_queries=True,new_labels=False,new_raw_features=False,strict_pit=False)


def cached_source_binding(raw):
    """Authenticate frozen206 cache bytes and identity proof, without opening NPZ."""
    pins={};context=read(CONTEXT)
    for name,digest in context['evidence_sha256'].items():bind(pins,ROOT/name,digest)
    source_plan=read(SOURCE/'PLAN.json');closed=read(SOURCE/'COMPLETION.json');prepared=read(SOURCE/'prepared/COMPLETION.json')
    require(source_plan['raw_source_hashes']==raw and len(raw)==64,'Original raw source lineage changed')
    for name,digest in source_plan['input_sha256'].items():bind(pins,Path(name),digest)
    require(closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prepared['status']=='COMPLETE'
            and closed['plan_sha256']==prepared['plan_sha256']==sha(SOURCE/'PLAN.json'),'206 source closure incomplete')
    require(set(prepared['artifacts_sha256'])==PREPARED_FILES,'206 exact prepared manifest required')
    for name,digest in prepared['artifacts_sha256'].items():bind(pins,SOURCE/'prepared'/name,digest)
    old=read(SOURCE/'prepared/INDEPENDENT_PREPARE.json');parity=read(SOURCE/'prepared/SOURCE_PARITY.json')
    for key in ('passed','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked','four_target_common_cohort_checked','original_source_and_local_parity','availability_unchanged','no_future_or_outcome_inputs','audit_checked'):
        require(old['model'].get(key) is True,'206 independent cache evidence missing '+key)
    require(old.get('model_prefixes')==[0,1,53,88,122] and old.get('model_prefix_sources_truncated') is True
            and old.get('original_arrays_unchanged') is True,'206 physical prefix evidence incomplete')
    require(parity.get('original_training_cache_exact') is True and parity.get('original_identity_exact') is True
            and parity.get('original_control_constant_exact') is True and parity.get('raw_source_hashes')==raw,'206 cache source parity differs')
    m=read(SOURCE/'prepared/META.json');metadata=read(REBUILD/'MONTHLY_META.json')
    require(m==metadata['monthly']==read(LOCAL/'META.json'),'Original query identity differs')
    require(len(m['dates'])==len(m['ends'])==len(m['symbols'])==122 and all(len(row)==len(set(row))==49 for row in m['symbols']),'Original122x49 identity required')
    require(isinstance(m['etfs'],list) and len(m['etfs'])==len(set(m['etfs']))==17
            and all(isinstance(x,str) and x and x.strip()==x for x in m['etfs']),'Original17 ETF identities required')
    verify(pins)
    return dict(directory=SOURCE.name,input_sha256=pins,source_plan_sha256=sha(SOURCE/'PLAN.json'),
        source_forecasts_sha256=sha(SOURCE/'prepared/FORECASTS.npz'),source_query_meta_sha256=sha(SOURCE/'prepared/META.json'),
        training_inputs_sha256=sha(REBUILD/'TRAIN_INPUTS.npz'),training_meta_sha256=sha(REBUILD/'MONTHLY_META.json'),
        etfs=m['etfs'],prior_annual_asof_forecasts=True,not_realtime_archival_vintages=True,
        cached_four_target_and_local_linear_reused=True,annual_pooled_log_refit_required=True,
        new_labels=False,new_raw_features=False,strict_pit=False)


def cached_prefix(source, rows, width, n):
    require(type(n) is int and 0<=n<=rows and set(source)==SOURCE_KEYS,'Cached206 source keys/prefix differ')
    require(all(np.asarray(value).shape==(rows*width,) for value in source.values()),'Cached206 source axes differ')
    return {key:np.asarray(value)[:n*width].copy() for key,value in source.items()}


def require_cached_vectors(source,arrays):
    rows,width=arrays['baseline'].shape;cached=cached_prefix(source,rows,width,rows)
    for key in ('source_log','local_linear','local_constant'):
        a=np.asarray(arrays['raw_'+key]).reshape(-1);b=cached['raw_'+key]
        require(a.dtype==b.dtype and a.shape==b.shape,'Original cached dtype/shape differs '+key)
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(cached['raw_control'],cached['raw_local_constant'])
    for src,dst in (('available','model_available'),('local_available','local_available')):
        a=np.asarray(arrays[dst]).reshape(-1);b=cached[src]
        require(a.dtype==b.dtype==np.dtype(bool),'Original cached availability dtype differs')
        np.testing.assert_array_equal(a,b)
    return True


def require_control_parity(forecast,cached,ranks,deltas,orders,old_ranks,old_orders):
    for key in ('source_log','local_linear','local_constant'):
        np.testing.assert_array_equal(forecast['raw'][key],cached['raw_'+key])
    for key in ('available','local_available'):np.testing.assert_array_equal(forecast[key],cached[key])
    np.testing.assert_array_equal(forecast['raw']['control'],cached['raw_candidate'])
    old_arm='peer_residual_transfer';new_arm=NAMES[2]
    for kind,values in (('rank',ranks),('delta',deltas)):
        np.testing.assert_array_equal(values[new_arm],old_ranks[kind+'_'+old_arm])
    require(orders[new_arm]==old_orders[old_arm],'Control orders differ from frozen206 all-peer')
    return dict(cached_source_local_vectors_exact=True,availability_exact=True,
        control_raw_exact=True,control_rank_exact=True,control_delta_exact=True,control_orders_exact=True,
        source_arm=old_arm,new_control_arm=new_arm)


def tests():
    names=('test_monthly_asset_peer213_host.py','test_monthly_asset_peer213_model.py',
           'test_monthly_asset_peer213_independent.py')
    modules=[module('test_asset_peer213_'+str(i),ROOT/'work'/name) for i,name in enumerate(names)]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic validation failed')
    require(not TEST_RECEIPT.exists(),'Test receipt is write-once')
    dump(TEST_RECEIPT,dict(passed=True,synthetic_only=True,tests=result.testsRun,runtime=ev.runtime(),
        code_sha256={str(p):sha(p) for p in code_files()}))


def register():
    require(not OUT.exists(),'Fresh experiment directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    require(not state['registered_queue'],'Existing registered experiment')
    previous=predecessor_binding(state);validation=read(TEST_RECEIPT)
    require(validation['passed'] is True and validation['synthetic_only'] is True and validation['runtime']==ev.runtime(),'Synthetic tests required')
    require(validation['code_sha256']=={str(p):sha(p) for p in code_files()},'Code changed since tests')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);raw=read(LOCAL/'PLAN.json')['raw_source_hashes']
    source=model_source_binding(raw);review=code_review_binding()
    for binding in (previous,source,review):
        for name,digest in binding['input_sha256'].items():bind(pins,Path(name),digest)
    for p in code_files()+[TEST_RECEIPT,PROTECTED,CALENDAR,LOCAL/'PLAN.json']:
        bind(pins,p)
    for name in ('DATA.npz','META.json','INPUT_HASHES.json','CACHE_HASHES.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'):
        bind(pins,LOCAL/name)
    for name in ('TRAIN_INPUTS.npz','MONTHLY_META.json','PARITY.json'):bind(pins,REBUILD/name)
    bind(pins,REBUILD.parent/'PLAN.json')
    for name,digest in raw.items():bind(pins,Path(name),digest)
    for name,digest in ev.ENGINE_HASHES.items():bind(pins,ev.ENGINE_DIR/name,digest)
    verify(pins);protected=read(PROTECTED);verify(protected)
    require(len(protected)==21444,'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as stream:stream.write(before)
    plan=dict(schema='monthly_asset_peer213_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        model_source_binding=source,numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_peer_corrections_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_new_asset_peer_fits',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare same-asset versus all-peer transfer and independent audit, then freeze first88 selection.')
    commit_state(before,state,'register')
    print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def slice_arrays(arrays,n):
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid query prefix')
    out={key:value[:n].copy() if np.asarray(value).ndim and len(value)==shape[0] else value.copy()
         for key,value in arrays.items()}
    for key in ('eligible','forecast_available','model_available','extension_available','state_weak_narrow','state_calm10'):
        if key in out:
            require(all(isinstance(v,(bool,np.bool_)) for v in np.asarray(out[key]).flat),'Exact boolean values required: '+key)
            out[key]=np.asarray(out[key],dtype=bool)
    return out


def slice_forecast(forecast,shape,n):
    require(type(n) is int and 0<=n<=shape[0],'Invalid forecast prefix')
    q=shape[0]*shape[1]
    require(set(forecast['raw'])>=set(RAW.values()),'Missing paired forecasts')
    raw={key:np.asarray(value) for key,value in forecast['raw'].items()}
    require(all(value.shape==(q,) for value in raw.values()),'Forecast vector shape changed')
    require(np.asarray(forecast['available']).shape==(q,),'Availability shape changed')
    available=np.asarray(forecast['available'])[:n*shape[1]].copy()
    require(all(isinstance(v,(bool,np.bool_)) for v in available),'Exact boolean forecast availability required')
    return dict(raw={key:value[:n*shape[1]].copy() for key,value in raw.items()},
                available=available.astype(bool))


def rank_and_delta(study,arrays,forecast,n):
    shape=np.asarray(arrays['baseline']).shape
    a=slice_arrays(arrays,n);f=slice_forecast(forecast,shape,n);smallshape=(n,shape[1])
    oldranks,_=study.ranked_forecasts({key:a['raw_'+key] for key in study.ARMS},a['model_available'],a,n)
    old=oldranks['local_linear'];_,use=study.original_rank_inputs(a,n)
    paired={'local_linear':f['raw']['candidate'].reshape(smallshape).astype(float),
            'local_constant':f['raw']['control'].reshape(smallshape).astype(float)}
    rr,ready=study.ranked_forecasts(paired,f['available'].reshape(smallshape),a,n)
    # Frozen adapter validates exact booleans on the consumed prefix only.
    weak=study.adapter._state(a,'state_weak_narrow',n,n)
    calm=study.adapter._state(a,'state_calm10',n,n)
    extension=study.adapter._matrix(a['extension_available'],smallshape,n,'extension availability',boolean=True)
    coefficient=.25*(.05*weak[:,None]+.01*(calm&~weak)[:,None]*extension)
    ranks,deltas={},{}
    for name,key in ((NAMES[1],'local_linear'),(NAMES[2],'local_constant')):
        rank=rr[key].copy();rank[~ready]=old[~ready]
        delta=np.zeros(smallshape);active=use&ready[:,None]
        delta[active]=(coefficient*(rank-old))[active]
        if n:delta[0]=0.
        ranks[name],deltas[name]=rank,delta
    return ranks,deltas


def decisions(study,arrays,meta,forecast,n):
    a=slice_arrays(arrays,n);shape=arrays['baseline'].shape
    f=slice_forecast(forecast,shape,n)
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choices,scores=study.choices(study.configs()[4],a,meta,n,policy=policy)
    orders={NAMES[0]:study.rp.make_orders(choices,scores,meta)}
    ranks,deltas=rank_and_delta(study,a,f,n)
    for name in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],a,n)
        changed=deltas[name]!=0.;pre[changed]+=deltas[name][changed]
        choices,scores=study.adapter.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,meta,n,
            pre_score=pre,effects=a['effects'],episode_policy=policy,boundary_selector=study.ep.select_boundary)
        orders[name]=study.rp.make_orders(choices,scores,meta)
    return orders,ranks,deltas


def model_inputs(train,meta,arrays,limit=None):
    """Physically discard future labels/queries before model code sees a prefix."""
    m,tm=meta['monthly'],meta['training'];rows,width=arrays['baseline'].shape
    n=rows if limit is None else limit
    require(type(n) is int and 0<=n<=rows and len(m['dates'])==rows,'Invalid model prefix')
    symbols=np.array([s for pool in tm['symbols'] for s in pool])
    event_symbols=np.array([s for pool in m['symbols'][:n] for s in pool])
    require(len(event_symbols)==n*width,'Query identity widths differ')
    dates=np.asarray(train['dates']);ends=np.asarray(train['ends'])
    require(symbols.shape==dates.shape==ends.shape,'Training identity lengths differ')
    keep=np.ones(len(dates),bool)
    if limit is not None:
        if n:
            year=int(str(m['dates'][n-1])[:4]);cutoff=np.datetime64('%04d-12-31'%(year-1),'D')
            keep=np.asarray(ends,dtype='datetime64[D]')<=cutoff
        else:keep[:]=False
    q=n*width;features=np.asarray(train['event_plain'])[:q].astype(float,copy=True)
    good=np.asarray(arrays['eligible'])[:n].reshape(-1)&np.isfinite(features).all(axis=1)
    return (dates[keep].copy(),ends[keep].copy(),symbols[keep],np.asarray(train['train_plain'])[keep].copy(),
        {key:np.asarray(train['target_'+key])[keep].copy() for key in TARGETS},
        np.asarray(train['common_good'])[keep].copy(),np.asarray(train['event_dates'])[:q].copy(),
        event_symbols,features,good)


def require_model_audit(check,forecast,expected_rows=5978,expected_years=11):
    for key in MODEL_FLAGS:require(check.get(key) is True,'Independent model evidence missing '+key)
    require(check.get('atol')==1e-12 and check.get('computes_returns') is False,'Independent model scope/tolerance differs')
    require(check.get('residuals_are_mature_in_sample') is True and check.get('residuals_are_oof') is False,'Residual validation claim differs')
    require(set(forecast['raw'])==set(RAW_KEYS),'Raw forecast keys differ')
    available=np.asarray(forecast['available']);own=np.asarray(forecast['local_available'])
    require(available.shape==own.shape==(expected_rows,) and available.dtype==own.dtype==np.dtype(bool),'Query availability axes differ')
    require(not np.any(own&~available),'Local availability exceeds pooled availability')
    for key,value in forecast['raw'].items():
        value=np.asarray(value)
        require(value.shape==(expected_rows,) and not np.isinf(value).any() and np.array_equal(np.isfinite(value),available),'Raw support differs '+key)
    audit=forecast['audit'];counts=audit['counts']
    require(check.get('annual_cohorts')==len(audit['annual'])==expected_years,'Annual audit coverage')
    require(check.get('local_queries')==len(audit['local'])==int(available.sum()),'Local audit coverage')
    require(set(counts)==set(MODEL_COUNT_FIELDS) and all(type(v) is int and v>=0 for v in counts.values()),'Integer model accounting required')
    require(counts['actual_regression_solves']==counts['annual_pooled_log_refits']==sum(row['prediction_rows']>0 for row in audit['annual']),'Annual regression solve accounting')
    require(counts['cached_forecast_rows']==int(available.sum()),'Cached forecast coverage')
    supported=[row for row in audit['local'] if row['status']=='same_type_supported']
    fallback=[row for row in audit['local'] if row['status']=='same_type_missing_month']
    require(counts['same_type_supported_queries']==len(supported) and counts['same_type_fallback_queries']==len(fallback),'Type support query counts')
    require(counts['own_weight_evaluations']==counts['all_peer_query_evaluations']==len(supported)+len(fallback),'Own/all peer query counts')
    require(counts['all_peer_month_evaluations']==sum(len(row['peer_months']) for row in audit['local']) and
            counts['all_peer_row_observations']==sum(month['peer_count'] for row in audit['local'] for month in row['peer_months']),'All-peer count arithmetic')
    require(counts['same_type_month_evaluations']==sum(len(row['peer_months']) for row in supported) and
            counts['same_type_row_observations']==sum(month['same_peer_count'] for row in supported for month in row['peer_months']),'Supported same-type count arithmetic')
    require(isinstance(check.get('counts'),dict) and set(check['counts'])==set(counts) and all(type(v) is int for v in check['counts'].values()) and check['counts']==counts,'Independent accounting differs')
    for record in (audit,check):
        require(record.get('cached_reference_used') is True and record.get('correction_accounting')==CORRECTION_ACCOUNTING,'Cached-reference accounting differs')
        for key in ('source_four_target_refits','local_linear_refits'):
            require(type(record.get(key)) is int and record[key]==0,'Unexpected repeated source/local fit '+key)
    require(type(check.get('independent_pooled_fits')) is int and check['independent_pooled_fits']==counts['annual_pooled_log_refits'],'Independent pooled-fit accounting differs')

def require_slot_audit(check,deltas):
    require(set(deltas)==set(RAW),'Delta arm keys differ')
    for value in deltas.values():
        require(np.asarray(value).shape==(122,49) and np.isfinite(value).all() and np.max(np.abs(value))<=.0125+1e-12 and np.array_equal(value[0],np.zeros(49)),'Delta axes/bounds/initial allocation differ')
    for key in SLOT_FLAGS:require(check.get(key) is True,'Independent slot evidence missing '+key)
    require(set(check.get('ready_rows',{}))==set(RAW),'Missing readiness coverage')
    require(all(type(v) is int and 0<=v<=122 for v in check['ready_rows'].values()) and
            len(set(check['ready_rows'].values()))==1,'Candidate readiness differs')
    require(check.get('changed_cells')=={key:int(np.count_nonzero(value)) for key,value in deltas.items()},'Changed-cell audit differs')


def transfer_diagnostics(meta,forecast,ranks,orders,arrays):
    """Declared descriptive counts only; no refit, outcome, or selection input."""
    etfs=set(meta['etfs']);symbols=np.asarray(meta['symbols']);shape=symbols.shape
    require(shape==np.asarray(arrays['baseline']).shape,'Diagnostic identity shape')
    available=np.asarray(forecast['available']).reshape(shape)
    use=np.asarray(arrays['forecast_available'])&np.asarray(arrays['eligible'])
    raw_difference=(forecast['raw']['candidate']!=forecast['raw']['control']).reshape(shape)&available
    rank_difference=(ranks[NAMES[1]]!=ranks[NAMES[2]])&use
    summary={typ:dict(available_queries=0,same_type_supported_queries=0,same_type_fallback_queries=0,
        insufficient_own_history_queries=0,raw_candidate_control_differences=0,rank_candidate_control_differences_in_original_U=0)
        for typ in ('stock','etf')}
    status_key={'same_type_supported':'same_type_supported_queries','same_type_missing_month':'same_type_fallback_queries','insufficient_own_history':'insufficient_own_history_queries'}
    for row in forecast['audit']['local']:
        i,j=divmod(row['event_row'],shape[1]);typ='etf' if symbols[i,j] in etfs else 'stock'
        require(row['symbol']==symbols[i,j] and row['asset_type']==typ and bool(available[i,j]),'Diagnostic local identity')
        require(row['status'] in status_key,'Unknown diagnostic query status')
        summary[typ]['available_queries']+=1;summary[typ][status_key[row['status']]]+=1
    for typ in summary:
        mask=np.array([[('etf' if x in etfs else 'stock')==typ for x in row] for row in symbols],dtype=bool)
        summary[typ]['raw_candidate_control_differences']=int((raw_difference&mask).sum())
        summary[typ]['rank_candidate_control_differences_in_original_U']=int((rank_difference&mask).sum())
    holdings={}
    for label,left,right in (('candidate_vs_incumbent',NAMES[1],NAMES[0]),('control_vs_incumbent',NAMES[2],NAMES[0]),('candidate_vs_control',NAMES[1],NAMES[2])):
        require(len(orders[left])==len(orders[right])==shape[0],'Diagnostic orders axis')
        changed=[i for i,(a,b) in enumerate(zip(orders[left],orders[right])) if set(a['symbols'])!=set(b['symbols'])]
        by_type={typ:[i for i,(a,b) in enumerate(zip(orders[left],orders[right]))
            if {x for x in a['symbols'] if (x in etfs)==(typ=='etf')}!={x for x in b['symbols'] if (x in etfs)==(typ=='etf')}] for typ in summary}
        holdings[label]=dict(changed_months=changed,first88=sum(i<88 for i in changed),early53=sum(i<53 for i in changed),
            recent35=sum(53<=i<88 for i in changed),later34=sum(i>=88 for i in changed),changed_months_by_asset_type=by_type)
    return dict(descriptive_only=True,used_for_selection=False,by_asset_type=summary,holding_changes=holdings,
        raw_domain='cached available queries',rank_domain='original forecast_available AND eligible; both types ranked together')


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_asset_peer213_model as model
    import monthly_asset_peer213_independent as independent
    study=local();m,a,px,op=study.load()
    require(len(m['dates'])==122 and a['baseline'].shape==(122,49) and len(a)==27,'Original122x49/27 arrays required')
    before={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()}
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');meta=read(REBUILD/'MONTHLY_META.json')
    require(meta['monthly']==m and meta['training']==read(study.arch.P/'META.json')['training'],'Original row identity changed')
    original=ev.load_arrays(study.target.P/'TRAIN_INPUTS.npz')
    require(set(train)==set(original),'Original training cache keys differ')
    for key,value in train.items():
        require(value.dtype==original[key].dtype and value.shape==original[key].shape,'Original cache axes/type differ: '+key)
        np.testing.assert_array_equal(value,original[key],err_msg=key)
    widths=np.array([len(s) for s in meta['training']['symbols']])
    np.testing.assert_array_equal(train['offsets'],np.r_[0,np.cumsum(widths)])
    for key in ('dates','ends'):
        np.testing.assert_array_equal(train[key].astype('datetime64[D]'),np.repeat(np.array(meta['training'][key],dtype='datetime64[D]'),widths))
    cached=ev.load_arrays(SOURCE/'prepared/FORECASTS.npz')
    require(read(SOURCE/'prepared/META.json')==m,'Cached206 query identity differs')
    require_cached_vectors(cached,a)
    forecast=model.annual_forecasts(*model_inputs(train,meta,a),etfs=m['etfs'],cached_reference=cached)
    require(set(forecast['raw'])==set(RAW_KEYS),'Unexpected model raw schema')
    np.testing.assert_array_equal(forecast['raw']['control'],cached['raw_candidate'])
    np.testing.assert_array_equal(forecast['available'],a['model_available'].reshape(-1))
    np.testing.assert_array_equal(forecast['local_available'],a['local_available'].reshape(-1))
    check=independent.check_prepare(train,meta,a,forecast,cached);require_model_audit(check,forecast)
    orders,ranks,deltas=decisions(study,a,m,forecast,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    slot=independent.verify_rank_slot(a,forecast,ranks,deltas);require_slot_audit(slot,deltas)
    control_parity=require_control_parity(forecast,cached,ranks,deltas,orders,
        ev.load_arrays(SOURCE/'prepared/RANKS_DELTAS.npz'),read(SOURCE/'prepared/ORDERS.json'))
    prefix_evidence=[]
    for n in MODEL_PREFIXES:
        inputs=model_inputs(train,meta,a,limit=n)
        smallcache=cached_prefix(cached,122,49,n)
        small=model.annual_forecasts(*inputs,etfs=m['etfs'],cached_reference=smallcache);q=n*49
        for key in RAW_KEYS:np.testing.assert_array_equal(small['raw'][key],forecast['raw'][key][:q])
        for key in ('available','local_available'):np.testing.assert_array_equal(small[key],forecast[key][:q])
        prefix_evidence.append(dict(prefix=n,training_rows=len(inputs[0]),query_rows=q,
            query_date=m['dates'][n-1] if n else None,
            cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(inputs[1],dtype='datetime64[D]').max()) if len(inputs[1]) else None,
            counts=small['audit']['counts']))
    for n in range(123):
        aa=slice_arrays(a,n);mm=dict(m)
        for key in ('dates','ends','symbols'):mm[key]=m[key][:n]
        require(len(aa)==27 and all(value.shape[0]==n for value in aa.values()),'All27 physical cache axes required')
        oo,rr,dd=decisions(study,aa,mm,slice_forecast(forecast,a['baseline'].shape,n),n)
        require(all(oo[key]==orders[key][:n] for key in NAMES),'Prefix order parity failed')
        for key in RAW:
            np.testing.assert_array_equal(rr[key],ranks[key][:n]);np.testing.assert_array_equal(dd[key],deltas[key][:n])
    require(before=={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()},'Original arrays modified')
    diagnostics=transfer_diagnostics(m,forecast,ranks,orders,a)
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{'raw_'+key:value for key,value in forecast['raw'].items()},
        available=forecast['available'],local_available=forecast['local_available'])
    np.savez_compressed(dest/'RANKS_DELTAS.npz',**{'rank_'+key:v for key,v in ranks.items()},**{'delta_'+key:v for key,v in deltas.items()})
    dump(dest/'MODEL_AUDIT.json',forecast['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',m)
    dump(dest/'SOURCE_PARITY.json',dict(original_training_cache_exact=True,original_identity_exact=True,
        raw_source_hashes=plan['raw_source_hashes'],model_source_binding=plan['model_source_binding'],
        new_labels=False,new_raw_features=False,cached206_source_vectors_exact=True,
        fixed_etfs=m['etfs'],cached_four_target_and_local_linear_reused=True,
        annual_pooled_log_refit_required=True,control206_parity=control_parity,transfer_diagnostics=diagnostics))
    evidence=dict(model=check,slot=slot,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        model_prefix_sources_truncated=True,model_prefix_cached_forecasts_truncated=True,
        model_prefix_future_labels_removed=True,order_prefix_original_arrays_and_metadata_truncated=True,
        control206_parity=control_parity,transfer_diagnostics=diagnostics,model_prefix_evidence=prefix_evidence,
        original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,
        original_core_metadata_and_support_preserved=True,single_original_episode=True,
        no_new_score_budget=True,model_counts=forecast['audit']['counts'])
    dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,
        original_array_count=27,model_counts=forecast['audit']['counts'],
        model_fit_accounting=contract()['model_fit_accounting']))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def verify_prepared_evidence(c,forecast,deltas,audit,meta):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require_model_audit(audit['model'],forecast);require_slot_audit(audit['slot'],deltas)
    require(audit.get('prefixes')==123 and audit.get('model_prefixes')==list(MODEL_PREFIXES),'Missing prefix coverage')
    for key in ('model_prefix_sources_truncated','model_prefix_cached_forecasts_truncated','model_prefix_future_labels_removed','order_prefix_original_arrays_and_metadata_truncated','original_orders_exact','original_arrays_unchanged',
                'original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget'):
        require(audit.get(key) is True,'Prepared invariant missing: '+key)
    require(audit.get('original_array_count')==c.get('original_array_count')==27,'Original array count differs')
    require(c.get('new_returns_computed') is False,'Preparation computed portfolio returns')
    diagnostics=audit.get('transfer_diagnostics',{})
    require(diagnostics.get('descriptive_only') is True and diagnostics.get('used_for_selection') is False
            and set(diagnostics.get('by_asset_type',{}))=={'stock','etf'},'Missing descriptive type evidence')
    parity=audit.get('control206_parity',{})
    for key in ('cached_source_local_vectors_exact','availability_exact','control_raw_exact','control_rank_exact','control_delta_exact','control_orders_exact'):
        require(parity.get(key) is True,'Frozen206 control parity missing '+key)
    require(parity.get('source_arm')=='peer_residual_transfer' and parity.get('new_control_arm')==NAMES[2],'Control parity arm identity')
    counts=forecast['audit']['counts']
    require(audit.get('model_counts')==c.get('model_counts')==counts,'Main-pass model accounting differs')
    evidence=audit.get('model_prefix_evidence')
    require(isinstance(evidence,list) and [row.get('prefix') for row in evidence]==list(MODEL_PREFIXES),'Model prefix evidence missing')
    require(all(row.get('query_rows')==row['prefix']*49 and type(row.get('training_rows')) is int and row['training_rows']>=0 for row in evidence),'Model prefix row coverage differs')
    require(evidence[0]['training_rows']==0 and evidence[0].get('latest_label_end') is None and evidence[0].get('query_date') is None and evidence[0].get('cutoff') is None,'Empty prefix contains source rows')
    for row in evidence[1:]:
        day=meta['dates'][row['prefix']-1];cutoff='%04d-12-31'%(int(day[:4])-1)
        require(row.get('query_date')==day and row.get('cutoff')==cutoff and
                (row['latest_label_end'] is None or row['latest_label_end']<=cutoff),'Physical prefix maturity clock differs')


def prepared(expected):
    gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared artifact '+name)
    values=ev.load_arrays(dest/'FORECASTS.npz')
    require(set(values)=={'raw_'+key for key in RAW_KEYS}|{'available','local_available'},'Stored forecast schema differs')
    forecast=dict(raw={key:values['raw_'+key] for key in RAW_KEYS},available=values['available'],
        local_available=values['local_available'],audit=read(dest/'MODEL_AUDIT.json'))
    ranks=ev.load_arrays(dest/'RANKS_DELTAS.npz')
    require(set(ranks)=={'rank_'+key for key in RAW}|{'delta_'+key for key in RAW},'Stored rank/delta schema differs')
    deltas={key:ranks['delta_'+key] for key in RAW}
    verify_prepared_evidence(c,forecast,deltas,read(dest/'INDEPENDENT_PREPARE.json'),read(dest/'META.json'))
    return read(dest/'META.json'),read(dest/'ORDERS.json')

def close(expected):
    gate(expected); completion = read(OUT / 'results/COMPLETION.json')
    require(completion['status'] == 'COMPLETE' and completion['plan_sha256'] == expected and
            completion['paths'] == 9 and completion['slices'] == 45, 'Incomplete diagnostics')
    require_result_manifest(completion['artifacts_sha256'])
    for name, digest in completion['artifacts_sha256'].items(): require(sha(OUT / 'results' / name) == digest, 'Changed result ' + name)
    require_outcome_audit(read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), read(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    _, orders = prepared(expected); frozen = frozen_selection(expected, orders)
    require(completion['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'), 'Frozen selection receipt changed')
    results = read(OUT / 'results/RESULTS.json')
    require(results['selection'] == frozen['selection'], 'Closure selection mismatch')
    before = CENTRAL.read_bytes(); state = json.loads(before); state_check(state)
    protected = read(PROTECTED); verify(protected); require(len(protected) == 21444, 'Incomplete protection')
    require(not (OUT / 'COMPLETION.json').exists(), 'Closure is write-once')
    best = state['best_training_selected_candidate']; pick = results['selection']['primary']; arm = results['arms'][pick]
    metrics = arm['same_close_zero']['metrics']
    if results['selection']['admitted'] and metrics['cagr'] > best['cagr']:
        best = dict(directory=OUT.name, spec=dict(arm=pick), cagr=metrics['cagr'], sharpe=metrics['sharpe'],
            max_drawdown=metrics['max_drawdown'], next_open_5bp_cagr=arm['next_open_cost']['metrics']['cagr'],
            selected_on_first88=True, observer_promotion=False, limitations=contract()['limitation'])
    receipt = dict(status='COMPLETED_INDEPENDENTLY_VERIFIED', created_utc=now(), plan_sha256=expected,
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, model_counts=read(OUT/'prepared/COMPLETION.json')['model_counts'],
        model_fit_accounting=contract()['model_fit_accounting'],
        original_array_count=27, protected_files=len(protected), best_training_selected_candidate=best,
        target_reached=best['cagr'] >= .7, observers_modified=False)
    dump(OUT / 'COMPLETION.json', receipt)
    state['completed'].append(dict(directory=OUT.name, trials=3, completed_utc=now(), primary=pick,
        primary_cagr=metrics['cagr'], primary_sharpe=metrics['sharpe'], new_admitted=int(results['selection']['admitted']),
        independent_execution_paths=9, protected_files=len(protected), target_reached=receipt['target_reached'],
        completion=dict(path=str(OUT / 'COMPLETION.json'), sha256=sha(OUT / 'COMPLETION.json'))))
    state.update(best_training_selected_candidate=best, latest_training_selected_candidate=dict(directory=OUT.name,
        arm=pick, selection=results['selection'], full_period_cagr=metrics['cagr'], full_period_sharpe=metrics['sharpe'], selected_on_first88=True),
        best_tracking_note=contract()['historical_best_tracking'], registered_queue=[], current_status='completed_independently_verified',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Review same-asset peer transfer and source limitations; preserve frozen observers.')
    require(len(state['completed']) == 211 and sum(r['trials'] for r in state['completed']) == 1150, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
