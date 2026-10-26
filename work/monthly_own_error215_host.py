"""Round215: causal annual own-security error quantiles of cached Local67 forecasts.

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
OUT = ROOT/'outputs/monthly_own_error215'
LOCAL = ROOT/'outputs/monthly_local_linear70'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
PRIOR = ROOT/'outputs/monthly_level_growth214'
SOURCE = ROOT/'outputs/monthly_identity_transfer206'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67','own_error_tail','own_error_median')
RAW = {NAMES[1]:'candidate', NAMES[2]:'control'}
RAW_KEYS = ('candidate','control')
SOURCE_KEYS = {'raw_source_log','raw_local_linear','raw_local_constant','raw_candidate','raw_control','available','local_available'}
TRAIN_KEYS = ('dates','ends','train_plain','common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility')
TARGETS = ('simple_excess','log_excess','vol_scaled','loss_utility')
TRAIN = 88
MODEL_PREFIXES = tuple(range(123))
MODEL_COUNT_FIELDS = ('annual_cohorts','own_history_groups','supported_year_security_groups','cold_start_groups','quantile_estimates','joined_label_rows','invalid_label_rows','skipped_initial_rows','error_observations','cached_forecast_rows')
CONTEXT = ROOT/'work/monthly_own_error215_context.json'
CONTEXT_SHA = '593d72172c007c28427d6cae5721ff0f869e7f66a0f348a1d10af9232be5681a'
COVERAGE = ROOT/'work/after214_own_error_coverage.json'
COVERAGE_SHA = 'caf4727cd05dd947f5326fc6a6ca586fb63f1b56dab8dc4c859a29f30a29a83b'
SOURCE_REVIEW = ROOT/'work/own_error215_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = 'b146c39268a280b9bdef2cb639c8540b0c8b204967c507e48a81ed27358d4dbb'
TEST_RECEIPT = ROOT/'work/monthly_own_error215_TESTS.json'
EXTERNAL_RUNTIME = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED = {
 'PLAN.json':'4267a65d5fa80934d5ade02059acd46afafa42cd7f06c75159c093ca125be8fa',
 'COMPLETION.json':'10e3451195e2ae4b5448cbacbdc6a9fa0f5d21e43e9c0cd9c1947f929d8833f3',
 'results/COMPLETION.json':'b949a2b5fc157979e44a1108b45a569b69fc1a340f0a0d802563c38eb7cc7a5d',
 'RESEARCH_REVIEW.json':'734b4857be609ca9b174b2d9c5d272db1f07d2f557c609d21257ac09ebc48157'}

MODEL_FLAGS = ('passed','independent_scalar_rebuild','exact_identity_join_checked','cached_forecasts_reused','local_linear_source_checked','annual_prior_december_cutoffs_checked','five_year_mature_window_checked','four_target_common_cohort_checked','historical_original_universe_checked','same_security_errors_checked','nearest_rank_q75_checked','conventional_median_checked','nonnegative_penalties_checked','cold_start_copy_checked','no_local_history_filter','availability_unchanged','original_targets_unchanged','no_future_or_outcome_inputs','source_forecasts_not_refitted','audit_checked','parameter_estimates_counted')
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
    return module('monthly_own_error215_original', LOCAL / 'study.py')

def contract():
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective context')
    require(sha(COVERAGE)==COVERAGE_SHA,'Changed coverage evidence')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,arms=list(NAMES),
        candidate=NAMES[1],control_nonpromotable=NAMES[2],period=context['evaluation']['period'],
        training_months=88,later_months=34,model_prefixes=list(MODEL_PREFIXES),all123_order_prefixes=True,all123_physical_model_prefixes=True,max_absolute_delta=.0125,
        model_fit_accounting=context['formation']['fit_accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,
        original_pool_preserved=True,no_parameter_search=True,no_new_score_budget=True)


def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_own_error215_host.py',
        'test_monthly_own_error215_host.py','monthly_own_error215_model.py',
        'test_monthly_own_error215_model.py','monthly_own_error215_independent.py',
        'test_monthly_own_error215_independent.py','monthly_own_error215_context.json','after214_own_error_coverage.json',
        'own_error215_SOURCE_REVIEW.json',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_own_error215_model as model
    import monthly_own_error215_independent as independent
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
    backup = CENTRAL.parent / ('PRE_MONTHLY_OWN_ERROR215_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/own_error215_CODE_REVIEW.json'
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
        close_vs_median_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_median_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_own_error215_independent as independent
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
    import monthly_own_error215_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, annual_own_error_reserve=True,cached_source_models_refitted=False, same_support_both_arms=True, observer_promotion=False))
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
    require(len(state['completed'])==212 and sum(row['trials'] for row in state['completed'])==1153,
            'Expected212 completed entries /1153 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round214 must be the completed predecessor')


def model_source_binding(raw):
    """Metadata and byte provenance only; no forecast/label arrays are decoded."""
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed cached-source review')
    review=read(SOURCE_REVIEW); context=read(CONTEXT)
    require(review.get('passed') is True and review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    require(review.get('evidence_sha256')==context['evidence_sha256'],'Source evidence differs from context')
    for key in ('cached_forecasts_are_prior_annual_asof_replays','not_realtime_archival_vintages','numeric_calibration_coverage_not_yet_computed','numeric_own_security_support_not_yet_computed','source_bytes_reverified'):
        require(review.get(key) is True,'Source scope missing '+key)
    for key in ('uses_fitted_residuals_as_historical_errors','source_npz_decoded','observers_modified'):
        require(review.get(key) is False,'Source review scope differs '+key)
    for key in ('actual_new_losses','actual_new_coefficients','actual_new_parameters','actual_new_quantiles','actual_new_orders','actual_new_returns'):
        require(type(review.get(key)) is int and review[key]==0,'Source review performed new research '+key)
    require(review.get('current_and_historical_prediction_key')=='raw_local_linear' and
            review.get('own_history_mask')=='originalU AND206available','Own error source identity/domain differs')
    require(review.get('matched_formal_months')==113 and len(review.get('source_annual',[]))==11,'Source history metadata incomplete')
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in context['evidence_sha256'].items():bind(pins,ROOT/name,digest)
    source_plan=read(SOURCE/'PLAN.json')
    require(source_plan['raw_source_hashes']==raw and len(raw)==64,'Original raw source lineage changed')
    for name,digest in source_plan['input_sha256'].items():bind(pins,Path(name),digest)
    closed=read(SOURCE/'COMPLETION.json'); prepared=read(SOURCE/'prepared/COMPLETION.json')
    require(closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prepared['status']=='COMPLETE'
            and closed['plan_sha256']==prepared['plan_sha256']==sha(SOURCE/'PLAN.json'),'206 source closure incomplete')
    require(set(prepared['artifacts_sha256'])==PREPARED_FILES,'206 exact prepared manifest required')
    for name,digest in prepared['artifacts_sha256'].items():bind(pins,SOURCE/'prepared'/name,digest)
    old=read(SOURCE/'prepared/INDEPENDENT_PREPARE.json'); parity=read(SOURCE/'prepared/SOURCE_PARITY.json')
    for key in ('passed','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked','four_target_common_cohort_checked','original_source_and_local_parity','availability_unchanged','no_future_or_outcome_inputs','audit_checked'):
        require(old['model'].get(key) is True,'206 independent source evidence '+key)
    require(old.get('model_prefixes')==[0,1,53,88,122] and old.get('model_prefix_sources_truncated') is True
            and old.get('original_arrays_unchanged') is True,'206 physical source evidence incomplete')
    require(parity.get('original_training_cache_exact') is True and parity.get('original_identity_exact') is True
            and parity.get('original_control_constant_exact') is True and parity.get('raw_source_hashes')==raw,'206 cache parity differs')
    saved_parity=read(REBUILD/'PARITY.json')
    require(saved_parity.get('passed') is True,'Saved169 training-cache parity incomplete')
    for key in TRAIN_KEYS+('offsets',):
        record=saved_parity['checks']['monthly'][key]
        require(record.get('exact') is True or record.get('max_abs_error')==0.,'Saved169 training-cache parity missing '+key)
    monthly=read(SOURCE/'prepared/META.json'); metadata=read(REBUILD/'MONTHLY_META.json')
    require(monthly==metadata['monthly']==read(LOCAL/'META.json'),'Original query identity differs')
    require(len(monthly['dates'])==len(monthly['ends'])==len(monthly['symbols'])==122 and
            all(len(row)==49 and len(set(row))==49 for row in monthly['symbols']),'Full query identity coverage')
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        directory=SOURCE.name,source_plan_sha256=sha(SOURCE/'PLAN.json'),
        source_forecasts_sha256=sha(SOURCE/'prepared/FORECASTS.npz'),
        source_query_meta_sha256=sha(SOURCE/'prepared/META.json'),
        training_inputs_sha256=sha(REBUILD/'TRAIN_INPUTS.npz'),training_meta_sha256=sha(REBUILD/'MONTHLY_META.json'),
        saved_training_parity_sha256=sha(REBUILD/'PARITY.json'),fresh_training_array_comparison=False,
        original_cached_training_and_queries=True,prior_annual_asof_forecasts=True,
        source_model_refits=0,uses_fitted_residuals_as_historical_errors=False,
        new_labels=False,new_raw_features=False,strict_pit=False)

def tests():
    names=('test_monthly_own_error215_host.py','test_monthly_own_error215_model.py',
           'test_monthly_own_error215_independent.py')
    modules=[module('test_own_error215_'+str(i),ROOT/'work'/name) for i,name in enumerate(names)]
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
    plan=dict(schema='monthly_own_error215_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        model_source_binding=source,numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_losses_before_registration=0,actual_new_parameters_before_registration=0,actual_new_quantiles_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_new_own_error_calibration',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare cached own-error quantile calibration and independent audit, then freeze first88 selection.')
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



def original_history_use(arrays,n):
    """Historical original U; source model availability is a separate model input."""
    a=slice_arrays(arrays,n)
    return a['forecast_available'] & a['eligible']

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


def prefix_inputs(train, training_meta, query_meta, source, historical_use, n):
    """Physical source/history prefix with future label rows removed before numeric use."""
    rows=len(query_meta['dates']); shape=np.asarray(historical_use).shape
    require(len(shape)==2 and shape[0]==rows and shape[1]>0 and type(n) is int and 0<=n<=rows,'Invalid physical model prefix')
    q=n*shape[1]; qm=dict(query_meta)
    for key in ('dates','ends','symbols'):
        require(isinstance(query_meta.get(key),list) and len(query_meta[key])==rows,'Query metadata axis '+key)
        qm[key]=[list(v) if isinstance(v,list) else v for v in query_meta[key][:n]]
    for key,value in source.items():require(np.asarray(value).shape==(rows*shape[1],),'Source query axis '+key)
    smallsource={key:np.asarray(value)[:q].copy() for key,value in source.items()}
    months=len(training_meta['dates']); tm=dict(training_meta)
    for key in ('dates','ends','symbols'):
        require(isinstance(training_meta.get(key),list) and len(training_meta[key])==months,'Training metadata axis '+key)
    widths=np.array([len(row) for row in training_meta['symbols']],dtype=int)
    offsets=np.r_[0,np.cumsum(widths)]
    require(np.asarray(train['offsets']).dtype.kind in 'iu' and np.array_equal(train['offsets'],offsets),'Training offsets differ from identity metadata')
    for key in TRAIN_KEYS:require(np.asarray(train[key]).shape[0]==int(offsets[-1]),'Training row axis '+key)
    dates=np.asarray(training_meta['dates'],dtype='datetime64[D]')
    ends=np.asarray(training_meta['ends'],dtype='datetime64[D]')
    require(not np.isnat(dates).any() and not np.isnat(ends).any() and np.all(ends>dates)
            and (len(dates)<2 or np.all(dates[1:]>dates[:-1])),'Training metadata calendar')
    for key,values in (('dates',dates),('ends',ends)):
        require(np.array_equal(np.asarray(train[key],dtype='datetime64[D]'),np.repeat(values,widths)),'Training flat date identity '+key)
    keep=np.zeros(months,bool)
    if n:
        cutoff=np.datetime64('%04d-12-31'%(int(qm['dates'][-1][:4])-1),'D');keep=ends<=cutoff
    ids=np.flatnonzero(keep); flat=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids]) if len(ids) else np.array([],dtype=int)
    for key in ('dates','ends','symbols','pool_origins'):
        if key in training_meta:
            require(len(training_meta[key])==months,'Training metadata field axis '+key)
            tm[key]=[list(training_meta[key][i]) if isinstance(training_meta[key][i],list) else training_meta[key][i] for i in ids]
    smalltrain={key:np.asarray(train[key])[flat].copy() for key in TRAIN_KEYS}
    smalltrain['offsets']=np.r_[0,np.cumsum(widths[ids])]
    return smalltrain,tm,qm,smallsource,np.asarray(historical_use)[:n].copy()

def require_model_audit(check,forecast,expected_rows=5978,expected_years=11):
    for key in MODEL_FLAGS:require(check.get(key) is True,'Independent model evidence missing '+key)
    require(check.get('computes_returns') is False and check.get('producer_modules_imported') is False and check.get('atol')==0,'Independent model scope/tolerance differs')
    require(check.get('annual_cohorts')==expected_years and check.get('query_rows')==expected_rows,'Independent model coverage differs')
    require(set(forecast['raw'])==set(RAW_KEYS),'Raw forecast keys differ')
    available=np.asarray(forecast['available'])
    require(available.shape==(expected_rows,) and available.dtype==np.dtype(bool),'Forecast availability axes differ')
    for key,value in forecast['raw'].items():
        value=np.asarray(value)
        require(value.shape==(expected_rows,) and value.dtype==np.dtype('float64') and not np.isinf(value).any()
                and np.array_equal(np.isfinite(value),available),'Raw forecast support differs '+key)
    audit=forecast['audit'];counts=audit['counts'];annual=audit['annual']
    require(isinstance(annual,list) and len(annual)==expected_years,'Annual calibration coverage')
    require(isinstance(counts,dict) and set(counts)==set(MODEL_COUNT_FIELDS)
            and all(type(v) is int and v>=0 for v in counts.values()),'Integer calibration accounting required')
    expected={key:0 for key in MODEL_COUNT_FIELDS};expected['annual_cohorts']=expected_years
    expected['cached_forecast_rows']=int(np.count_nonzero(available))
    predicted=0
    for year in annual:
        groups=year['groups'];require(isinstance(groups,list),'Annual own groups missing')
        require(len({g['symbol'] for g in groups})==len(groups),'Duplicate annual security groups')
        supported=sum(g['status']=='calibrated' for g in groups)
        require(all(g['status'] in ('calibrated','cold_start') for g in groups),'Unknown own-history status')
        local=dict(queried_security_count=len(groups),supported_groups=supported,cold_start_groups=len(groups)-supported,
                   quantile_estimates=2*supported,prediction_rows=sum(len(g['query_row_ids']) for g in groups))
        for key in ('joined_label_rows','invalid_label_rows','skipped_initial_rows'):
            require(all(type(g.get(key)) is int and g[key]>=0 for g in groups),'Invalid own group count '+key)
            local[key]=sum(g[key] for g in groups)
        local['error_observations']=sum(len(g['history']) for g in groups)
        for g in groups:
            require(type(g.get('own_rows')) is int and type(g.get('own_months')) is int and
                    g['own_rows']==g['own_months']==len(g['history']),'Own monthly support/count differs')
            require((g['own_months']>=18)==(g['status']=='calibrated'),'Own-history threshold differs')
            require(g['joined_label_rows']==g['invalid_label_rows']+g['own_rows'],'Joined-label accounting differs')
        require(all(type(year.get(key)) is int and year[key]==v for key,v in local.items()),'Annual own-error accounting differs')
        expected['own_history_groups']+=len(groups);expected['supported_year_security_groups']+=supported
        for key in ('cold_start_groups','quantile_estimates','joined_label_rows','invalid_label_rows','skipped_initial_rows','error_observations'):
            expected[key]+=local[key]
        predicted+=local['prediction_rows']
    require(counts==expected and predicted==expected['cached_forecast_rows'],'Own-error count identities differ')
    require(isinstance(check.get('counts'),dict) and set(check['counts'])==set(counts)
            and all(type(v) is int and v>=0 for v in check['counts'].values())
            and check['counts']==counts,'Independent calibration accounting differs')
    require(audit.get('query_rows')==expected_rows and audit.get('query_months')*audit.get('symbols_per_event')==expected_rows,
            'Calibration query metadata differs')
    require(audit.get('min_months')==18 and audit.get('lookback_calendar_years')==5 and
            audit.get('historical_errors')=='cached raw_local_linear minus unchanged target_log_excess',
            'Fixed own-error protocol changed')
    for key in ('historical_use_explicit','current_availability_unchanged'):
        require(audit.get(key) is True,'Own-error audit scope '+key)
    for key in ('source_forecasts_refitted','labels_recomputed','local_available_filtered'):
        require(audit.get(key) is False,'Own-error audit scope '+key)
    for key in ('new_model_fits','predictive_model_fits'):
        require(type(audit.get(key)) is int and type(check.get(key)) is int and
                audit[key]==check[key]==counts['quantile_estimates'],'Two calibration parameters per supported group must be counted '+key)
    for key in ('new_ridge_fits','new_tree_fits'):
        require(type(audit.get(key)) is int and type(check.get(key)) is int and audit[key]==check[key]==0,
                'Cached source model must not be refitted '+key)


def require_slot_audit(check,deltas):
    require(set(deltas)==set(RAW),'Delta arm keys differ')
    for value in deltas.values():
        require(np.asarray(value).shape==(122,49) and np.isfinite(value).all() and np.max(np.abs(value))<=.0125+1e-12 and np.array_equal(value[0],np.zeros(49)),'Delta axes/bounds/initial allocation differ')
    for key in SLOT_FLAGS:require(check.get(key) is True,'Independent slot evidence missing '+key)
    require(set(check.get('ready_rows',{}))==set(RAW),'Missing readiness coverage')
    require(all(type(v) is int and 0<=v<=122 for v in check['ready_rows'].values()) and
            len(set(check['ready_rows'].values()))==1,'Candidate readiness differs')
    require(check.get('changed_cells')=={key:int(np.count_nonzero(value)) for key,value in deltas.items()},'Changed-cell audit differs')



def require_cached_source(source, arrays):
    require(set(source)==SOURCE_KEYS,'Cached206 forecast keys differ')
    shape=np.asarray(arrays['baseline']).shape;q=int(np.prod(shape))
    require(len(shape)==2 and shape[1]>0,'Original query axes required')
    for key in SOURCE_KEYS:
        require(np.asarray(source[key]).shape==(q,),'Cached206 source axes differ '+key)
    for key in ('source_log','local_linear','local_constant'):
        supplied=np.asarray(source['raw_'+key]);original=np.asarray(arrays['raw_'+key]).reshape(-1)
        require(supplied.dtype==original.dtype==np.dtype('float64'),'Cached206 forecast dtype differs '+key)
        np.testing.assert_array_equal(supplied,original,err_msg='Cached206 original vector '+key)
    np.testing.assert_array_equal(source['raw_control'],source['raw_local_constant'])
    for src,dst in (('available','model_available'),('local_available','local_available')):
        supplied=np.asarray(source[src]);original=np.asarray(arrays[dst]).reshape(-1)
        require(supplied.dtype==original.dtype==np.dtype(bool),'Cached206 availability dtype differs '+src)
        np.testing.assert_array_equal(supplied,original)
    return dict(passed=True,error_prediction_source='raw_local_linear',original_cached_vectors_exact=True,
                availability_exact=True)


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_own_error215_model as model
    import monthly_own_error215_independent as independent
    study=local();m,a,px,op=study.load()
    require(len(m['dates'])==122 and a['baseline'].shape==(122,49) and len(a)==27,'Original122x49/27 arrays required')
    before={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()}
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original row identity changed')
    source=ev.load_arrays(SOURCE/'prepared/FORECASTS.npz')
    require(read(SOURCE/'prepared/META.json')==m,'Cached206 query identity differs')
    cached_check=require_cached_source(source,a)
    inputs=prefix_inputs(train,metadata['training'],m,source,original_history_use(a,122),122)
    forecast=model.build_forecasts(*inputs)
    np.testing.assert_array_equal(forecast['available'],source['available'])
    check=independent.check_forecasts(*inputs,forecast);require_model_audit(check,forecast)
    orders,ranks,deltas=decisions(study,a,m,forecast,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    slot=independent.verify_rank_slot(a,forecast,ranks,deltas);require_slot_audit(slot,deltas)
    prefix_evidence=[]
    for n in MODEL_PREFIXES:
        smallinputs=prefix_inputs(train,metadata['training'],m,source,original_history_use(a,122),n)
        small=model.build_forecasts(*smallinputs);q=n*49
        for key in RAW_KEYS:np.testing.assert_array_equal(small['raw'][key],forecast['raw'][key][:q])
        np.testing.assert_array_equal(small['available'],forecast['available'][:q])
        aa=slice_arrays(a,n);require(len(aa)==27 and all(value.shape[0]==n for value in aa.values()),'All27 physical cache axes required')
        oo,rr,dd=decisions(study,aa,smallinputs[2],small,n)
        require(all(oo[key]==orders[key][:n] for key in NAMES),'Physical prefix order parity failed')
        for key in RAW:
            np.testing.assert_array_equal(rr[key],ranks[key][:n]);np.testing.assert_array_equal(dd[key],deltas[key][:n])
        prefix_evidence.append(dict(prefix=n,training_rows=len(smallinputs[0]['dates']),query_rows=q,
            query_date=m['dates'][n-1] if n else None,
            cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(smallinputs[0]['ends'],dtype='datetime64[D]').max()) if len(smallinputs[0]['ends']) else None,
            source_history_starts_at_original_zero=True,counts=small['audit']['counts']))
    require(before=={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()},'Original arrays modified')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{'raw_'+key:value for key,value in forecast['raw'].items()},available=forecast['available'])
    np.savez_compressed(dest/'RANKS_DELTAS.npz',**{'rank_'+key:v for key,v in ranks.items()},**{'delta_'+key:v for key,v in deltas.items()})
    dump(dest/'MODEL_AUDIT.json',forecast['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',m)
    dump(dest/'SOURCE_PARITY.json',dict(original_training_cache_exact=True,original_identity_exact=True,
        raw_source_hashes=plan['raw_source_hashes'],model_source_binding=plan['model_source_binding'],
        original_training_cache_exact_basis='Pinned TRAIN_INPUTS bytes and saved169/206 parity certificates reused; not a fresh original-array reconstruction',
        fresh_training_array_comparison=False,saved_training_parity_sha256=sha(REBUILD/'PARITY.json'),
        cached206_source_vectors_exact=True,cached206_control_constant_exact=True,source_availability_exact=True,
        error_prediction_source='raw_local_linear',cached_source_check=cached_check,historical_original_universe_exact=True,
        target_source='original169_target_log_excess_without_recentring',
        no_source_model_refits=True,no_fitted_residual_error_inputs=True,new_labels=False,new_raw_features=False))
    evidence=dict(model=check,slot=slot,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        model_prefix_sources_truncated=True,model_prefix_metadata_truncated=True,model_prefix_original_arrays_truncated=True,
        model_prefix_future_labels_removed=True,source_history_starts_at_original_zero=True,model_prefix_evidence=prefix_evidence,
        original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,
        original_core_metadata_and_support_preserved=True,single_original_episode=True,
        no_new_score_budget=True,historical_original_universe_exact=True,error_prediction_source_local_linear=True,model_counts=forecast['audit']['counts'])
    dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,
        original_array_count=27,model_counts=forecast['audit']['counts'],
        source_model_refits=0,model_fit_accounting=contract()['model_fit_accounting']))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def verify_prepared_evidence(c,forecast,deltas,audit,meta):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require_model_audit(audit['model'],forecast);require_slot_audit(audit['slot'],deltas)
    require(audit.get('prefixes')==123 and audit.get('model_prefixes')==list(MODEL_PREFIXES),'Missing prefix coverage')
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
                'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged',
                'original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
                'historical_original_universe_exact','error_prediction_source_local_linear'):
        require(audit.get(key) is True,'Prepared invariant missing: '+key)
    require(audit.get('original_array_count')==c.get('original_array_count')==27,'Original array count differs')
    require(c.get('new_returns_computed') is False,'Preparation computed portfolio returns')
    counts=forecast['audit']['counts']
    require(audit.get('model_counts')==c.get('model_counts')==counts,'Main-pass model accounting differs')
    evidence=audit.get('model_prefix_evidence')
    require(isinstance(evidence,list) and [row.get('prefix') for row in evidence]==list(MODEL_PREFIXES),'Model prefix evidence missing')
    require(all(row.get('query_rows')==row['prefix']*49 and type(row.get('training_rows')) is int and row['training_rows']>=0 for row in evidence),'Model prefix row coverage differs')
    require(evidence[0]['training_rows']==0 and evidence[0].get('latest_label_end') is None and evidence[0].get('query_date') is None and evidence[0].get('cutoff') is None,'Empty prefix contains source rows')
    for row in evidence[1:]:
        day=meta['dates'][row['prefix']-1];cutoff='%04d-12-31'%(int(day[:4])-1)
        require(row.get('query_date')==day and row.get('cutoff')==cutoff and
                (row['latest_label_end'] is None or row['latest_label_end']<=cutoff) and row.get('source_history_starts_at_original_zero') is True,'Physical prefix maturity clock differs')


def prepared(expected):
    gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared artifact '+name)
    values=ev.load_arrays(dest/'FORECASTS.npz')
    require(set(values)=={'raw_'+key for key in RAW_KEYS}|{'available'},'Stored forecast schema differs')
    forecast=dict(raw={key:values['raw_'+key] for key in RAW_KEYS},available=values['available'],
        audit=read(dest/'MODEL_AUDIT.json'))
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
        next_step='Review annual own-error quantile calibration and source limitations; preserve frozen observers.')
    require(len(state['completed']) == 213 and sum(r['trials'] for r in state['completed']) == 1156, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
