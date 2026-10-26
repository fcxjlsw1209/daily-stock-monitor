"""Round226: joint versus diagonal as-of forecast-error risk for the third holding.
Actual covariance estimates, choices and returns require a frozen PLAN.
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
OUT = ROOT/'outputs/monthly_joint_error226'
LOCAL = ROOT/'outputs/monthly_local_linear70'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
PRIOR = ROOT/'outputs/monthly_relative_duration225'
SOURCE = ROOT/'outputs/monthly_identity_transfer206'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67','joint_prediction_error_variance','diagonal_prediction_error_variance')
RAW = {NAMES[1]:'candidate', NAMES[2]:'control'}
RAW_KEYS = ('candidate','control')
SOURCE_KEYS = {'raw_source_log','raw_local_linear','raw_local_constant','raw_candidate','raw_control','available','local_available'}
TRAIN_KEYS = ('dates','ends','train_plain','common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility')
TARGETS = ('simple_excess','log_excess','vol_scaled','loss_utility')
TRAIN = 88
MODEL_PREFIXES = (0,1,53,88,122)
MODEL_COUNT_FIELDS = ('query_months', 'initial_fallback_queries', 'domain_fallback_queries', 'history_fallback_queries', 'supported_queries', 'boundary_candidates', 'attempted_history_months', 'complete_history_months', 'incomplete_history_months', 'joined_label_rows', 'invalid_label_rows', 'missing_security_rows', 'unsupported_source_rows', 'skipped_initial_months', 'error_observations', 'security_mean_estimates', 'security_variance_estimates', 'joint_variance_estimates', 'diagonal_variance_estimates', 'parameter_estimates', 'cached_forecast_rows')
CONTEXT = ROOT/'work/monthly_joint_error226_context.json'
CONTEXT_SHA = 'e3e73b2c5294d67377f28b0c778605dff4d38bb9833ac8058e03ac3b14f13224'
COVERAGE = ROOT/'work/after225_prediction_error_covariance_coverage.json'
COVERAGE_SHA = 'bfc526e34c4a1473340aa14792c555ded106c3d323a74f37e161fc468dbfa26a'
SOURCE_REVIEW = ROOT/'work/joint_error226_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '1c0c210004408e40a3f230f7ad8cf533877dfbf4aff04887608bfb7b4e61ae76'
TEST_RECEIPT = ROOT/'work/monthly_joint_error226_TESTS.json'
EXTERNAL_RUNTIME = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED = {'PLAN.json':'3d76cbca8e7508830aa1660d6b080a61d8da19a3b5ed5d48a6038bc90001273a','COMPLETION.json':'e193922b7ca753eb4b84c7236fcbee9ef35edd1c5bb8f535774b05adef81de40','results/COMPLETION.json':'9e0c8766ff3151e90d310c7804142ea09f918530adf13fc5a3f88cb505e0b418','RESEARCH_REVIEW.json':'2ccbe280f21f0f44a846539ee98ab93749f85b78057bb66cfd92b74cac89ff86'}


MODEL_FLAGS = ('passed', 'independent_scalar_rebuild', 'exact_identity_join_checked', 'cached_local_linear_reused', 'annual_prior_december_cutoffs_checked', 'five_year_mature_window_checked', 'four_target_common_cohort_checked', 'historical_original_universe_checked', 'fixed_top_two_checked', 'original_boundary_domain_checked', 'common_complete_history_checked', 'minimum_18_months_checked', 'original_targets_unchanged', 'column_centering_checked', 'joint_gram_identity_checked', 'gram_positive_semidefinite_checked', 'matched_diagonal_checked', 'negative_risk_effect_ranks_checked', 'constant_risk_zero_effect_checked', 'paired_fallback_checked', 'no_future_or_outcome_inputs', 'audit_checked', 'estimates_counted')
SELECTION_FLAGS = ('passed', 'independent_original_pre_checked', 'original_support_checked', 'original_budget_checked', 'fixed_top_two_checked', 'replacement_not_additive_checked', 'one_episode_per_arm', 'paired_fallback_checked', 'exact_scores_and_choices_checked')
SOURCE_PREPARED_FILES = {'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
PREPARED_FILES = {'RISKS_EFFECTS.npz','SCORES.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
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
    return module('monthly_joint_error226_original', LOCAL / 'study.py')

def contract():
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective226 context')
    require(sha(COVERAGE)==COVERAGE_SHA,'Changed limited coverage evidence')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
        arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],period=['2016-08-17','2026-09-24'],
        training_months=88,later_months=34,model_prefixes=list(MODEL_PREFIXES),all123_order_prefixes=True,
        physical_model_prefixes=True,minimum_common_months=18,lookback_years=5,
        history_annual_prior_december_cutoff=True,whole_domain_shared_complete_history=True,
        fixed_original_top2=True,third_rank_ceiling=5,third_gap=.02,no_gap_epsilon=True,
        original_episode_budget=.05,replace_original_episode_effects=True,constant_risk_effect_zero=True,
        no_new_score_budget=True,source_models_refitted=False,new_covariance_estimates=True,
        model_fit_accounting=context['accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_joint_error226_host.py',
        'test_monthly_joint_error226_host.py','monthly_joint_error226_model.py',
        'test_monthly_joint_error226_model.py','monthly_joint_error226_independent.py',
        'test_monthly_joint_error226_independent.py','monthly_joint_error226_context.json','after225_prediction_error_covariance_coverage.json',
        'joint_error226_SOURCE_REVIEW.json',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_joint_error226_model as model
    import monthly_joint_error226_independent as independent
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
    backup = CENTRAL.parent / ('PRE_MONTHLY_JOINT_ERROR226_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/joint_error226_CODE_REVIEW.json'
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
        close_vs_diagonal_error_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_diagonal_error_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_joint_error226_independent as independent
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
    import monthly_joint_error226_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, joint_error_portfolio_variance=True,cached_source_models_refitted=False, same_support_both_arms=True, observer_promotion=False))
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
    require(len(state['completed'])==223 and sum(row['trials'] for row in state['completed'])==1186,
            'Expected223 completed entries /1186 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round225 must be the completed predecessor')


def model_source_binding(raw):
    """Metadata and byte provenance only; no forecast/label arrays are decoded."""
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed cached-source review')
    review=read(SOURCE_REVIEW); context=read(CONTEXT)
    require(review.get('passed') is True and review.get('source_only') is True and
            review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    require(review.get('evidence_sha256')==context['evidence_sha256'] and
            review.get('coverage_sha256')==COVERAGE_SHA,'Source/coverage evidence differs from context')
    for key in ('source_admission_metadata_only','cached_forecasts_are_prior_annual_asof_replays',
                'not_realtime_archival_vintages','source_bytes_reverified','metadata_axes_verified',
                'calendar_clock_verified','annual_cutoff_calendar_verified','raw_source_hashes_verified',
                'raw_source_map_matches_original'):
        require(review.get(key) is True,'Source scope missing '+key)
    for key in ('uses_fitted_residuals_as_historical_errors','source_npz_decoded','npz_decoded',
                'training_npz_decoded','actual_joint_history_support_computed','local_available_filter',
                'actual_new_features','actual_new_orders','actual_new_returns','observers_modified'):
        require(review.get(key) is False,'Source review scope differs '+key)
    require(type(review.get('actual_new_estimates')) is int and review['actual_new_estimates']==0,
            'Source review performed new estimates')
    require(review.get('current_and_historical_prediction_key')=='raw_local_linear' and
            review.get('target_key')=='target_log_excess' and
            review.get('historical_use')=='originalU AND206available','Joint error source identity/domain differs')
    require(review.get('matched_formal_months')==113 and len(review.get('source_annual',[]))==11,
            'Source history metadata incomplete')
    require(review.get('raw_source_hashes')==raw and len(raw)==64,'Original raw64 map differs')
    require(isinstance(review.get('source_sha256'),dict) and review['source_sha256'],'Full source pin closure required')
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['source_sha256'].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    for name,digest in context['evidence_sha256'].items():bind(pins,ROOT/name,digest)
    source_plan=read(SOURCE/'PLAN.json')
    require(source_plan['raw_source_hashes']==raw and len(raw)==64,'Original raw source lineage changed')
    for name,digest in source_plan['input_sha256'].items():bind(pins,Path(name),digest)
    closed=read(SOURCE/'COMPLETION.json'); prepared=read(SOURCE/'prepared/COMPLETION.json')
    require(closed['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and prepared['status']=='COMPLETE'
            and closed['plan_sha256']==prepared['plan_sha256']==sha(SOURCE/'PLAN.json'),'206 source closure incomplete')
    require(set(prepared['artifacts_sha256'])==SOURCE_PREPARED_FILES,'206 exact prepared manifest required')
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
    require(not TEST_RECEIPT.exists(),'Test receipt is write-once')
    names=('test_monthly_joint_error226_host.py','test_monthly_joint_error226_model.py',
           'test_monthly_joint_error226_independent.py')
    modules=[module('test_joint_error226_'+str(i),ROOT/'work'/name) for i,name in enumerate(names)]
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
    plan=dict(schema='monthly_joint_error226_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        model_source_binding=source,numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_losses_before_registration=0,actual_new_parameters_before_registration=0,actual_new_covariance_estimates_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_joint_error_statistics',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare joint versus diagonal historical prediction error variance on a shared mature cohort, then freeze first88 selection.')
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

def slice_built(built,n):
    require(set(built)=={'arrays','audit'},'Risk build schema differs')
    values=built['arrays']; shape=np.asarray(values['candidate_risk']).shape
    require(len(shape)==2 and type(n) is int and 0<=n<=shape[0],'Invalid risk prefix')
    require(set(values)=={'candidate_risk','control_risk','candidate_effect','control_effect','ready'},'Risk array keys differ')
    audit=dict(built['audit']);events=audit['event_history'][:n]
    audit.update(query_months=n,event_history=events,
        counts={key:sum(e['counts'][key] for e in events) for key in MODEL_COUNT_FIELDS})
    audit.update(new_statistical_estimation_queries=audit['counts']['supported_queries'],
        parameter_estimates=audit['counts']['parameter_estimates'])
    return dict(arrays={key:np.asarray(v)[:n].copy() for key,v in values.items()},audit=audit)





def decisions(meta,arrays,built,n,study=None):
    study=local() if study is None else study;a=slice_arrays(arrays,n);b=slice_built(built,n)
    m=dict(meta)
    for key in ('dates','ends','symbols'):m[key]=meta[key][:n]
    require(len(m['dates'])==n,'Decision metadata prefix differs')
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choice,score=study.choices(study.configs()[4],a,m,n,policy=policy)
    choices={NAMES[0]:choice};scores={NAMES[0]:score}
    pre=study.modified_pre_score(study.configs()[4],a,n)
    ready=b['arrays']['ready'];require(ready.dtype==np.dtype(bool) and ready.shape==(n,),'Decision ready mask')
    for name,key in RAW.items():
        effects=a['effects'].copy()
        effects[ready]=b['arrays'][key+'_effect'][ready]
        choice,score=study.adapter.score_and_choose('incumbent66',0.,np.full(pre.shape,np.nan),a,m,n,
            pre_score=pre.copy(),effects=effects,episode_policy=policy,boundary_selector=study.ep.select_boundary)
        np.testing.assert_array_equal(score[~ready],scores[NAMES[0]][~ready])
        require(all(choice[i]==choices[NAMES[0]][i] for i in np.flatnonzero(~ready)),'Whole-query fallback differs')
        choices[name]=choice;scores[name]=score
    return {name:study.rp.make_orders(choices[name],scores[name],m) for name in NAMES},scores,choices



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

def require_model_audit(check,built,rows):
    for key in MODEL_FLAGS:require(check.get(key) is True,'Independent model evidence missing '+key)
    require(check.get('query_rows')==rows and check.get('numerical_tolerance')==0. and
        check.get('gram_relative_tolerance')==1e-12 and check.get('producer_modules_imported') is False and
        check.get('computes_returns') is False,'Independent model scope/coverage differs')
    require(set(built)=={'arrays','audit'},'Build mapping differs')
    a=built['audit'];v=built['arrays'];counts=a['counts'];events=a['event_history']
    require(a.get('query_months')==rows and a.get('method')=='joint_asof_error_boundary_variance' and
        a.get('arms')==list(NAMES) and a.get('min_months')==18 and a.get('lookback_calendar_years')==5 and
        a.get('ceiling')==5 and a.get('gap')==.02,'Fixed joint-error protocol differs')
    require(set(counts)==set(MODEL_COUNT_FIELDS) and all(type(x) is int and x>=0 for x in counts.values()),'Integer estimation accounting required')
    require(isinstance(events,list) and len(events)==rows,'Missing event histories')
    width=a['symbols_per_event'];require(type(width) is int and width>0,'Invalid width')
    require(set(v)=={'candidate_risk','control_risk','candidate_effect','control_effect','ready'},'Risk array keys differ')
    ready=np.asarray(v['ready']);require(ready.dtype==bool and ready.shape==(rows,),'Ready axes/dtype')
    mask=np.zeros((rows,width),bool)
    for i,e in enumerate(events):
        c=e['counts'];require(set(c)==set(MODEL_COUNT_FIELDS) and all(type(x) is int and x>=0 for x in c.values()),'Event integer counts')
        require(e['query_index']==i and c['query_months']==1 and len(e['symbols'])==width,'Event query coverage')
        require(e['status'] in ('initial_partial','insufficient_candidates','insufficient_history','ready'),'Invalid event status')
        supported=e['status']=='ready';require(bool(ready[i])==supported,'Status/readiness differs')
        js=e['J'];b=e['B'];require(b==e['ordered_top2']+js and len(set(b))==len(b),'Fixed common domain differs')
        if supported:
            require(len(e['ordered_top2'])==2 and len(js)>=2 and len(e['H'])>=18,'Insufficient common support')
            mask[i,js]=True
        expected=2*len(b)+2*len(js) if supported else 0
        require(c['parameter_estimates']==expected and
            c['security_mean_estimates']==c['security_variance_estimates']==(len(b) if supported else 0) and
            c['joint_variance_estimates']==c['diagonal_variance_estimates']==(len(js) if supported else 0),'Parameter accounting differs')
        require(c['supported_queries']==int(supported),'Supported count differs')
    require(counts=={key:sum(e['counts'][key] for e in events) for key in MODEL_COUNT_FIELDS} and
        counts['query_months']==rows and counts['supported_queries']==int(ready.sum()) and
        sum(counts[k] for k in ('initial_fallback_queries','domain_fallback_queries','history_fallback_queries','supported_queries'))==rows,
        'Query count identities differ')
    require(check.get('counts')==counts,'Independent counts differ')
    for key in ('source_model_refits','new_ridge_fits','new_tree_fits'):
        require(type(a.get(key)) is int and type(check.get(key)) is int and a[key]==check[key]==0,'Source models refitted '+key)
    for key,countkey in (('new_statistical_estimation_queries','supported_queries'),('parameter_estimates','parameter_estimates')):
        require(type(a.get(key)) is int and type(check.get(key)) is int and a[key]==check[key]==counts[countkey],'Estimation count differs '+key)
    for key in ('source_forecasts_refitted','labels_recomputed','local_available_filtered'):
        require(a.get(key) is False,'Unexpected source mutation '+key)
    for key in ('historical_use_explicit','common_complete_history'):
        require(a.get(key) is True,'Missing source invariant '+key)
    for key in RAW_KEYS:
        risk=np.asarray(v[key+'_risk']);effect=np.asarray(v[key+'_effect'])
        require(risk.dtype==effect.dtype==np.dtype('float64') and risk.shape==effect.shape==mask.shape and
            not np.isinf(risk).any() and np.array_equal(np.isfinite(risk),mask) and (risk[mask]>=0).all(),
            'Risk support/values differ '+key)
        require(np.isfinite(effect).all() and (np.abs(effect)<=.5).all() and not effect[~mask].any(),
            'Effect support/bounds differ '+key)



def require_selection_audit(check,rows):
    for key in SELECTION_FLAGS:require(check.get(key) is True,'Independent selection evidence missing '+key)
    require(check.get('ordered_choices_checked')==3*rows and check.get('numerical_tolerance')==0. and
        check.get('original_source_parity_tolerance')==1e-12 and check.get('producer_modules_imported') is False and
        check.get('computes_returns') is False,'Independent selection scope differs')




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
    import monthly_joint_error226_model as model
    import monthly_joint_error226_independent as independent
    study=local();m,a,px,op=study.load()
    require(len(m['dates'])==122 and a['baseline'].shape==(122,49) and len(a)==27,'Original122x49/27 arrays required')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original row identity changed')
    source=ev.load_arrays(SOURCE/'prepared/FORECASTS.npz')
    require(read(SOURCE/'prepared/META.json')==m,'Cached206 query identity differs')
    cached_check=require_cached_source(source,a)
    def snapshot(values):return {key:(v.dtype.str,v.shape,v.tobytes()) for key,v in values.items()}
    before=[snapshot(v) for v in (a,train,source)]
    pre=study.modified_pre_score(study.configs()[4],a,122)
    use=original_history_use(a,122);eligible=slice_arrays(a,122)['eligible']
    inputs=prefix_inputs(train,metadata['training'],m,source,use,122)+(pre,eligible)
    built=model.build(*inputs)
    check=independent.check_build(*inputs,built);require_model_audit(check,built,122)
    orders,scores,choices=decisions(m,a,built,122,study=study)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    selection=independent.verify_selection(a,m,built,scores,choices);require_selection_audit(selection,122)
    prefix_evidence=[]
    for n in MODEL_PREFIXES:
        smallinputs=prefix_inputs(train,metadata['training'],m,source,use,n)+(pre[:n].copy(),eligible[:n].copy())
        small=model.build(*smallinputs)
        smallcheck=independent.check_build(*smallinputs,small);require_model_audit(smallcheck,small,n)
        for key in built['arrays']:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
        aa=slice_arrays(a,n);require(len(aa)==27 and all(value.shape[0]==n for value in aa.values()),'All27 physical cache axes required')
        oo,ss,cc=decisions(smallinputs[2],aa,small,n,study=study)
        smallselection=independent.verify_selection(aa,smallinputs[2],small,ss,cc);require_selection_audit(smallselection,n)
        require(all(oo[key]==orders[key][:n] and cc[key]==choices[key][:n] for key in NAMES),'Physical prefix order parity failed')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
        prefix_evidence.append(dict(prefix=n,training_rows=len(smallinputs[0]['dates']),query_rows=n,
            query_date=m['dates'][n-1] if n else None,cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(smallinputs[0]['ends'],dtype='datetime64[D]').max()) if len(smallinputs[0]['ends']) else None,
            source_history_starts_at_original_zero=True,counts=small['audit']['counts'],model=smallcheck,selection=smallselection))
    for n in range(123):
        oo,ss,cc=decisions(m,a,built,n,study=study)
        require(all(oo[key]==orders[key][:n] and cc[key]==choices[key][:n] for key in NAMES),'Order prefix parity failed')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
    require(before==[snapshot(v) for v in (a,train,source)],'Original input arrays modified')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'RISKS_EFFECTS.npz',**built['arrays'])
    np.savez_compressed(dest/'SCORES.npz',**scores)
    dump(dest/'MODEL_AUDIT.json',built['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',m)
    parity=dict(original_training_cache_exact=True,original_identity_exact=True,
        raw_source_hashes=plan['raw_source_hashes'],model_source_binding=plan['model_source_binding'],
        original_training_cache_exact_basis='Pinned TRAIN_INPUTS bytes and saved169/206 parity certificates reused; not a fresh original-array reconstruction',
        fresh_training_array_comparison=False,saved_training_parity_sha256=sha(REBUILD/'PARITY.json'),
        cached206_source_vectors_exact=True,cached206_control_constant_exact=True,source_availability_exact=True,
        error_prediction_source='raw_local_linear',cached_source_check=cached_check,historical_original_universe_exact=True,
        target_source='original169_target_log_excess_without_recentring',
        no_source_model_refits=True,no_fitted_residual_error_inputs=True,new_labels=False,new_raw_features=False)
    dump(dest/'SOURCE_PARITY.json',parity)
    evidence=dict(model=check,selection=selection,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        model_prefix_sources_truncated=True,model_prefix_metadata_truncated=True,model_prefix_original_arrays_truncated=True,
        model_prefix_future_labels_removed=True,source_history_starts_at_original_zero=True,model_prefix_evidence=prefix_evidence,
        original_orders_exact=True,original_arrays_unchanged=True,original_source_arrays_unchanged=True,original_array_count=27,
        original_core_metadata_and_support_preserved=True,single_original_episode=True,
        no_new_score_budget=True,historical_original_universe_exact=True,error_prediction_source_local_linear=True,model_counts=built['audit']['counts'])
    dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    completion=dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,
        original_array_count=27,model_counts=built['audit']['counts'],source_model_refits=0,
        model_fit_accounting=contract()['model_fit_accounting'])
    verify_prepared_evidence(completion,m,evidence,parity,built)
    dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def verify_prepared_evidence(c,meta,audit,parity,built):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(v)==49 for v in meta['symbols']),'Full query metadata required')
    require_model_audit(audit['model'],built,122);require_selection_audit(audit['selection'],122)
    require(audit.get('prefixes')==123 and audit.get('model_prefixes')==list(MODEL_PREFIXES),'Missing prefix coverage')
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged',
        'original_source_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
        'historical_original_universe_exact','error_prediction_source_local_linear'):
        require(audit.get(key) is True,'Prepared invariant missing: '+key)
    require(audit.get('original_array_count')==c.get('original_array_count')==27,'Original array count differs')
    require(c.get('new_returns_computed') is False and type(c.get('source_model_refits')) is int and c['source_model_refits']==0,'Preparation scope differs')
    require(audit.get('model_counts')==c.get('model_counts')==built['audit']['counts'],'Main-pass statistical accounting differs')
    evidence=audit.get('model_prefix_evidence')
    require(isinstance(evidence,list) and [row.get('prefix') for row in evidence]==list(MODEL_PREFIXES),'Model prefix evidence missing')
    for row in evidence:
        n=row['prefix'];require(row.get('query_rows')==n and type(row.get('training_rows')) is int and row['training_rows']>=0,'Model prefix coverage differs')
        require(row.get('source_history_starts_at_original_zero') is True,'Source prefix starts late')
        require_model_audit(row['model'],slice_built(built,n),n);require_selection_audit(row['selection'],n)
        require(row.get('counts')==row['model']['counts'],'Prefix statistical count differs')
        if n:
            day=meta['dates'][n-1];cutoff='%04d-12-31'%(int(day[:4])-1)
            require(row.get('query_date')==day and row.get('cutoff')==cutoff and
                (row['latest_label_end'] is None or row['latest_label_end']<=cutoff),'Physical prefix maturity clock differs')
        else:require(row['training_rows']==0 and all(row.get(k) is None for k in ('latest_label_end','query_date','cutoff')),'Empty prefix contains source rows')
    for key in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact',
        'cached206_control_constant_exact','source_availability_exact','historical_original_universe_exact','no_source_model_refits','no_fitted_residual_error_inputs'):
        require(parity.get(key) is True,'Source parity missing '+key)
    require(parity.get('fresh_training_array_comparison') is False and parity.get('new_labels') is False and
        parity.get('new_raw_features') is False and parity.get('error_prediction_source')=='raw_local_linear' and
        parity.get('target_source')=='original169_target_log_excess_without_recentring','Source parity scope differs')



def prepared(expected):
    plan=gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared artifact '+name)
    built=dict(arrays=ev.load_arrays(dest/'RISKS_EFFECTS.npz'),audit=read(dest/'MODEL_AUDIT.json'))
    m=read(dest/'META.json');audit=read(dest/'INDEPENDENT_PREPARE.json');parity=read(dest/'SOURCE_PARITY.json')
    verify_prepared_evidence(c,m,audit,parity,built)
    require(parity['model_source_binding']==plan['model_source_binding'] and parity['raw_source_hashes']==plan['raw_source_hashes'] and
        parity['saved_training_parity_sha256']==sha(REBUILD/'PARITY.json'),'Stored source proof differs from registration')
    scores=ev.load_arrays(dest/'SCORES.npz');orders=read(dest/'ORDERS.json')
    require(set(scores)==set(orders)==set(NAMES),'Stored score/order arms differ')
    for key in NAMES:
        require(scores[key].shape==(122,49) and scores[key].dtype==np.dtype('float64') and len(orders[key])==122,'Stored score/order axes differ')
    return m,orders


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
        next_step='Review joint versus diagonal prediction error portfolio risk and common-history support; preserve frozen observers.')
    require(len(state['completed']) == 224 and sum(r['trials'] for r in state['completed']) == 1189, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
