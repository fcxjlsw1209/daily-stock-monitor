"""Round230: separate early-history joint versus diagonal third-slot error risk.
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
OUT = ROOT/'outputs/monthly_separate_history230'
LOCAL = ROOT/'outputs/monthly_local_linear70'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
PRIOR = ROOT/'outputs/monthly_expanded_error228'
EARLY = ROOT/'outputs/monthly_early_history229'
EARLY_EXPECTED = {'PLAN.json':'7cd73009ffaae9c6ab4418221049306066da94dc48ca9223f13ab9d24335f1e6',
    'COMPLETION.json':'32920689375b500ff645d243181b8984dbeca7ce686749b5ba083fc9693377cd',
    'prepared/COMPLETION.json':'927787a77d99c738cf979edda2bc872b18c408ccd390a73fe973097417b8994a',
    'RESEARCH_REVIEW.json':'c600d0ba429dc08a15748e1c869b755155e31a11bc320f0fd32b8a9e5d0d4157'}
EXPANDED = ROOT/'outputs/monthly_ineligible_source227'
SOURCE = ROOT/'outputs/monthly_identity_transfer206'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67','joint_early_history_error_variance','diagonal_early_history_error_variance')
RAW = {NAMES[1]:'candidate', NAMES[2]:'control'}
RAW_KEYS = ('candidate','control')
SOURCE_KEYS = {'raw_source_log','raw_local_linear','raw_local_constant','raw_candidate','raw_control','available','local_available'}
TRAIN_KEYS = ('dates','ends','train_plain','common_good','target_simple_excess','target_log_excess','target_vol_scaled','target_loss_utility')
TARGETS = ('simple_excess','log_excess','vol_scaled','loss_utility')
TRAIN = 88
MODEL_PREFIXES = (0,1,53,88,122)
MODEL_COUNT_FIELDS = ('query_months', 'initial_fallback_queries', 'domain_fallback_queries', 'history_fallback_queries', 'supported_queries', 'boundary_candidates', 'attempted_history_months', 'complete_history_months', 'incomplete_history_months', 'joined_label_rows', 'invalid_label_rows', 'missing_security_rows', 'unsupported_source_rows', 'skipped_initial_months', 'error_observations', 'security_mean_estimates', 'security_variance_estimates', 'joint_variance_estimates', 'diagonal_variance_estimates', 'parameter_estimates', 'cached_forecast_rows')
CONTEXT = ROOT/'work/monthly_separate_history230_context.json'
CONTEXT_SHA = 'badb7ce7a17e46c9f44a50e3dd7916f23bc92780610075989f16f223586bf31e'
COVERAGE = ROOT/'work/separate_history230_PROPOSAL.json'
COVERAGE_SHA = '2c5d9d72cb7a43c0ac098b6a01e22522b2fa4b23cb5e25304589093964442103'
SOURCE_REVIEW = ROOT/'work/separate_history230_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '3ca65e6f2b07fe89268d6e7988da4a5eaad153cf1f5288fe9e76fd85aa2a0245'
TEST_RECEIPT = ROOT/'work/monthly_separate_history230_TESTS.json'
EXTERNAL_RUNTIME = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED = {'PLAN.json':'8360603efb9116377401a3b45d879b66944df7d10911d8f73338ec6a92813654','COMPLETION.json':'8d65e817a0c34e79950b957f3263d651b792fbaf6c128681a98f1be1ed59df7f','results/COMPLETION.json':'bd6053de8945d5e3be311e72a0dfad3e81159c42baba9a3cabf2fd96aa357ab5','RESEARCH_REVIEW.json':'0c04840c9199d4e60419872e548bf62ce7f8505b373d5b922322952c5ac21952'}


MODEL_FLAGS = ('passed', 'independent_scalar_rebuild', 'exact_identity_join_checked', 'cached_local_linear_reused', 'annual_prior_december_cutoffs_checked', 'five_year_mature_window_checked', 'four_target_common_cohort_checked', 'historical_original_universe_checked', 'fixed_top_two_checked', 'original_boundary_domain_checked', 'common_complete_history_checked', 'minimum_18_months_checked', 'original_targets_unchanged', 'column_centering_checked', 'joint_gram_identity_checked', 'gram_positive_semidefinite_checked', 'matched_diagonal_checked', 'negative_risk_effect_ranks_checked', 'constant_risk_zero_effect_checked', 'paired_fallback_checked', 'no_future_or_outcome_inputs', 'audit_checked', 'estimates_counted', 'separate_history_axis_checked', 'current_availability_counts_checked')
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
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective230 context')
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
        historical_source_expanded=True,expanded_labels_risk_only=True,current_trading_eligibility_unchanged=True,
        historical_mask='early229.available followed by originalU OR newlyavailable227',
        separate_history_axis=True,history_months=189,current_query_months=122,early_history_months=67,
        source227_completion_sha256=sha(EXPANDED/'COMPLETION.json'),source229_completion_sha256=sha(EARLY/'COMPLETION.json'),
        model_fit_accounting=context['accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    names=('monthly_separate_history230_host.py','test_monthly_separate_history230_host.py',
        'monthly_separate_history230_model.py','test_monthly_separate_history230_model.py',
        'monthly_separate_history230_independent.py','test_monthly_separate_history230_independent.py',
        'monthly_expanded_error228_host.py','test_monthly_expanded_error228_host.py',
        'monthly_expanded_error228_adapter.py','test_monthly_expanded_error228_adapter.py',
        'monthly_expanded_error228_independent.py','test_monthly_expanded_error228_independent.py',
        'monthly_joint_error226_model.py','test_monthly_joint_error226_model.py',
        'monthly_joint_error226_independent.py','test_monthly_joint_error226_independent.py',
        'monthly_early_history229_host.py','monthly_early_history229_model.py','monthly_early_history229_independent.py',
        'monthly_separate_history230_context.json','separate_history230_PROPOSAL.json','after229_research_checkpoint.json',
        'separate_history230_SOURCE_REVIEW.json','ptr169_rebuild_evaluate.py','ptr169_results_independent.py')
    direct=[ROOT/'work'/name for name in names]
    import monthly_separate_history230_model as model
    import monthly_separate_history230_independent as independent
    import monthly_expanded_error228_adapter as adapter
    import monthly_expanded_error228_independent as adapter_independent
    deps=set()
    for obj in (model,independent,adapter,adapter_independent):deps.update(obj.dependency_files())
    return direct+sorted(deps-set(direct))

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
    backup = CENTRAL.parent / ('PRE_MONTHLY_SEPARATE_HISTORY230_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/separate_history230_CODE_REVIEW.json'
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
    import monthly_separate_history230_independent as independent
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
    import monthly_separate_history230_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, joint_error_portfolio_variance=True,separate_history_axis=True,early_history_source_reused=True,cached_source_models_refitted=False, same_support_both_arms=True, observer_promotion=False))
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
    require(len(state['completed'])==225 and sum(row['trials'] for row in state['completed'])==1192,
            'Expected225 completed strategy entries /1192 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round228 must be completed strategy predecessor')
    records=state.get('source_experiments',[])
    require(records and records[-1]['directory']==EARLY.name and records[-1]['strategy_trials']==0,
            'Completed229 source-only predecessor required')
    require(records[-1]['completion']['sha256']==EARLY_EXPECTED['COMPLETION.json'] and
            sha(EARLY/'COMPLETION.json')==EARLY_EXPECTED['COMPLETION.json'],'229 central source receipt changed')

def model_source_binding(raw):
    """Frozen old lineage plus new source-only227 metadata; no arrays decoded."""
    import monthly_joint_error226_host as old
    original=old.model_source_binding(raw)
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed230 source review')
    review=read(SOURCE_REVIEW)
    require(review.get('context_sha256')==CONTEXT_SHA and isinstance(review.get('evidence_sha256'),dict) and bool(review['evidence_sha256']),'230 source-context binding differs')
    for key in ('passed','source_only','metadata_only'):
        require(review.get(key) is True,'Missing230 source flag '+key)
    for key in ('source_npz_decoded','actual_support_computed','actual_returns_computed','observers_modified'):
        require(review.get(key) is False,'230 source review scope '+key)
    pins=dict(original['input_sha256']);bind(pins,SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    for field in ('source_sha256','evidence_sha256'):
        for name,digest in review[field].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    for name,digest in read(CONTEXT)['evidence_sha256'].items():bind(pins,ROOT/name,digest)
    plan=read(EXPANDED/'PLAN.json');closed=read(EXPANDED/'COMPLETION.json');prepared=read(EXPANDED/'prepared/COMPLETION.json')
    require(closed['status']=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and closed['source_only'] is True
        and closed['strategy_trials']==0 and closed['new_portfolio_returns'] is False
        and closed['protected_files']==21444 and closed['observers_modified'] is False,'227 closure scope differs')
    require(closed['plan_sha256']==prepared['plan_sha256']==sha(EXPANDED/'PLAN.json') and
        closed['prepared_sha256']==sha(EXPANDED/'prepared/COMPLETION.json'),'227 receipt chain differs')
    require(set(prepared['artifacts_sha256'])=={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'},'227 exact manifest')
    for p in ('PLAN.json','COMPLETION.json','prepared/COMPLETION.json','RESEARCH_REVIEW.json'):bind(pins,EXPANDED/p)
    for name,digest in plan['input_sha256'].items():bind(pins,Path(name),digest)
    for name,digest in prepared['artifacts_sha256'].items():bind(pins,EXPANDED/'prepared'/name,digest)
    meta=read(EXPANDED/'prepared/META.json');original_meta=read(REBUILD/'MONTHLY_META.json')
    require(meta['monthly']==original_meta['monthly'] and meta['training']==original_meta['training'],'227 exact original identities required')
    for k,proof in read(EXPANDED/'prepared/INDEPENDENT.json').items():require(proof['passed'] is True,'227 independent '+k)
    require(review.get('raw_source_hashes')==raw,'230 raw-source map differs')
    early_binding=early_source_binding(pins,original_meta['training'])
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,original_source_binding=original,
        directory=EXPANDED.name,source_plan_sha256=sha(EXPANDED/'PLAN.json'),
        source_completion_sha256=sha(EXPANDED/'COMPLETION.json'),source_prepared_sha256=sha(EXPANDED/'prepared/COMPLETION.json'),
        source_forecasts_sha256=sha(EXPANDED/'prepared/FORECASTS.npz'),source_labels_sha256=sha(EXPANDED/'prepared/LABELS.npz'),
        early_source_binding=early_binding,
        risk_history_only=True,current_trading_universe_unchanged=True,original_models_not_retrained=True,
        source_model_refits=0,new_labels=False,new_raw_features=False,strict_pit=False)

def early_metadata_prefix(training_meta,n):
    total=len(training_meta['dates'])
    require(type(n) is int and 0<=n<=total,'Invalid early metadata prefix')
    return {key:([list(x) if isinstance(x,list) else x for x in value[:n]]
                 if isinstance(value,list) and len(value)==total else value)
            for key,value in training_meta.items()}


def early_source_binding(pins,training_meta):
    """Metadata/hash-only source229 admission, prior to every actual array load."""
    for name,digest in EARLY_EXPECTED.items():bind(pins,EARLY/name,digest)
    plan=read(EARLY/'PLAN.json');closed=read(EARLY/'COMPLETION.json');prepared=read(EARLY/'prepared/COMPLETION.json')
    require(closed.get('status')=='SOURCE_COMPLETED_INDEPENDENTLY_VERIFIED' and closed.get('source_only') is True
        and closed.get('strategy_trials')==0 and closed.get('new_portfolio_returns') is False
        and closed.get('original_strategy_unchanged') is True and closed.get('protected_files')==21444
        and closed.get('observers_modified') is False,'229 source-only closure scope differs')
    require(closed['plan_sha256']==prepared['plan_sha256']==EARLY_EXPECTED['PLAN.json'] and
        closed['prepared_sha256']==EARLY_EXPECTED['prepared/COMPLETION.json'],'229 closure chain differs')
    require(prepared.get('status')=='COMPLETE' and prepared.get('source_only') is True and
        prepared.get('strategy_trials')==0 and prepared.get('portfolio_returns') is False,'229 preparation scope differs')
    require(set(prepared['artifacts_sha256'])=={'FORECASTS.npz','LABELS.npz','META.json','SOURCE_AUDIT.json','INDEPENDENT.json'},'229 exact source manifest required')
    for name,digest in plan['input_sha256'].items():bind(pins,Path(name),digest)
    for name,digest in prepared['artifacts_sha256'].items():bind(pins,EARLY/'prepared'/name,digest)
    bind(pins,EARLY/'RESEARCH_REVIEW.json',EARLY_EXPECTED['RESEARCH_REVIEW.json'])
    review=read(EARLY/'RESEARCH_REVIEW.json');require(review.get('passed') is True,'229 final saved review required')
    meta=read(EARLY/'prepared/META.json')
    require(meta.get('current_query_axis_unchanged') is True and meta.get('history')==early_metadata_prefix(training_meta,67),
            '229 exact first67 original training identities required')
    require(meta.get('training_reference')==str(REBUILD/'MONTHLY_META.json'),'229 training identity source differs')
    audit=read(EARLY/'prepared/SOURCE_AUDIT.json')
    require(audit.get('copied_label_exact') is True and audit.get('copied_label_rows')==3283 and
        audit.get('source227_label_sha256')==sha(EXPANDED/'prepared/LABELS.npz') and
        audit.get('new_history_months')==67 and audit.get('original_trading_queries')==122 and
        audit.get('no_new_orders') is True and audit.get('no_portfolio_returns') is True,'229 source-copy scope differs')
    import monthly_early_history229_host as early_host
    early_host.require_check(read(EARLY/'prepared/INDEPENDENT.json'),dict(audit=audit['forecasts']))
    return dict(directory=EARLY.name,plan_sha256=EARLY_EXPECTED['PLAN.json'],
        completion_sha256=EARLY_EXPECTED['COMPLETION.json'],prepared_sha256=EARLY_EXPECTED['prepared/COMPLETION.json'],
        forecasts_sha256=sha(EARLY/'prepared/FORECASTS.npz'),labels_sha256=sha(EARLY/'prepared/LABELS.npz'),
        metadata_sha256=sha(EARLY/'prepared/META.json'),review_sha256=sha(EARLY/'RESEARCH_REVIEW.json'),
        early_months=67,early_rows=3283,labels_copied_not_appended=True)


def exact_array(left,right,label):
    a,b=np.asarray(left),np.asarray(right)
    require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'Exact source bytes differ: '+label)


def join_history(training_meta,query_meta,source_meta229,early,early_labels,full_labels227,adapted):
    """Append no labels: prepend admitted229 forecasts to the adapted228 history."""
    import monthly_expanded_error228_adapter as adapter
    import monthly_separate_history230_model as model
    require(set(adapted)=={'train','source','historical_use','audit'},'Adapted228 schema differs')
    require(isinstance(source_meta229,dict) and source_meta229.get('current_query_axis_unchanged') is True,
            '229 source metadata scope differs')
    hm=source_meta229.get('history');require(isinstance(hm,dict),'229 history metadata missing')
    h=len(hm.get('dates',[]));n=len(query_meta['dates'])
    require(hm==early_metadata_prefix(training_meta,h),'Early history is not exact original training prefix')
    use=np.asarray(adapted['historical_use']);require(use.ndim==2 and use.dtype.kind=='b' and use.shape[0]==n and use.shape[1]>0,'Adapted historical-use axis')
    width=use.shape[1];q=h*width
    model._metadata(hm,width);model._metadata(query_meta,width,initial_query_exception=True)
    require(all(day<'2016-08-17' for day in hm['dates']) and
        (not h or not n or hm['dates'][-1]<query_meta['dates'][0]),'Early/current source axes overlap')
    expected_keys={'raw_source_log','raw_local_constant','raw_local_linear','available','local_available','query_eligible','expansion_allowed'}
    require(set(early)==expected_keys,'229 exact forecast schema required')
    for key in ('available','local_available','query_eligible','expansion_allowed'):
        require(np.asarray(early[key]).shape==(q,) and np.asarray(early[key]).dtype.kind=='b','229 boolean axis '+key)
    available=np.asarray(early['available']);qeligible=np.asarray(early['query_eligible']);allowed=np.asarray(early['expansion_allowed'])
    require(not np.any(early['local_available'] & ~available),'229 local availability exceeds pooled support')
    train=adapted['train']
    exact_array(qeligible,np.asarray(train['train_eligible'])[:q],'229 original train eligibility')
    finite=np.isfinite(np.asarray(train['train_plain'])[:q]).all(axis=1)
    require(not np.any(available & ~(finite & (qeligible|allowed))),'229 available outside signal-time union')
    for key in ('raw_source_log','raw_local_constant','raw_local_linear'):
        value=np.asarray(early[key]);require(value.shape==(q,) and value.dtype==np.dtype('float64') and
            not np.isinf(value).any() and np.array_equal(np.isfinite(value),available),'229 finite raw support '+key)
    require(set(early_labels)==set(full_labels227)==adapter.LABEL_KEYS,'229/227 exact label schema required')
    for key,value in full_labels227.items():
        expected=np.asarray(value)[:h+1] if key=='offsets' else np.asarray(value)[:q]
        exact_array(early_labels[key],expected,'229 label prefix '+key)
    # The extended target/common-good arrays already occupy their original rows.
    # In particular229 vol63 is compared to227 LABELS, not the untouched old train vol63.
    for key in ('common_good',)+tuple('target_'+k for k in TARGETS):exact_array(train[key],full_labels227[key],'adapted full227 '+key)
    late=adapted['source'];require(set(late)=={'raw_local_linear','available'},'Adapted risk-source schema')
    late_available=np.asarray(late['available']);late_raw=np.asarray(late['raw_local_linear'])
    require(late_available.dtype.kind=='b' and late_available.shape==(n*width,) and
        late_raw.dtype==np.dtype('float64') and late_raw.shape==(n*width,) and not np.isinf(late_raw).any() and
        np.array_equal(np.isfinite(late_raw),late_available),'Adapted current forecast support differs')
    early_use=available & (qeligible | (~qeligible & allowed))
    require(np.array_equal(early_use,available),'229 history mask must equal admitted source availability')
    combined={key:[list(x) if isinstance(x,list) else x for x in hm[key]+query_meta[key]] for key in ('dates','ends','symbols')}
    model._metadata(combined,width,initial_query_exception=True,history_axis=True)
    history_source=dict(raw_local_linear=np.concatenate([early['raw_local_linear'],late_raw]),
                        available=np.concatenate([available,late_available]))
    historical_use=np.concatenate([early_use.reshape(h,width),use],axis=0)
    audit=dict(early_months=h,current_months=n,history_months=h+n,width=width,early_rows=q,
        current_rows=n*width,history_rows=(h+n)*width,early_available_rows=int(available.sum()),
        current_available_rows=int(late_available.sum()),early_history_use_rows=int(early_use.sum()),
        current_history_use_rows=int(use.sum()),history_use_rows=int(historical_use.sum()),
        exact_early_training_prefix=True,exact_early_label_copy=True,labels_appended=False,
        early_history_use_equals_available=True,current_adapted_history_unchanged=True,
        current_query_axis_unchanged=True,current_available_axis_separate=True,source_forecasts_refitted=False,
        labels_recomputed=False,history_use_rule='early229.available followed by originalU OR newlyavailable227')
    return dict(history_meta=combined,history_source=history_source,historical_use=historical_use,
        current_available=late_available.reshape(n,width).copy(),audit=audit)


def tests():
    require(not TEST_RECEIPT.exists(),'Formal synthetic test receipt is write-once')
    names=('test_monthly_separate_history230_host.py','test_monthly_separate_history230_model.py',
        'test_monthly_separate_history230_independent.py')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module('separate230test'+str(i),ROOT/'work'/name))
                             for i,name in enumerate(names)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TEST_RECEIPT,dict(passed=True,tests=result.testsRun,synthetic_only=True,runtime=ev.runtime(),
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
    plan=dict(schema='monthly_separate_history230_v1',created_utc=now(),runtime=ev.runtime(),
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

def slice_built(built,n,*,history_months=None):
    require(set(built)=={'arrays','audit'},'Risk build schema differs')
    values=built['arrays']; shape=np.asarray(values['candidate_risk']).shape
    require(len(shape)==2 and type(n) is int and 0<=n<=shape[0],'Invalid risk prefix')
    require(set(values)=={'candidate_risk','control_risk','candidate_effect','control_effect','ready'},'Risk array keys differ')
    audit=dict(built['audit']);events=audit['event_history'][:n]
    if history_months is not None:
        require(type(history_months) is int and 0<=history_months<=audit['history_months'],'Invalid history prefix count')
        audit['history_months']=history_months
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



def prefix_inputs(train,training_meta,query_meta,history_meta,source,historical_use,current_available,n):
    """Return six model inputs plus separate current mask, physically truncated."""
    rows=len(query_meta['dates']);use=np.asarray(historical_use);current=np.asarray(current_available)
    require(use.ndim==2 and use.shape[1]>0 and len(history_meta['dates'])==use.shape[0] and
        type(n) is int and 0<=n<=rows,'Invalid separate-axis model prefix')
    width=use.shape[1]
    require(current.shape==(rows,width),'Current availability axis differs')
    qm=dict(query_meta)
    for key in ('dates','ends','symbols'):
        require(isinstance(query_meta.get(key),list) and len(query_meta[key])==rows,'Query metadata axis '+key)
        qm[key]=[list(v) if isinstance(v,list) else v for v in query_meta[key][:n]]
    hm=dict(history_meta);history_rows=use.shape[0]
    for key in ('dates','ends','symbols'):
        require(isinstance(history_meta.get(key),list) and len(history_meta[key])==history_rows,'History metadata axis '+key)
    history_dates=history_meta['dates']
    require(history_dates==sorted(set(history_dates)),'History dates must be increasing and unique')
    keep_history=[i for i,day in enumerate(history_dates) if n and day<=qm['dates'][-1]]
    require(keep_history==list(range(len(keep_history))),'History must be an initial chronological prefix')
    h=len(keep_history)
    for key in ('dates','ends','symbols'):hm[key]=[list(v) if isinstance(v,list) else v for v in history_meta[key][:h]]
    for key,value in source.items():require(np.asarray(value).shape==(history_rows*width,),'Source history axis '+key)
    smallsource={key:np.asarray(value)[:h*width].copy() for key,value in source.items()}
    months=len(training_meta['dates']);tm=dict(training_meta)
    for key in ('dates','ends','symbols'):require(isinstance(training_meta.get(key),list) and len(training_meta[key])==months,'Training metadata axis '+key)
    widths=np.array([len(row) for row in training_meta['symbols']],dtype=int);offsets=np.r_[0,np.cumsum(widths)]
    require(np.asarray(train['offsets']).dtype.kind in 'iu' and np.array_equal(train['offsets'],offsets),'Training offsets differ from identity metadata')
    for key in TRAIN_KEYS:require(np.asarray(train[key]).shape[0]==int(offsets[-1]),'Training row axis '+key)
    dates=np.asarray(training_meta['dates'],dtype='datetime64[D]');ends=np.asarray(training_meta['ends'],dtype='datetime64[D]')
    require(not np.isnat(dates).any() and not np.isnat(ends).any() and np.all(ends>dates) and
        (len(dates)<2 or np.all(dates[1:]>dates[:-1])),'Training metadata calendar')
    for key,values in (('dates',dates),('ends',ends)):
        require(np.array_equal(np.asarray(train[key],dtype='datetime64[D]'),np.repeat(values,widths)),'Training flat date identity '+key)
    keep=np.zeros(months,bool)
    if n:keep=ends<=np.datetime64('%04d-12-31'%(int(qm['dates'][-1][:4])-1),'D')
    ids=np.flatnonzero(keep);flat=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids]) if len(ids) else np.array([],dtype=int)
    for key in ('dates','ends','symbols','pool_origins'):
        if key in training_meta:
            require(len(training_meta[key])==months,'Training metadata field axis '+key)
            tm[key]=[list(training_meta[key][i]) if isinstance(training_meta[key][i],list) else training_meta[key][i] for i in ids]
    smalltrain={key:np.asarray(train[key])[flat].copy() for key in TRAIN_KEYS};smalltrain['offsets']=np.r_[0,np.cumsum(widths[ids])]
    cm=current[:n].copy();require(cm.dtype.kind=='b','Current availability must be exact boolean')
    return (smalltrain,tm,qm,hm,smallsource,use[:h].copy()),cm


def require_model_audit(check,built,rows,history_rows=None):
    for key in MODEL_FLAGS:require(check.get(key) is True,'Independent model evidence missing '+key)
    require(check.get('query_rows')==rows and check.get('numerical_tolerance')==0. and
        check.get('gram_relative_tolerance')==1e-12 and check.get('producer_modules_imported') is False and
        check.get('computes_returns') is False,'Independent model scope/coverage differs')
    require(set(built)=={'arrays','audit'},'Build mapping differs')
    a=built['audit'];v=built['arrays'];counts=a['counts'];events=a['event_history']
    require(a.get('separate_history_axis') is True and type(a.get('history_months')) is int and a['history_months']>=0,
            'Separate history axis audit missing')
    require(check.get('history_rows')==a['history_months'] and
            (history_rows is None or a['history_months']==history_rows),'Independent history coverage differs')
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
    import monthly_separate_history230_model as model
    import monthly_separate_history230_independent as independent
    study=local();m,a,px,op=study.load()
    require(len(m['dates'])==122 and a['baseline'].shape==(122,49) and len(a)==27,'Original122x49/27 arrays required')
    train=ev.load_arrays(REBUILD/'TRAIN_INPUTS.npz');metadata=read(REBUILD/'MONTHLY_META.json')
    require(metadata['monthly']==m and metadata['training']==read(study.arch.P/'META.json')['training'],'Original row identity changed')
    source=ev.load_arrays(SOURCE/'prepared/FORECASTS.npz')
    require(read(SOURCE/'prepared/META.json')==m,'Cached206 query identity differs')
    cached_check=require_cached_source(source,a)
    def snapshot(values):return {key:(v.dtype.str,v.shape,v.tobytes()) for key,v in values.items()}
    expanded=ev.load_arrays(EXPANDED/'prepared/FORECASTS.npz')
    labels=ev.load_arrays(EXPANDED/'prepared/LABELS.npz')
    expanded_meta=read(EXPANDED/'prepared/META.json')
    early=ev.load_arrays(EARLY/'prepared/FORECASTS.npz');early_labels=ev.load_arrays(EARLY/'prepared/LABELS.npz')
    early_meta=read(EARLY/'prepared/META.json')
    before=[snapshot(v) for v in (a,train,source,expanded,labels,early,early_labels)]
    import monthly_expanded_error228_adapter as adapter
    import monthly_expanded_error228_independent as adapter_independent
    original_use=original_history_use(a,122)
    adapter_args=(train,metadata['training'],m,source,expanded,labels,original_use)
    adapted=adapter.adapt(*adapter_args,source_meta227=expanded_meta)
    adaptation_check=adapter_independent.check_adapter(*adapter_args,adapted,source_meta227=expanded_meta)
    adapter_independent.require_receipt(adaptation_check,adapted['audit'])
    joined=join_history(metadata['training'],m,early_meta,early,early_labels,labels,adapted)
    require(joined['audit']['early_months']==67 and joined['audit']['history_months']==189 and
        joined['audit']['early_rows']==3283 and joined['audit']['current_months']==122,'Full67+122 source axes required')
    risk_train=adapted['train'];history_meta=joined['history_meta'];risk_source=joined['history_source']
    use=joined['historical_use'];current_available=joined['current_available']
    pre=study.modified_pre_score(study.configs()[4],a,122)
    eligible=slice_arrays(a,122)['eligible']
    base,current=prefix_inputs(risk_train,metadata['training'],m,history_meta,risk_source,use,current_available,122)
    inputs=base+(pre,eligible)
    built=model.build(*inputs,current_available=current)
    check=independent.check_build(*inputs,built,current_available=current);require_model_audit(check,built,122,189)
    orders,scores,choices=decisions(m,a,built,122,study=study)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    selection=independent.verify_selection(a,m,built,scores,choices);require_selection_audit(selection,122)
    prefix_evidence=[]
    for n in MODEL_PREFIXES:
        smallbase,smallcurrent=prefix_inputs(risk_train,metadata['training'],m,history_meta,risk_source,use,current_available,n)
        smallinputs=smallbase+(pre[:n].copy(),eligible[:n].copy())
        small=model.build(*smallinputs,current_available=smallcurrent)
        smallcheck=independent.check_build(*smallinputs,small,current_available=smallcurrent)
        require_model_audit(smallcheck,small,n,len(smallinputs[3]['dates']))
        require(small['audit']['event_history']==built['audit']['event_history'][:n],'Physical model audit prefix differs')
        for key in built['arrays']:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
        aa=slice_arrays(a,n);require(len(aa)==27 and all(value.shape[0]==n for value in aa.values()),'All27 physical cache axes required')
        oo,ss,cc=decisions(smallinputs[2],aa,small,n,study=study)
        smallselection=independent.verify_selection(aa,smallinputs[2],small,ss,cc);require_selection_audit(smallselection,n)
        require(all(oo[key]==orders[key][:n] and cc[key]==choices[key][:n] for key in NAMES),'Physical prefix order parity failed')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
        prefix_evidence.append(dict(prefix=n,training_rows=len(smallinputs[0]['dates']),query_rows=n,
            history_rows=len(smallinputs[3]['dates']),history_last_date=smallinputs[3]['dates'][-1] if smallinputs[3]['dates'] else None,
            current_available_rows=int(smallcurrent.sum()),
            query_date=m['dates'][n-1] if n else None,cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(smallinputs[0]['ends'],dtype='datetime64[D]').max()) if len(smallinputs[0]['ends']) else None,
            source_history_starts_at_original_zero=True,counts=small['audit']['counts'],model=smallcheck,selection=smallselection))
    for n in range(123):
        oo,ss,cc=decisions(m,a,built,n,study=study)
        require(all(oo[key]==orders[key][:n] and cc[key]==choices[key][:n] for key in NAMES),'Order prefix parity failed')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
    require(before==[snapshot(v) for v in (a,train,source,expanded,labels,early,early_labels)],'Original or227 input arrays modified')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'RISKS_EFFECTS.npz',**built['arrays'])
    np.savez_compressed(dest/'SCORES.npz',**scores)
    dump(dest/'MODEL_AUDIT.json',built['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',m)
    parity=dict(original_training_cache_exact=True,original_identity_exact=True,
        raw_source_hashes=plan['raw_source_hashes'],model_source_binding=plan['model_source_binding'],
        original_training_cache_exact_basis='Pinned TRAIN_INPUTS bytes and saved169/206 parity certificates reused; not a fresh original-array reconstruction',
        fresh_training_array_comparison=False,saved_training_parity_sha256=sha(REBUILD/'PARITY.json'),
        cached206_source_vectors_exact=True,cached206_control_constant_exact=True,source_availability_exact=True,
        error_prediction_source='raw_local_linear',cached_source_check=cached_check,historical_expansion_contract_checked=True,
        target_source='source227_original_center_extended_log_excess',
        no_source_model_refits=True,no_fitted_residual_error_inputs=True,new_labels=False,new_raw_features=False,
        risk_history_adapter=adapted['audit'],independent_adapter=adaptation_check,
        history_join=joined['audit'],history_meta=history_meta,source229_binding=plan['model_source_binding']['early_source_binding'],
        historical_mask_semantics='early229.available then originalU OR newlyavailable227; historical_original_universe_checked verifies the supplied mask',
        original_ridge_training_unchanged=True,current_trading_eligibility_unchanged=True)
    dump(dest/'SOURCE_PARITY.json',parity)
    evidence=dict(model=check,selection=selection,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        model_prefix_sources_truncated=True,model_prefix_metadata_truncated=True,model_prefix_original_arrays_truncated=True,
        model_prefix_future_labels_removed=True,source_history_starts_at_original_zero=True,model_prefix_evidence=prefix_evidence,
        original_orders_exact=True,original_arrays_unchanged=True,original_source_arrays_unchanged=True,original_array_count=27,
        original_core_metadata_and_support_preserved=True,single_original_episode=True,
        no_new_score_budget=True,historical_expansion_contract_checked=True,error_prediction_source_local_linear=True,separate_history_axis=True,
        source229_labels_exact_copy=True,current_available_counts_separate=True,model_counts=built['audit']['counts'])
    dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    completion=dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,
        original_array_count=27,model_counts=built['audit']['counts'],source_model_refits=0,
        model_fit_accounting=contract()['model_fit_accounting'])
    verify_prepared_evidence(completion,m,evidence,parity,built)
    dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def verify_prepared_evidence(c,meta,audit,parity,built):
    import monthly_expanded_error228_independent as adapter_independent
    adapter_independent.require_receipt(parity.get('independent_adapter'),parity.get('risk_history_adapter'))
    require(parity.get('original_ridge_training_unchanged') is True and parity.get('current_trading_eligibility_unchanged') is True,'Expanded risk-only source scope missing')
    require(parity.get('historical_mask_semantics')=='early229.available then originalU OR newlyavailable227; historical_original_universe_checked verifies the supplied mask','Expanded history provenance semantics changed')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(v)==49 for v in meta['symbols']),'Full query metadata required')
    hm=parity.get('history_meta');join=parity.get('history_join')
    require(isinstance(hm,dict) and isinstance(join,dict),'Separate source join proof missing')
    for key in ('dates','ends','symbols'):
        require(isinstance(hm.get(key),list) and len(hm[key])==189 and hm[key][67:]==meta[key],'Combined source/current metadata differs')
    require(join.get('early_months')==67 and join.get('current_months')==122 and join.get('history_months')==189 and
        join.get('width')==49 and join.get('early_rows')==3283 and join.get('current_rows')==5978 and join.get('history_rows')==9261,'Combined source dimensions differ')
    for key in ('exact_early_training_prefix','exact_early_label_copy','early_history_use_equals_available',
        'current_adapted_history_unchanged','current_query_axis_unchanged','current_available_axis_separate'):
        require(join.get(key) is True,'Source join evidence missing '+key)
    for key in ('labels_appended','source_forecasts_refitted','labels_recomputed'):require(join.get(key) is False,'Source join scope '+key)
    require(join.get('history_use_rule')=='early229.available followed by originalU OR newlyavailable227','Combined historical mask differs')
    require(join['history_use_rows']==join['early_history_use_rows']+join['current_history_use_rows'] and
        join['early_history_use_rows']==join['early_available_rows'],'Combined historical count identity')
    require_model_audit(audit['model'],built,122,189);require_selection_audit(audit['selection'],122)
    require(built['audit']['counts']['cached_forecast_rows']==join['current_available_rows'],'Current availability count uses wrong source axis')
    require(audit.get('prefixes')==123 and audit.get('model_prefixes')==list(MODEL_PREFIXES),'Missing prefix coverage')
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged',
        'original_source_arrays_unchanged','original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
        'historical_expansion_contract_checked','error_prediction_source_local_linear','separate_history_axis',
        'source229_labels_exact_copy','current_available_counts_separate'):
        require(audit.get(key) is True,'Prepared invariant missing: '+key)
    require(audit.get('original_array_count')==c.get('original_array_count')==27,'Original array count differs')
    require(c.get('new_returns_computed') is False and type(c.get('source_model_refits')) is int and c['source_model_refits']==0,'Preparation scope differs')
    require(audit.get('model_counts')==c.get('model_counts')==built['audit']['counts'],'Main-pass statistical accounting differs')
    evidence=audit.get('model_prefix_evidence')
    require(isinstance(evidence,list) and [row.get('prefix') for row in evidence]==list(MODEL_PREFIXES),'Model prefix evidence missing')
    for row in evidence:
        n=row['prefix'];require(row.get('query_rows')==n and type(row.get('training_rows')) is int and row['training_rows']>=0,'Model prefix coverage differs')
        require(row.get('source_history_starts_at_original_zero') is True,'Source prefix starts late')
        expected_h=sum(day<=meta['dates'][n-1] for day in hm['dates']) if n else 0
        require(row.get('history_rows')==expected_h and row.get('history_last_date')==(hm['dates'][expected_h-1] if expected_h else None),
            'Physical history was not sliced by current signal date')
        require_model_audit(row['model'],slice_built(built,n,history_months=expected_h),n,expected_h);require_selection_audit(row['selection'],n)
        require(row.get('current_available_rows')==row['model']['counts']['cached_forecast_rows'],'Prefix current availability count differs')
        require(row.get('counts')==row['model']['counts'],'Prefix statistical count differs')
        if n:
            day=meta['dates'][n-1];cutoff='%04d-12-31'%(int(day[:4])-1)
            require(row.get('query_date')==day and row.get('cutoff')==cutoff and
                (row['latest_label_end'] is None or row['latest_label_end']<=cutoff),'Physical prefix maturity clock differs')
        else:require(row['training_rows']==0 and all(row.get(k) is None for k in ('latest_label_end','query_date','cutoff')),'Empty prefix contains source rows')
    for key in ('original_training_cache_exact','original_identity_exact','cached206_source_vectors_exact',
        'cached206_control_constant_exact','source_availability_exact','historical_expansion_contract_checked','no_source_model_refits','no_fitted_residual_error_inputs'):
        require(parity.get(key) is True,'Source parity missing '+key)
    require(parity.get('fresh_training_array_comparison') is False and parity.get('new_labels') is False and
        parity.get('new_raw_features') is False and parity.get('error_prediction_source')=='raw_local_linear' and
        parity.get('target_source')=='source227_original_center_extended_log_excess','Source parity scope differs')



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
    early=read(EARLY/'prepared/META.json')['history']
    require(parity['history_meta']=={key:early[key]+m[key] for key in ('dates','ends','symbols')} and
        parity['source229_binding']==plan['model_source_binding']['early_source_binding'],'Stored early source axis differs from registration')
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
        next_step='Review separate early229-history joint versus diagonal prediction-error results and support; keep frozen observers unchanged.')
    require(len(state['completed']) == 226 and sum(r['trials'] for r in state['completed']) == 1195, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
