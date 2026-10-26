"""Round231 price-only current-market drift, with one full-eligible selection hook.
All actual features, selections and returns require the immutable PLAN gate.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
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
OUT = ROOT/'outputs/monthly_market_drift231'
LOCAL = ROOT/'outputs/monthly_local_linear70'
PRIOR = ROOT/'outputs/monthly_separate_history230'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67','market_conditioned_drift','unconditional_drift_control')
RAW = {NAMES[1]:'candidate',NAMES[2]:'control'}
TRAIN = 88
MODEL_PREFIXES = FEATURE_PREFIXES = (0,1,53,88,122)
CONTEXT = ROOT/'work/monthly_market_drift231_context.json'
CONTEXT_SHA = 'bfa2fc65db4a3e1a50cda8979eef65e513049f196a70cc0647864e0d50b66de9'
PROPOSAL = ROOT/'work/market_drift231_PROPOSAL.json'
PROPOSAL_SHA = '46556aa1b5d6176c6960ae08fbfe5e1190b7042c15d8f321eaa4d1f7e6463b73'
CLARIFICATIONS = ROOT/'work/market_drift231_CLARIFICATIONS.json'
CLARIFICATIONS_SHA = '40675e741285248f15abc47114ba4105268451d6b1f9eb883af7f35b271dbf9c'
SOURCE_REVIEW = ROOT/'work/market_drift231_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = '908ecdab164621bb7dd4571654cc5e05dcb87511c626b78f9a691ab20a43539a'
TEST_RECEIPT = ROOT/'work/monthly_market_drift231_TESTS.json'
EXTERNAL_RUNTIME = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED = {'PLAN.json':'f8e0c1baceee92b2975117825ab077909b4cdabe0c7ab02fb5f59a9dc50e8b5c',
 'COMPLETION.json':'ea29a346fccc2a6fdc05696377d3cb6c487f557a7d87c0e92e420db81f14b23f',
 'results/COMPLETION.json':'7a0d3d49bf1bb8407bb7a971cb03aacac507a87ef8d7a4b2c632d3d933907966',
 'RESEARCH_REVIEW.json':'548e0bad94a450a7790efa26f4016b7dd69e64b975f1a0fc4da966c7018146ef'}
MODEL_FLAGS = ('passed','independent_scalar_rebuild','original_calm_window_checked',
 'official_session_windows_checked','skip42_stock_window_checked','current_market_state_checked',
 'previous_day_market_states_checked','strict_spy200_comparison_checked','mean_log_drift_checked',
 'same_window_unconditional_control_checked','fixed_63_shrinkage_checked','minimum_42_matches_checked',
 'complete_eligible_universe_checked','whole_month_common_fallback_checked','symmetric_eligible_effect_ranks_checked',
 'constant_drift_zero_effect_checked','initial_partial_unchanged','no_forward_labels','no_future_inputs',
 'audit_checked','counts_checked','no_predictive_fits_checked')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_support_checked',
 'original_budget_checked','eligible_universe_reranked_checked','replacement_not_additive_checked',
 'one_episode_per_arm','paired_fallback_checked','constant_effect_not_fallback_checked','exact_scores_and_choices_checked')
MODEL_COUNT_FIELDS = ('query_months','eligible_cells','supported_months','fallback_months',
 'initial_partial_months','domain_fallback_months','source_fallback_months','state_support_fallback_months',
 'checked_stock_cells','source_valid_stock_cells','stock_price_observations','stock_log_returns',
 'drift_estimates','ranked_observations','market_state_evaluations','market_price_observations')
ARRAY_KEYS = ('candidate_raw','control_raw','candidate_effect','control_effect','ready')
PREPARED_FILES = {'FEATURES.npz','SCORES.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
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
    backup = CENTRAL.parent / ('PRE_MONTHLY_MARKET_DRIFT231_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/market_drift231_CODE_REVIEW.json'
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
    require(p['price_source_binding'] == price_source_binding(p['raw_source_hashes']), 'Registered model source binding changed')
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
        open_vs_incumbent=guard(metrics[NAMES[1]]['next_open_zero'], metrics[NAMES[0]]['next_open_zero'], 0.),
        close_vs_unconditional_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_unconditional_control=guard(metrics[NAMES[1]]['next_open_zero'], metrics[NAMES[2]]['next_open_zero'], 0.),
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
        for path, mode, cost in (ev.PATHS[0], ev.PATHS[1]):
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
    import monthly_market_drift231_independent as independent
    helper = independent.accounting()
    require(set(training) == set(NAMES), 'Training arms differ')
    for arm in NAMES:
        require(set(training[arm]) == {'same_close_zero', 'next_open_zero'}, 'Training paths differ')
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
    import monthly_market_drift231_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, market_conditioned_drift=True, current_market_state=True, full_eligible_single_effect_replacement=True, zero_cost_selection=True, next_open_cost_diagnostic_only=True, same_support_both_arms=True, observer_promotion=False))
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

def tests():
    require(not TEST_RECEIPT.exists(),'Formal synthetic test receipt is write-once')
    names=('test_monthly_market_drift231_host.py','test_monthly_market_drift231_model.py',
        'test_monthly_market_drift231_independent.py')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module('market231test'+str(i),ROOT/'work'/name))
                             for i,name in enumerate(names)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic tests failed')
    dump(TEST_RECEIPT,dict(passed=True,tests=result.testsRun,synthetic_only=True,runtime=ev.runtime(),
        code_sha256={str(p):sha(p) for p in code_files()}))

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

def require_selection_audit(check,rows):
    for key in SELECTION_FLAGS:require(check.get(key) is True,'Independent selection evidence missing '+key)
    require(check.get('ordered_choices_checked')==3*rows and check.get('numerical_tolerance')==0. and
        check.get('original_source_parity_tolerance')==1e-12 and check.get('producer_modules_imported') is False and
        check.get('computes_returns') is False,'Independent selection scope differs')

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
        next_step='Review current-market conditional drift against unconditional drift on zero-cost selection paths; keep frozen observers unchanged.')
    require(len(state['completed']) == 227 and sum(r['trials'] for r in state['completed']) == 1198, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

def contract():
    for path,digest in ((CONTEXT,CONTEXT_SHA),(PROPOSAL,PROPOSAL_SHA),(CLARIFICATIONS,CLARIFICATIONS_SHA)):
        require(sha(path)==digest,'Prospective231 definition changed: '+str(path))
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,proposal_sha256=PROPOSAL_SHA,
        clarifications_sha256=CLARIFICATIONS_SHA,arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],
        period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,
        model_prefixes=list(MODEL_PREFIXES),physical_model_prefixes=True,all123_order_prefixes=True,
        formation_returns_calm=210,formation_returns_other=252,skip_stock_sessions=42,
        current_market_state=True,historical_state_previous_day=True,spy_mean_sessions=200,
        minimum_state_matches=42,shrinkage_days=63,whole_month_common_fallback=True,
        constant_supported_zero_effect=True,full_eligible_reranking=True,replace_single_episode_hook=True,
        original_core_and_support_preserved=True,no_new_score_budget=True,
        selection_paths=['same_close_zero','next_open_zero'],next_open_cost_diagnostic_only=True,
        model_fit_accounting=context['accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 zero-cost paths select; full-period best is descriptive among training-selected candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,new_model_fits=0,predictive_model_fits=0,
        no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)


def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_market_drift231_host.py','test_monthly_market_drift231_host.py',
        'monthly_market_drift231_model.py','test_monthly_market_drift231_model.py',
        'monthly_market_drift231_independent.py','test_monthly_market_drift231_independent.py',
        'monthly_market_drift231_context.json','market_drift231_PROPOSAL.json','market_drift231_CLARIFICATIONS.json',
        'market_drift231_SOURCE_REVIEW.json','market_drift231_source_metadata.py',
        'test_monthly_joint_error226_independent.py','monthly_joint_error226_independent.py',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_market_drift231_model as model
    import monthly_market_drift231_independent as independent
    deps=set(model.dependency_files())|set(independent.dependency_files())
    return direct+sorted(deps-set(direct))


def state_check(state):
    require(len(state['completed'])==226 and sum(row['trials'] for row in state['completed'])==1195,
            'Expected226 completed entries /1195 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round230 must be completed predecessor')


def price_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Source metadata review changed')
    review=read(SOURCE_REVIEW)
    for key in ('passed','source_only','metadata_only','source_bytes_reverified','metadata_axes_verified',
        'calendar_clock_verified','raw_source_map_matches_original','core_calm_source_definition_reviewed',
        'single_episode_budget_source_reviewed','price_unit_source_fingerprints_bound'):
        require(review.get(key) is True,'Source admission missing '+key)
    for key in ('source_npz_decoded','npz_decoded','price_rows_read','source_models_imported','source_loader_executed',
        'actual_support_computed','actual_features_computed','actual_labels_computed','actual_forecasts_computed',
        'actual_returns_computed','actual_orders_computed','actual_new_parameters_computed','observers_modified',
        'central_modified','frozen_files_modified','strict_pit'):
        require(review.get(key) is False,'Source review scope differs '+key)
    require(type(review.get('actual_model_fits')) is int and review['actual_model_fits']==0,'Source review actual fits')
    require(review.get('context_sha256')==CONTEXT_SHA and review.get('proposal_sha256')==PROPOSAL_SHA and
        review.get('clarifications_sha256')==CLARIFICATIONS_SHA,'Source prospective definition differs')
    require(review.get('raw_source_count')==64 and review.get('raw_source_hashes')==raw and len(raw)==64,
        'Original64 raw source map differs')
    pins={};bind(pins,SOURCE_REVIEW,SOURCE_REVIEW_SHA)
    for field in ('input_sha256','source_sha256','evidence_sha256','direct_feature_source_sha256','reviewed_code_sha256'):
        mapping=review.get(field);require(isinstance(mapping,dict) and mapping,'Source evidence missing '+field)
        for name,digest in mapping.items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    context_evidence=read(CONTEXT)['evidence_sha256']
    for name,digest in context_evidence.items():bind(pins,ROOT/name,digest)
    for name,digest in raw.items():bind(pins,Path(name),digest)
    for name in ('PLAN.json','DATA.npz','META.json','INPUT_HASHES.json','CACHE_HASHES.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'):
        bind(pins,LOCAL/name)
    bind(pins,CALENDAR)
    for name,digest in read(CALENDAR)['provenance']['source_hashes'].items():bind(pins,Path(name),digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        context_evidence_sha256=context_evidence,review_evidence_sha256=review['evidence_sha256'],
        no_new_training_labels=True,no_new_historical_forecasts=True,strict_pit=False)


# Compatibility alias for synthetic source-binding tests; the PLAN uses price_source_binding.
model_source_binding=price_source_binding


def register():
    require(not OUT.exists(),'Fresh experiment directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    require(not state['registered_queue'],'Existing registered experiment')
    previous=predecessor_binding(state);validation=read(TEST_RECEIPT)
    require(validation['passed'] is True and validation['synthetic_only'] is True and validation['runtime']==ev.runtime(),'Synthetic tests required')
    require(validation['code_sha256']=={str(p):sha(p) for p in code_files()},'Code changed since tests')
    pins=dict(read(PRIOR/'PLAN.json')['input_sha256']);raw=read(LOCAL/'PLAN.json')['raw_source_hashes']
    source=price_source_binding(raw);review=code_review_binding()
    for binding in (previous,source,review):
        for name,digest in binding['input_sha256'].items():bind(pins,Path(name),digest)
    for path in code_files()+[TEST_RECEIPT,PROTECTED,CALENDAR,LOCAL/'PLAN.json']:bind(pins,path)
    for name in ('DATA.npz','META.json','INPUT_HASHES.json','CACHE_HASHES.json','local_linear_ORDERS.json','INCUMBENT_POLICY.json'):
        bind(pins,LOCAL/name)
    for name,digest in raw.items():bind(pins,Path(name),digest)
    for name,digest in ev.ENGINE_HASHES.items():bind(pins,ev.ENGINE_DIR/name,digest)
    verify(pins);protected=read(PROTECTED);verify(protected)
    require(len(protected)==21444,'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as stream:stream.write(before)
    plan=dict(schema='monthly_market_drift231_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        price_source_binding=source,numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_parameters_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_conditional_market_drift',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare fixed market-conditioned and unconditional drift; freeze first88 zero-cost selection before later34 diagnostics.')
    commit_state(before,state,'register')
    print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)


def slice_built(built,n):
    require(set(built)=={'arrays','audit'} and set(built['arrays'])==set(ARRAY_KEYS),'Five-array build required')
    v=built['arrays'];shape=np.asarray(v['candidate_raw']).shape
    require(len(shape)==2 and type(n) is int and 0<=n<=shape[0],'Invalid build prefix')
    audit=dict(built['audit']);events=audit['event_history'][:n]
    audit.update(events=n,event_history=events,counts={key:sum(e['counts'][key] for e in events) for key in MODEL_COUNT_FIELDS},
        reason_counts=dict(sorted(Counter(e['status'] for e in events).items())))
    return dict(arrays={key:np.asarray(value)[:n].copy() for key,value in v.items()},audit=audit)


def prefix_inputs(panel,meta,calendar,arrays,n):
    a=slice_arrays(arrays,n);m=dict(meta)
    for key in ('dates','ends','symbols'):m[key]=meta[key][:n]
    cutoff=m['dates'][-1] if n else None
    p={'adj':panel['adj'].loc[:cutoff].copy() if n else panel['adj'].iloc[:0].copy()}
    # Official calendar is static metadata, including the next session proving month-end.
    return p,m,calendar,a


def _full_selector(study,ready):
    """Detached replacement hook: old boundary is called ONLY on fallback rows."""
    def select(pre,eligible,symbols,effects,original,policy):
        score=np.asarray(pre).copy();n,width=score.shape
        require(ready.shape==(n,) and ready.dtype==bool,'Selector readiness axis')
        choices=[None]*n
        fallback=np.flatnonzero(~ready)
        if len(fallback):
            original_part={k:np.asarray(v)[fallback].copy() for k,v in original.items()}
            old_choices,old_scores=study.ep.select_boundary(score[fallback].copy(),eligible[fallback].copy(),
                [symbols[i] for i in fallback],effects[fallback].copy(),original_part,dict(policy))
            for j,i in enumerate(fallback):choices[i]=old_choices[j];score[i]=old_scores[j]
        support=(original['baseline']-original['core'])/(1-original['core_weight'])
        calm=np.asarray(original['state_calm10']);weak=np.asarray(original['state_weak_narrow'])
        budget=np.minimum(.05,np.maximum(0.,.10-(.025*calm+.05*weak)))
        for i in np.flatnonzero(ready):
            ids=np.flatnonzero(eligible[i]).tolist()
            require(len(ids)>=2,'Supported selector requires at least2 eligible names')
            for j in ids:
                target=min(.5,max(-.5,support[i,j]+effects[i,j]))
                score[i,j]+=budget[i]*(target-support[i,j])
            choices[i]=sorted(ids,key=lambda j:(-score[i,j],symbols[i][j]))[:3]
        return choices,score
    return select


def decisions(meta,arrays,built,n,study=None):
    study=local() if study is None else study;a=slice_arrays(arrays,n);b=slice_built(built,n)
    m=dict(meta)
    for key in ('dates','ends','symbols'):m[key]=meta[key][:n]
    require(len(m['dates'])==n,'Decision metadata prefix differs')
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choice,score=study.choices(study.configs()[4],a,m,n,policy=policy)
    choices={NAMES[0]:choice};scores={NAMES[0]:score}
    pre=study.modified_pre_score(study.configs()[4],a,n)
    ready=np.asarray(b['arrays']['ready']);require(ready.dtype==bool and ready.shape==(n,),'Ready mask differs')
    require(not any(ready[i] and m['dates'][i]=='2016-08-17' for i in range(n)),'Initial partial must fallback')
    for name,key in RAW.items():
        effects=a['effects'].copy();new=b['arrays'][key+'_effect']
        require(new.shape==pre.shape and new.dtype==np.float64 and np.isfinite(new).all() and (np.abs(new)<=.5).all(),
            'Bounded finite effect matrix required')
        require(not new[~(ready[:,None]&a['eligible'])].any(),'Effects outside supported eligible domain')
        effects[ready]=new[ready]
        choice,score=study.adapter.score_and_choose('incumbent66',0.,np.full(pre.shape,np.nan),a,m,n,
            pre_score=pre.copy(),effects=effects,episode_policy=policy,boundary_selector=_full_selector(study,ready))
        np.testing.assert_array_equal(score[~ready],scores[NAMES[0]][~ready])
        require(all(choice[i]==choices[NAMES[0]][i] for i in np.flatnonzero(~ready)),'Whole-month fallback differs')
        choices[name]=choice;scores[name]=score
    return {name:study.rp.make_orders(choices[name],scores[name],m) for name in NAMES},scores,choices


def calm_parity(panel,meta,calendar,arrays):
    """Independent host clock check of cached calm; no changes to cached states."""
    sessions=pd.DatetimeIndex(calendar);dates=meta['dates'];n=len(dates)
    require(sessions.is_unique and sessions.is_monotonic_increasing and 'SPY' in panel['adj'],'Official SPY calendar required')
    cached=np.asarray(arrays['state_calm10']);require(cached.dtype==bool and cached.shape in ((n,),(n,1)),'Original calm boolean shape')
    cached=cached.reshape(n);adj=panel['adj']
    # Compute the original pandas rolling specification on the official axis.
    spy=adj['SPY'].reindex(sessions)
    returns=spy.pct_change(fill_method=None)
    volatility=returns.rolling(63).std()*np.sqrt(252.)
    records=[]
    for i,day in enumerate(dates):
        t=sessions.get_loc(pd.Timestamp(day));require(type(t) in (int,np.int64) and t>=63,'Insufficient calm clock')
        window=sessions[t-63:t+1]
        require(not len(window.difference(adj.index)),'Missing official calm session')
        values=spy.loc[window].to_numpy()
        require(all(isinstance(v,(float,int,np.floating,np.integer)) and not isinstance(v,(bool,np.bool_)) for v in values)
                and np.isfinite(values).all() and (values>0).all(),'Invalid calm source prices')
        vol=float(volatility.loc[day]);require(np.isfinite(vol),'Invalid current63 sample volatility')
        actual=bool(vol<.10);require(actual==bool(cached[i]),'Cached calm10 differs at '+day)
        records.append(dict(date=day,price_start=str(window[0].date()),price_end=str(window[-1].date()),
            sample_returns=63,annualized_sample_volatility=vol,calm=actual))
    return dict(passed=True,official_calendar=True,exact_cached_calm=True,query_rows=n,records=records)


def require_model_audit(check,built,rows):
    for key in MODEL_FLAGS:require(check.get(key) is True,'Independent model evidence missing '+key)
    require(check.get('query_rows')==rows and check.get('numerical_tolerance')==0. and
        check.get('producer_modules_imported') is False and check.get('computes_returns') is False,'Independent model scope differs')
    require(set(built)=={'arrays','audit'},'Build schema differs')
    a=built['audit'];v=built['arrays'];c=a['counts'];events=a['event_history'];width=a['symbols_per_event']
    require(a.get('events')==rows and a.get('method')=='current_market_conditioned_skip42_daily_drift' and
        a.get('arms')==list(NAMES) and a.get('skip_sessions')==42 and a.get('calm_returns')==210 and
        a.get('other_returns')==252 and a.get('market_symbol')=='SPY' and a.get('market_mean_sessions')==200 and
        a.get('min_matching_days')==42 and a.get('shrinkage_days')==63,'Fixed drift protocol differs')
    require(type(width) is int and width>=0 and (rows==0 or width>0),'Invalid width')
    require(set(v)==set(ARRAY_KEYS) and set(c)==set(MODEL_COUNT_FIELDS) and
        all(type(x) is int and x>=0 for x in c.values()),'Array/count schema differs')
    require(isinstance(events,list) and len(events)==rows,'Event history differs')
    ready=np.asarray(v['ready']);require(ready.dtype==bool and ready.shape==(rows,),'Ready axis differs')
    mask=np.zeros((rows,width),bool)
    for i,e in enumerate(events):
        ec=e['counts'];ids=e['eligible_indices'];cells=e['cells'];L=210 if e['calm'] else 252
        require(set(ec)==set(MODEL_COUNT_FIELDS) and all(type(x) is int and x>=0 for x in ec.values()),'Event count schema')
        require(e['row']==i and len(e['symbols'])==width and len(set(e['symbols']))==width and
            len(ids)==len(set(ids)) and ids==sorted(ids) and all(type(j) is int and 0<=j<width for j in ids),'Event axes differ')
        require(e['formation_returns']==L and len(cells)==len(ids) and
            [cell['column'] for cell in cells]==ids and [cell['symbol'] for cell in cells]==[e['symbols'][j] for j in ids],
            'Eligible cell identity differs')
        supported=e['status']=='supported';require(type(e['ready']) is bool and e['ready']==supported==bool(ready[i]),'Ready/status differs')
        require(ec['query_months']==1 and ec['eligible_cells']==len(ids) and ec['supported_months']==int(supported) and
            ec['fallback_months']==int(not supported),'Event support accounting')
        require(ec['initial_partial_months']==int(e['status']=='initial_partial') and
            ec['domain_fallback_months']==int(e['status']=='fewer_than_two_eligible') and
            ec['state_support_fallback_months']==int(e['status']=='insufficient_matching_state') and
            ec['source_fallback_months']==int(e['status'] not in ('initial_partial','fewer_than_two_eligible','insufficient_matching_state','supported')),
            'Fallback classification differs')
        checked=sum(cell['source_checked'] is True for cell in cells);valid=sum(cell['source_available'] is True for cell in cells)
        require(ec['checked_stock_cells']==checked and ec['source_valid_stock_cells']==valid and valid<=checked<=len(ids) and
            ec['stock_price_observations']==(L+1)*valid,'Source consumption counts differ')
        market_ok=e['query_state_up'] is not None
        require(ec['market_state_evaluations']==(L+1 if market_ok else 0) and
            ec['market_price_observations']==200*ec['market_state_evaluations'],'Market repeated consumption differs')
        require(ec['stock_log_returns']==(L*len(ids) if supported else 0) and
            ec['drift_estimates']==ec['ranked_observations']==(2*len(ids) if supported else 0),'Drift/rank accounting differs')
        if supported:
            require(len(ids)>=2 and valid==len(ids) and e['matching_state_days']>=42 and e['date']!='2016-08-17','Support contract differs')
            mask[i,ids]=True
        for cell in cells:
            j=cell['column']
            for arm in ('candidate','control'):
                value=cell[arm+'_raw']
                require((isinstance(value,(float,int)) and np.isfinite(value)) if supported else value is None,'Cell raw support differs')
                if supported:require(float(v[arm+'_raw'][i,j])==value,'Raw audit differs')
                require(float(v[arm+'_effect'][i,j])==cell[arm+'_effect'],'Effect audit differs')
    require(c=={key:sum(e['counts'][key] for e in events) for key in MODEL_COUNT_FIELDS} and c['query_months']==rows and
        c['supported_months']+c['fallback_months']==rows and
        sum(c[k] for k in ('initial_partial_months','domain_fallback_months','source_fallback_months','state_support_fallback_months'))==c['fallback_months'],
        'Aggregate counts differ')
    require(check.get('counts')==c and a['reason_counts']==dict(sorted(Counter(e['status'] for e in events).items())),'Independent counts/reasons differ')
    for key in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):
        require(type(a.get(key)) is int and type(check.get(key)) is int and a[key]==check[key]==0,'No predictive fitting allowed')
    for arm in ('candidate','control'):
        raw=np.asarray(v[arm+'_raw']);effect=np.asarray(v[arm+'_effect'])
        require(raw.dtype==effect.dtype==np.float64 and raw.shape==effect.shape==(rows,width) and
            not np.isinf(raw).any() and np.array_equal(np.isfinite(raw),mask),'Raw common support differs')
        require(np.isfinite(effect).all() and (np.abs(effect)<=.5).all() and not effect[~mask].any(),'Effect support differs')


def array_snapshot(arrays):
    return {key:(value.dtype.str,value.shape,value.tobytes()) for key,value in arrays.items()}


def array_fingerprints(arrays):
    return {key:dict(dtype=value.dtype.str,shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in arrays.items()}


def require_calm_audit(check,meta,model_audit):
    n=len(meta['dates'])
    for key in ('passed','official_calendar','exact_cached_calm'):
        require(check.get(key) is True,'Original calm evidence missing '+key)
    require(check.get('query_rows')==n and len(check.get('records',[]))==n,'Calm evidence coverage differs')
    for i,row in enumerate(check['records']):
        require(row['date']==meta['dates'][i] and row['price_end']==row['date'] and
            row['price_start']<row['price_end'] and row['sample_returns']==63,'Calm source clock differs')
        vol=row['annualized_sample_volatility']
        require(isinstance(vol,(float,int)) and not isinstance(vol,bool) and np.isfinite(vol) and vol>=0 and
            type(row['calm']) is bool and row['calm']==bool(vol<.10) and
            row['calm']==model_audit['event_history'][i]['calm'],'Calm value or model state differs')


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_market_drift231_model as model
    import monthly_market_drift231_independent as independent
    study=local();meta,arrays,px,op=study.load()
    require(meta==read(LOCAL/'META.json'),'Original metadata differs')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and
        arrays['baseline'].shape==(122,49) and len(arrays)==27 and
        all(len(row)==49 and len(set(row))==49 for row in meta['symbols']),'Original122x49/27 arrays required')
    before=array_snapshot(arrays)
    source,info=study.arch.old.panel_load(meta['etfs'])
    raw_hashes=study.raw_source_hashes(info)
    require(raw_hashes==plan['raw_source_hashes'],'Original raw source lineage changed');verify(raw_hashes)
    execution=study.verify_panel_parity(source,px,op)
    panel={'adj':source['adj']};calendar=read(CALENDAR)['sessions']
    calm=calm_parity(panel,meta,calendar,arrays)
    built=model.build(panel,meta,calendar,arrays)
    check=independent.check_build(panel,meta,calendar,arrays,built)
    require_model_audit(check,built,122);require_calm_audit(calm,meta,built['audit'])
    orders,scores,choices=decisions(meta,arrays,built,122,study=study)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    selection=independent.verify_selection(arrays,meta,built,scores,choices);require_selection_audit(selection,122)
    prefix_evidence=[]
    for n in MODEL_PREFIXES:
        pp,mm,cc,aa=prefix_inputs(panel,meta,calendar,arrays,n)
        require(len(aa)==27 and all(value.shape[0]==n for value in aa.values()),'All27 physical cache axes required')
        small_calm=calm_parity(pp,mm,cc,aa)
        small=model.build(pp,mm,cc,aa)
        smallcheck=independent.check_build(pp,mm,cc,aa,small)
        require_model_audit(smallcheck,small,n);require_calm_audit(small_calm,mm,small['audit'])
        require(small['audit']==slice_built(built,n)['audit'],'Physical price/metadata model audit prefix differs')
        for key in ARRAY_KEYS:np.testing.assert_array_equal(small['arrays'][key],built['arrays'][key][:n])
        oo,ss,choices_small=decisions(mm,aa,small,n,study=study)
        smallselection=independent.verify_selection(aa,mm,small,ss,choices_small);require_selection_audit(smallselection,n)
        require(all(oo[key]==orders[key][:n] and choices_small[key]==choices[key][:n] for key in NAMES),'Physical prefix choices/orders differ')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
        prefix_evidence.append(dict(prefix=n,query_rows=n,query_date=mm['dates'][-1] if n else None,
            price_rows=len(pp['adj']),latest_price_date=str(pp['adj'].index[-1].date()) if len(pp['adj']) else None,
            original_array_count=len(aa),metadata_rows=len(mm['symbols']),counts=small['audit']['counts'],
            model=smallcheck,selection=smallselection,calm=small_calm))
    for n in range(123):
        oo,ss,choices_small=decisions(meta,arrays,built,n,study=study)
        require(all(oo[key]==orders[key][:n] and choices_small[key]==choices[key][:n] for key in NAMES),'All123 decision prefix parity failed')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
    require(before==array_snapshot(arrays),'Original27 arrays modified')
    parity=dict(raw_source_hashes=raw_hashes,price_source_binding=plan['price_source_binding'],
        feature_panel='original stored adjusted closes;210/252 skip42 formation, previous-day SPY200 states and current-query SPY200 state',
        execution_prices=execution,calm_parity=calm,calendar_sha256=sha(CALENDAR),
        original_metadata_identity_verified=True,original_array_fingerprints=array_fingerprints(arrays),
        original_arrays_unchanged=True,original_orders_exact=True,new_training_labels=False,
        new_historical_forecasts=False,strict_pit=False)
    evidence=dict(model=check,selection=selection,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        model_prefix_sources_truncated=True,model_prefix_metadata_truncated=True,model_prefix_original_arrays_truncated=True,
        model_prefix_independently_verified=True,model_prefix_evidence=prefix_evidence,
        original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,
        original_core_metadata_and_support_preserved=True,single_original_episode=True,no_new_score_budget=True,
        full_eligible_replacement=True,constant_effect_not_fallback=True,calm_cache_independently_verified=True,
        selection_uses_zero_cost_paths_only=True,model_counts=built['audit']['counts'])
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FEATURES.npz',**built['arrays'])
    np.savez_compressed(dest/'SCORES.npz',**scores)
    dump(dest/'MODEL_AUDIT.json',built['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',meta)
    dump(dest/'SOURCE_PARITY.json',parity);dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    completion=dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_returns_computed=False,
        original_array_count=27,model_counts=built['audit']['counts'],new_model_fits=0,predictive_model_fits=0,
        new_ridge_fits=0,new_tree_fits=0,model_fit_accounting=contract()['model_fit_accounting'])
    verify_prepared_evidence(completion,meta,evidence,parity,built)
    dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)


def verify_prepared_evidence(completion,meta,evidence,parity,built):
    require(set(completion['artifacts_sha256'])==PREPARED_FILES,'Exact7 prepared artifacts required')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and
        all(len(row)==49 and len(set(row))==49 for row in meta['symbols']),'Full122x49 metadata required')
    require(built['audit']['symbols_per_event']==49,'Prepared model width differs')
    require_model_audit(evidence['model'],built,122);require_selection_audit(evidence['selection'],122)
    require([row['date'] for row in built['audit']['event_history']]==meta['dates'] and
        [row['symbols'] for row in built['audit']['event_history']]==meta['symbols'],'Model/query identity differs')
    require(evidence.get('prefixes')==123 and evidence.get('model_prefixes')==list(MODEL_PREFIXES),'Missing prefix coverage')
    for key in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_independently_verified','original_orders_exact','original_arrays_unchanged',
        'original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget',
        'full_eligible_replacement','constant_effect_not_fallback','calm_cache_independently_verified',
        'selection_uses_zero_cost_paths_only'):
        require(evidence.get(key) is True,'Prepared invariant missing '+key)
    require(evidence.get('original_array_count')==completion.get('original_array_count')==27,'Original array count differs')
    require(completion.get('new_returns_computed') is False,'Preparation scope differs')
    for key in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits'):
        require(type(completion.get(key)) is int and completion[key]==0,'No predictive fitting allowed')
    require(completion.get('model_fit_accounting')==contract()['model_fit_accounting'],'Model accounting scope differs')
    require(evidence.get('model_counts')==completion.get('model_counts')==built['audit']['counts'],'Main-pass descriptive counts differ')
    require_calm_audit(parity['calm_parity'],meta,built['audit'])
    prefix_evidence=evidence.get('model_prefix_evidence')
    require(isinstance(prefix_evidence,list) and [row.get('prefix') for row in prefix_evidence]==list(MODEL_PREFIXES),
        'Physical model prefix receipts missing')
    for row in prefix_evidence:
        n=row['prefix'];mm=dict(meta)
        for key in ('dates','ends','symbols'):mm[key]=meta[key][:n]
        small=slice_built(built,n)
        require(row.get('query_rows')==row.get('metadata_rows')==n and row.get('original_array_count')==27,
            'Physical prefix array/metadata coverage differs')
        require_model_audit(row['model'],small,n);require_selection_audit(row['selection'],n)
        require_calm_audit(row['calm'],mm,small['audit'])
        require(row['calm']['records']==parity['calm_parity']['records'][:n],'Physical calm source prefix differs')
        require(row.get('counts')==small['audit']['counts'],'Physical prefix descriptive counts differ')
        require(type(row.get('price_rows')) is int and row['price_rows']>=0,'Physical price row count differs')
        if n:
            require(row.get('query_date')==meta['dates'][n-1] and row['price_rows']>0 and
                row.get('latest_price_date')==meta['dates'][n-1],'Physical source not truncated to current query date')
        else:
            require(row['price_rows']==0 and row.get('query_date') is None and row.get('latest_price_date') is None,
                'Empty prefix contains future price data')
    for key in ('original_metadata_identity_verified','original_arrays_unchanged','original_orders_exact'):
        require(parity.get(key) is True,'Source parity missing '+key)
    for key in ('new_training_labels','new_historical_forecasts','strict_pit'):
        require(parity.get(key) is False,'Source parity scope differs '+key)
    require(parity.get('calendar_sha256')==sha(CALENDAR),'Official source calendar differs')
    fp=parity.get('original_array_fingerprints')
    require(isinstance(fp,dict) and len(fp)==27,'All27 original byte fingerprints required')
    for record in fp.values():
        require(isinstance(record,dict) and set(record)=={'dtype','shape','sha256'} and
            isinstance(record['shape'],list) and record['shape'][0]==122 and isinstance(record['dtype'],str) and
            isinstance(record['sha256'],str) and len(record['sha256'])==64,'Invalid original array byte fingerprint')
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):
        require(parity['execution_prices'].get(key) is True,'Execution source parity missing '+key)


def prepared(expected):
    plan=gate(expected);dest=OUT/'prepared';completion=read(dest/'COMPLETION.json')
    require(completion['status']=='COMPLETE' and completion['plan_sha256']==expected,'Preparation incomplete')
    require(set(completion['artifacts_sha256'])==PREPARED_FILES,'Exact7 prepared artifacts required')
    for name,digest in completion['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared artifact '+name)
    built=dict(arrays=ev.load_arrays(dest/'FEATURES.npz'),audit=read(dest/'MODEL_AUDIT.json'))
    meta=read(dest/'META.json');evidence=read(dest/'INDEPENDENT_PREPARE.json');parity=read(dest/'SOURCE_PARITY.json')
    verify_prepared_evidence(completion,meta,evidence,parity,built)
    require(meta==read(LOCAL/'META.json'),'Stored original query metadata changed')
    require(parity['price_source_binding']==plan['price_source_binding'] and
        parity['raw_source_hashes']==plan['raw_source_hashes'],'Stored source proof differs from registration')
    scores=ev.load_arrays(dest/'SCORES.npz');orders=read(dest/'ORDERS.json')
    require(set(scores)==set(orders)==set(NAMES),'Stored score/order arms differ')
    for name in NAMES:
        require(scores[name].shape==(122,49) and scores[name].dtype==np.dtype('float64') and
            np.isfinite(scores[name]).all() and len(orders[name])==122,'Stored score/order axes differ')
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Stored original order parity differs')
    return meta,orders


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
