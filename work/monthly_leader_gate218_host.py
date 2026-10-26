"""Round218: fixed six-leader shortlist followed by the original baseline score.

New arms consume only original B/core/core_weight/eligible cache cells. The
Local67 reference retains its complete original score and episode. Actual
selection, orders and returns are gated by the prospectively registered PLAN.
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_leader_gate218'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PRIOR=ROOT/'outputs/monthly_persistence217'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA='8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
BASELINE_ORDERS=ROOT/'outputs/monthly_model_scope70/original65_ORDERS.json'
NAMES=('incumbent_local67','leader6_gate','baseline_score')
TRAIN=88
PREFIXES=tuple(range(123))
CONTEXT=ROOT/'work/monthly_leader_gate218_context.json'
CONTEXT_SHA='faec461c4716c65c5bd0d4c7f695df5ca8b3bec395d04b416c671d846bfa1a8c'
COVERAGE=ROOT/'work/after217_leader_gate_coverage.json'
COVERAGE_SHA='cac3fc696efefc4ec6aed08ca1883e8bc7bc66689a028c405d51df466c7a4ba5'
SOURCE_REVIEW=ROOT/'work/leader_gate218_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='0b12bc7b74825c0c5af1bd4c6be99229b9a786e1010494cdd8b9633643fb2f9b'
TEST_RECEIPT=ROOT/'work/monthly_leader_gate218_TESTS.json'
EXTERNAL_RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED={'PLAN.json':'6df0b3fe4db851540a0e44b3452017eebf63b8ac533f34bcb2b0e3e5781d142c','COMPLETION.json':'feaf6bec5a51482c1acda37f249d0ecbb7d07305e1202df8c1afa6b8a964a1b3','results/COMPLETION.json':'2272d22fe629a4125451e1964546ff1606c30c0a5fd5d03087eb556ce19b0de5','RESEARCH_REVIEW.json':'59e4a6d08fe7a7f1d660b79488e3872d05f71411fb705b414a04af7df07b82bc'}
SOURCE_FLAGS=('source_formula_verified','baseline_assignment_chain_verified','original_eligibility_verified','no_industry_cap_verified','original_neutral_core_semantics_preserved','metadata_axes_verified','cache_metadata_hash_verified','no_actual_npz_or_new_factor_read','no_fresh_raw_pit_claim')
PREPARATION_FLAGS=('passed','independent_scalar_rebuild','original_eligibility_preserved','eligible_only_core_division','top6_cutoff_checked','exact_boundary_ties_included','baseline_scores_unchanged','lexical_ties_checked','two_three_cash_checked','no_industry_filter','initial_partial_same_rule','no_labels_or_models_consumed','prefix_only_inputs','audit_checked','zero_fits_and_parameters_checked')
COUNT_FIELDS=('query_months','eligible_cells','shortlisted_cells','gated_months','tie_expansion_months','tie_added_cells','changed_holding_months','changed_ordered_months','cash_months','two_holding_months','three_holding_months')
PREPARED_FILES={'SHORTLIST.npz','SELECTION_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json','BASELINE_PARITY.json'}
RESULT_FILES={arm+'_'+path[0]+'_'+kind+'.json' for arm in NAMES for path in ev.PATHS for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
sha,read,require,dump=ev.sha,ev.read,ev.require,ev.dump


def now():
    return datetime.now(timezone.utc).isoformat()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def local():
    return module('monthly_leader_gate218_original', LOCAL / 'study.py')

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
    backup = CENTRAL.parent / ('PRE_MONTHLY_LEADER_GATE218_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/leader_gate218_CODE_REVIEW.json'
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
    require(p['cache_source_binding'] == cache_source_binding(p['raw_source_hashes']), 'Registered model source binding changed')
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
        close_vs_baseline_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_baseline_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_leader_gate218_independent as independent
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
    import monthly_leader_gate218_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, leader_shortlist_gate=True, paired_arms_use_only_cached_baseline_core_eligibility=True, paired_arms_use_ml=False, paired_arms_use_episode=False, observer_promotion=False))
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
    names=('test_monthly_leader_gate218_host.py','test_monthly_leader_gate218_model.py',
           'test_monthly_leader_gate218_independent.py')
    modules=[module('test_leader_gate218_'+str(i),ROOT/'work'/name) for i,name in enumerate(names)]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),flush=True);require(result.wasSuccessful(),'Synthetic validation failed')
    require(not TEST_RECEIPT.exists(),'Test receipt is write-once')
    dump(TEST_RECEIPT,dict(passed=True,synthetic_only=True,tests=result.testsRun,runtime=ev.runtime(),
        code_sha256={str(p):sha(p) for p in code_files()}))

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
        result_completion_sha256=sha(OUT / 'results/COMPLETION.json'), new_trials=3, selection_counts=read(OUT/'prepared/COMPLETION.json')['selection_counts'],
        selection_accounting=contract()['selection_accounting'],
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
        next_step='Review long-momentum shortlist selection and source limitations; preserve frozen observers.')
    require(len(state['completed']) == 216 and sum(r['trials'] for r in state['completed']) == 1165, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

def contract():
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective context')
    require(sha(COVERAGE)==COVERAGE_SHA,'Changed coverage evidence')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
        arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],period=context['evaluation']['period'],
        training_months=88,later_months=34,physical_prefixes=list(PREFIXES),all123_order_prefixes=True,
        selection_accounting=context['formation']['fit_accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,
        no_parameter_search=True,paired_arms_use_ml=False,paired_arms_use_episode=False,
        baseline_control_name_is_not_a_performance_claim=True)

def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_leader_gate218_host.py','test_monthly_leader_gate218_host.py',
        'monthly_leader_gate218_model.py','test_monthly_leader_gate218_model.py','monthly_leader_gate218_independent.py',
        'test_monthly_leader_gate218_independent.py','monthly_leader_gate218_context.json','after217_leader_gate_coverage.json',
        'leader_gate218_SOURCE_REVIEW.json','ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_leader_gate218_model as model
    import monthly_leader_gate218_independent as independent
    return direct+sorted((set(model.dependency_files())|set(independent.dependency_files()))-set(direct))

def state_check(state):
    require(len(state['completed'])==215 and sum(row['trials'] for row in state['completed'])==1162,'Expected215 completed entries /1162 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round217 must be the completed predecessor')

def bind_source_evidence(pins, review, context):
    # The independent review records its inspected sources; the context records
    # the registered design's evidence. Both lists are binding, not identical.
    for label, document in (('review', review), ('context', context)):
        evidence=document.get('evidence_sha256')
        require(isinstance(evidence,dict) and evidence,'Source '+label+' evidence required')
        for name,digest in evidence.items():
            bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)

def cache_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed cached-source review')
    review=read(SOURCE_REVIEW);context=read(CONTEXT)
    require(review.get('passed') is True and review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    for key in SOURCE_FLAGS:require(review.get(key) is True,'Source scope missing '+key)
    require(isinstance(review.get('input_sha256'),dict) and review['input_sha256'],'Source input pins required')
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['input_sha256'].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    bind_source_evidence(pins,review,context)
    original_plan=read(LOCAL/'PLAN.json')
    require(original_plan['raw_source_hashes']==raw and len(raw)==64,'Original64 raw-source lineage changed')
    for name,digest in raw.items():bind(pins,Path(name),digest)
    for name in ('PLAN.json','DATA.npz','META.json','INPUT_HASHES.json','CACHE_HASHES.json','INCUMBENT_POLICY.json','local_linear_ORDERS.json'):
        bind(pins,LOCAL/name)
    for name,digest in read(LOCAL/'CACHE_HASHES.json').items():bind(pins,LOCAL/name,digest)
    for name,digest in read(LOCAL/'INPUT_HASHES.json').items():bind(pins,Path(name),digest)
    bind(pins,BASELINE_ORDERS)
    bind(pins,CALENDAR,CALENDAR_SHA)
    meta=read(LOCAL/'META.json')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(row)==49 and len(set(row))==49 for row in meta['symbols']),'Original122x49 identity differs')
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        review_evidence_sha256=review['evidence_sha256'],context_evidence_sha256=context['evidence_sha256'],
        original_cache_sha256=sha(LOCAL/'DATA.npz'),original_meta_sha256=sha(LOCAL/'META.json'),
        baseline_historical_orders_sha256=sha(BASELINE_ORDERS),original_query_identity_verified=True,
        source_scope='Immutable original27 cache and registered assignment chain; no raw reconstruction or fresh PIT audit',
        training_labels_consumed=False,source_forecasts_consumed_by_new_arms=False,new_labels=False,new_raw_features=False,strict_pit=False)

def register():
    require(not OUT.exists(),'Fresh experiment directory required')
    before=CENTRAL.read_bytes();state=json.loads(before);state_check(state)
    require(not state['registered_queue'],'Existing registered experiment')
    previous=predecessor_binding(state);validation=read(TEST_RECEIPT)
    require(validation['passed'] is True and validation['synthetic_only'] is True and validation['runtime']==ev.runtime(),'Synthetic tests required')
    require(validation['code_sha256']=={str(p):sha(p) for p in code_files()},'Code changed since tests')
    raw=read(LOCAL/'PLAN.json')['raw_source_hashes'];source=cache_source_binding(raw);review=code_review_binding();pins={}
    for binding in (previous,source,review):
        for name,digest in binding['input_sha256'].items():bind(pins,Path(name),digest)
    for p in code_files()+[TEST_RECEIPT,PROTECTED,LOCAL/'PLAN.json',BASELINE_ORDERS]:bind(pins,p)
    for name,digest in ev.ENGINE_HASHES.items():bind(pins,ev.ENGINE_DIR/name,digest)
    verify(pins);protected=read(PROTECTED);verify(protected)
    require(len(protected)==21444,'Incomplete protected archive manifest')
    OUT.mkdir()
    with (OUT/'STATE_BEFORE.json').open('xb') as stream:stream.write(before)
    plan=dict(schema='monthly_leader_gate218_v1',created_utc=now(),runtime=ev.runtime(),contract=contract(),
        input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,cache_source_binding=source,
        numerical_code_review=review,prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_parameters_before_registration=0,actual_new_transforms_before_registration=0,
        actual_new_orders_before_registration=0,actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_new_leader_shortlist',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare fixed leader shortlist and exact baseline parity, then freeze first88 selection.')
    commit_state(before,state,'register')
    print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def verify_calendar_dates(meta, calendar):
    sessions=calendar['sessions']
    require(isinstance(sessions,list) and sessions==sorted(set(sessions)),'Pinned trading calendar invalid')
    ends={day[:7]:day for day in sessions}
    dates=meta['dates']
    require(dates and dates[0]=='2016-08-17' and dates==sorted(set(dates)),'Original partial/date order differs')
    require(dates[0] in sessions and all(ends.get(day[:7])==day for day in dates[1:]),'Noninitial query is not formal month-end')
    return dict(passed=True,initial_partial=dates[0],formal_events=len(dates)-1,calendar_sha256=sha(CALENDAR))

def prefix_inputs(arrays,meta,n):
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Invalid array/metadata prefix')
    require(all(isinstance(meta.get(k),list) and len(meta[k])==shape[0] for k in ('dates','ends','symbols')),'Metadata axes differ')
    a={}
    for key,value in arrays.items():
        v=np.asarray(value)
        require(v.ndim>=1 and v.shape[0]==shape[0],'All original cache arrays must share monthly axis '+key)
        a[key]=v[:n].copy()
    for key in ('eligible','forecast_available','model_available','local_available','extension_available','state_weak_narrow','state_calm10'):
        if key in a:
            require(all(isinstance(v,(bool,np.bool_)) for v in a[key].flat),'Exact boolean values required '+key)
            a[key]=a[key].astype(bool)
    m=dict(meta)
    for key in ('dates','ends','symbols'):m[key]=[list(v) if isinstance(v,list) else v for v in meta[key][:n]]
    return a,m

def decisions(study,arrays,meta,built,n):
    a,m=prefix_inputs(arrays,meta,n)
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    choices,scores=study.choices(study.configs()[4],a,m,n,policy=policy)
    orders={NAMES[0]:study.rp.make_orders(choices,scores,m)}
    require(set(built['orders'])==set(NAMES[1:]),'Paired order names differ')
    for name in NAMES[1:]:
        require(len(built['orders'][name])==n,'Paired order count differs')
        orders[name]=built['orders'][name]
    return orders

def baseline_parity(study,arrays,meta,built,n,historical=None):
    a,m=prefix_inputs(arrays,meta,n)
    chosen=study.rp.choose(a['baseline'],a['eligible'],m['symbols'],retain=3)
    expected=study.rp.make_orders(chosen,a['baseline'],m)
    require(built['orders'][NAMES[2]]==expected,'Baseline control differs from original retain3 selector')
    if historical is not None:require(expected==historical[:n],'Baseline control differs from historical original65 orders')
    return dict(passed=True,original_selector_orders_exact=True,historical_original65_orders_exact=historical is not None,
        checked_months=n,baseline_name_is_not_a_performance_claim=True,original_selector='rp.choose(B,E,symbols,retain=3) then rp.make_orders',
        historical_orders_path=str(BASELINE_ORDERS),historical_orders_sha256=sha(BASELINE_ORDERS) if historical is not None else None)

def require_preparation_audit(check,built,n=122,width=49):
    for key in PREPARATION_FLAGS:require(check.get(key) is True,'Independent selection evidence missing '+key)
    require(check.get('computes_returns') is False and check.get('producer_modules_imported') is False,'Independent audit scope differs')
    require(set(built)=={'orders','shortlist','audit'} and set(built['orders'])==set(NAMES[1:]),'Paired result schema differs')
    mask=np.asarray(built['shortlist']);require(mask.shape==(n,width) and mask.dtype==np.dtype(bool),'Shortlist shape/dtype differs')
    require(all(len(v)==n for v in built['orders'].values()),'Paired order coverage differs')
    counts=built['audit']['counts']
    require(set(counts)==set(COUNT_FIELDS) and all(type(v) is int and v>=0 for v in counts.values()),'Selection counters differ')
    require(check.get('counts')==counts,'Independent counter equality differs')
    require(check.get('orders_checked')=={name:n for name in NAMES[1:]},'Independent order coverage differs')
    audit=built['audit'];records=audit.get('records')
    require(audit.get('method')=='eligible_core_top6_then_baseline_top3' and isinstance(records,list) and len(records)==n,'Selection method or row coverage differs')
    require([r.get('row') for r in records]==list(range(n)),'Selection row indices differ')
    expected=dict(query_months=n,eligible_cells=sum(r['eligible_count'] for r in records),
        shortlisted_cells=int(mask.sum()),gated_months=sum(r['eligible_count']>6 for r in records),
        tie_expansion_months=sum(r['tie_added_count']>0 for r in records),tie_added_cells=sum(r['tie_added_count'] for r in records),
        changed_holding_months=sum(set(built['orders'][NAMES[1]][i]['symbols'])!=set(built['orders'][NAMES[2]][i]['symbols']) for i in range(n)),
        changed_ordered_months=sum(built['orders'][NAMES[1]][i]['symbols']!=built['orders'][NAMES[2]][i]['symbols'] for i in range(n)),
        cash_months=sum(r['eligible_count']<2 for r in records),two_holding_months=sum(r['eligible_count']==2 for r in records),
        three_holding_months=sum(r['eligible_count']>=3 for r in records))
    require(counts==expected,'Selection counter identities differ')
    for i,row in enumerate(records):
        require(type(row.get('eligible_count')) is int and 0<=row['eligible_count']<=width and type(row.get('shortlist_count')) is int and row['shortlist_count']==int(mask[i].sum()) and row['shortlist_count']<=row['eligible_count'],'Row selection sizes differ')
        require(row.get('leader6_symbols')==built['orders'][NAMES[1]][i]['symbols'] and row.get('baseline_symbols')==built['orders'][NAMES[2]][i]['symbols'],'Audit/order symbols differ')
        require(row.get('changed_holdings')==(set(row['leader6_symbols'])!=set(row['baseline_symbols'])) and row.get('changed_order')==(row['leader6_symbols']!=row['baseline_symbols']),'Audit change flags differ')
        require(row.get('date')==built['orders'][NAMES[1]][i]['date']==built['orders'][NAMES[2]][i]['date'],'Audit/order dates differ')
    for key in ('new_model_fits','predictive_model_fits','new_ridge_fits','new_tree_fits','new_parameters'):
        require(type(built['audit'].get(key)) is int and built['audit'][key]==0 and type(check.get(key)) is int and check[key]==0,'Unexpected fitting '+key)

def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_leader_gate218_model as model
    import monthly_leader_gate218_independent as independent
    study=local();m=read(LOCAL/'META.json');a=ev.load_arrays(LOCAL/'DATA.npz')
    require(len(m['dates'])==122 and a['baseline'].shape==(122,49) and a['core_weight'].shape==(122,1) and len(a)==27,'Original122x49/27 arrays and column core_weight required')
    before={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()}
    calendar_check=verify_calendar_dates(m,read(CALENDAR))
    built=model.build(a,m,122);check=independent.audit_preparation(a,m,122,built);require_preparation_audit(check,built)
    historical=read(BASELINE_ORDERS);baseline=baseline_parity(study,a,m,built,122,historical)
    orders=decisions(study,a,m,built,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'OriginalLocal67 full order parity failed')
    prefix_evidence=[]
    for n in PREFIXES:
        aa,mm=prefix_inputs(a,m,n);require(len(aa)==27 and all(v.shape[0]==n for v in aa.values()),'All27 physical cache axes required')
        small=model.build(aa,mm,n);proof=independent.audit_preparation(aa,mm,n,small);require_preparation_audit(proof,small,n,49)
        oo=decisions(study,aa,mm,small,n)
        require(all(oo[key]==orders[key][:n] for key in NAMES),'Physical prefix order parity failed')
        np.testing.assert_array_equal(small['shortlist'],built['shortlist'][:n])
        baseline_parity(study,aa,mm,small,n,historical)
        prefix_evidence.append(dict(prefix=n,query_rows=n*49,query_date=m['dates'][n-1] if n else None,independent_passed=True,
            arrays_physically_truncated=True,metadata_physically_truncated=True,baseline_parity=True,counts=small['audit']['counts']))
    require(before=={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()},'Original arrays modified')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'SHORTLIST.npz',shortlist=built['shortlist'])
    dump(dest/'SELECTION_AUDIT.json',built['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',m);dump(dest/'BASELINE_PARITY.json',baseline)
    dump(dest/'SOURCE_PARITY.json',dict(original_identity_exact=True,raw_source_hashes=plan['raw_source_hashes'],cache_source_binding=plan['cache_source_binding'],
        original_cache_bytes_bound=True,training_labels_consumed=False,source_forecasts_consumed_by_new_arms=False,new_raw_features=False,
        original_core_neutral_semantics_preserved=True,calendar_check=calendar_check,strict_pit=False,fresh_raw_training_reconstruction=False))
    evidence=dict(selection=check,calendar_check=calendar_check,prefixes=123,physical_prefixes=list(PREFIXES),prefix_evidence=prefix_evidence,
        original_orders_exact=True,baseline_control_orders_exact=True,historical_baseline_orders_exact=True,
        original_arrays_unchanged=True,original_array_count=27,original_incumbent_policy_and_episode_preserved=True,
        paired_arms_use_ml=False,paired_arms_use_episode=False,paired_initial_partial_uses_formula=True,
        training_labels_consumed=False,physical_arrays_truncated=True,physical_metadata_truncated=True,selection_counts=built['audit']['counts'])
    dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    dump(dest/'COMPLETION.json',dict(status='COMPLETE',plan_sha256=expected,created_utc=now(),artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},
        new_returns_computed=False,original_array_count=27,selection_counts=built['audit']['counts'],new_model_fits=0,new_parameters=0,
        selection_accounting=contract()['selection_accounting']))
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'))),flush=True)

def verify_prepared_evidence(c,built,audit,meta,baseline):
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    require_preparation_audit(audit['selection'],built)
    require(audit.get('calendar_check')==verify_calendar_dates(meta,read(CALENDAR)),'Prepared calendar evidence differs')
    require(audit.get('prefixes')==123 and audit.get('physical_prefixes')==list(PREFIXES),'Missing physical prefix coverage')
    for key in ('original_orders_exact','baseline_control_orders_exact','historical_baseline_orders_exact','original_arrays_unchanged','original_incumbent_policy_and_episode_preserved','paired_initial_partial_uses_formula','physical_arrays_truncated','physical_metadata_truncated'):
        require(audit.get(key) is True,'Prepared invariant missing '+key)
    for key in ('paired_arms_use_ml','paired_arms_use_episode','training_labels_consumed'):require(audit.get(key) is False,'Prepared scope differs '+key)
    require(audit.get('original_array_count')==c.get('original_array_count')==27,'Original array count differs')
    require(c.get('new_returns_computed') is False and c.get('new_model_fits')==c.get('new_parameters')==0,'Preparation research scope differs')
    require(audit.get('selection_counts')==c.get('selection_counts')==built['audit']['counts'],'Selection counts differ')
    for key in ('passed','original_selector_orders_exact','historical_original65_orders_exact','baseline_name_is_not_a_performance_claim'):
        require(baseline.get(key) is True,'Baseline parity missing '+key)
    require(baseline.get('checked_months')==122 and baseline.get('historical_orders_sha256')==sha(BASELINE_ORDERS),'Historical baseline evidence differs')
    require(built['orders'][NAMES[2]]==read(BASELINE_ORDERS),'Stored baseline orders changed')
    rows=audit.get('prefix_evidence');require(isinstance(rows,list) and [r.get('prefix') for r in rows]==list(PREFIXES),'Prefix evidence missing')
    for row in rows:
        n=row['prefix'];require(row.get('query_rows')==n*49 and row.get('query_date')==(meta['dates'][n-1] if n else None),'Prefix identity differs')
        for key in ('independent_passed','arrays_physically_truncated','metadata_physically_truncated','baseline_parity'):require(row.get(key) is True,'Prefix proof missing '+key)
        require(set(row.get('counts',{}))==set(COUNT_FIELDS),'Prefix counter schema differs')

def prepared(expected):
    gate(expected);dest=OUT/'prepared';c=read(dest/'COMPLETION.json')
    require(c['status']=='COMPLETE' and c['plan_sha256']==expected,'Preparation incomplete')
    require(set(c['artifacts_sha256'])==PREPARED_FILES,'Prepared artifact set incomplete')
    for name,digest in c['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared artifact '+name)
    values=ev.load_arrays(dest/'SHORTLIST.npz');require(set(values)=={'shortlist'},'Stored shortlist schema differs')
    orders=read(dest/'ORDERS.json');require(set(orders)==set(NAMES),'Stored order arms differ')
    m=read(dest/'META.json');require(m==read(LOCAL/'META.json'),'Prepared original metadata differs')
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Stored incumbent orders changed')
    built=dict(orders={k:orders[k] for k in NAMES[1:]},shortlist=values['shortlist'],audit=read(dest/'SELECTION_AUDIT.json'))
    verify_prepared_evidence(c,built,read(dest/'INDEPENDENT_PREPARE.json'),m,read(dest/'BASELINE_PARITY.json'))
    return m,orders

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
