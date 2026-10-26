"""Round232 bounded reported-volume-rank weighted long log-price time slope.
All actual features/orders/returns require the immutable prospective PLAN.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, hashlib, importlib.util, io, json, os, tempfile, unittest
import numpy as np
import pandas as pd
import ptr169_rebuild_evaluate as ev
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_volume_slope232'
LOCAL=ROOT/'outputs/monthly_local_linear70'
PRIOR=ROOT/'outputs/monthly_market_drift231'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED=ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES=('incumbent_local67','volume_rank_slope','unweighted_slope_control')
RAW={NAMES[1]:'candidate',NAMES[2]:'control'}
TRAIN,DOSE=88,.01
MODEL_PREFIXES=(0,1,53,88,122)
CONTEXT=ROOT/'work/monthly_volume_slope232_context.json'
CONTEXT_SHA='d541a8533a6694dd6a74d19a23dda4d97a5a104d7ac85e9beeb733e6df9df6d8'
CLARIFICATIONS=ROOT/'work/volume_slope232_CLARIFICATIONS.json'
CLARIFICATIONS_SHA='5e0b50bb216fb73531b15d43d6e77a0ee029e56f892dc71894216810ddb91c1d'
COVERAGE=ROOT/'work/volume_slope232_COVERAGE.json'
COVERAGE_SHA='58fa25bde4c040140c10fee0f1931d5e5d5c8760b95e2e15a2da87314adca46a'
SOURCE_REVIEW=ROOT/'work/volume_slope232_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA='868858c5af2988e9863d63e0bd32f1a5952f31a3f42cc6fa87cbeedb33e96734'
TEST_RECEIPT=ROOT/'work/monthly_volume_slope232_TESTS.json'
EXTERNAL_RUNTIME={'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED={'PLAN.json':'ff0699567e0ebb6781bdb6386ad11df7d734d22e3dfff0584afee3fa8f8afeff','COMPLETION.json':'7a0060a5db6f9ed784b25e40d16955c1f06129f02617a5da9f6022865bf5fda5','results/COMPLETION.json':'a32cb9560db7f1ecd2fb43bff7df6d09957bfe4d5ee8b711b156c1d3bc3948c3','RESEARCH_REVIEW.json':'92013baa0d874f16ccc9c3cffd2175b4326b450ec252ffc30b1557b207cd226e'}
PREPARED_FILES={'FEATURES.npz','SCORES.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
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
    backup = CENTRAL.parent / ('PRE_MONTHLY_VOLUME_SLOPE232_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/volume_slope232_CODE_REVIEW.json'
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
        close_vs_unweighted_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_unweighted_control=guard(metrics[NAMES[1]]['next_open_zero'], metrics[NAMES[2]]['next_open_zero'], 0.),
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
    import monthly_volume_slope232_independent as independent
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
    import monthly_volume_slope232_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, volume_rank_time_slope=True, stored_volume_proxy=True, source_volume_units_certified=False, original_episode_preserved=True, zero_cost_selection=True, next_open_cost_diagnostic_only=True, same_support_both_arms=True, observer_promotion=False))
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
    names=('test_monthly_volume_slope232_host.py','test_monthly_volume_slope232_kernel.py',
        'test_monthly_volume_slope232_independent.py')
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module('slope232test'+str(i),ROOT/'work'/name))
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
        next_step='Review reported-volume weighted time slope against same-window unweighted trend; keep frozen observers unchanged.')
    require(len(state['completed']) == 228 and sum(r['trials'] for r in state['completed']) == 1201, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

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
    plan=dict(schema='monthly_volume_slope232_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        price_source_binding=source,numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_parameters_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_volume_time_slope',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare fixed reported-volume-weighted and unweighted time slopes; freeze first88 zero-cost selection before later34 diagnostics.')
    commit_state(before,state,'register')
    print(json.dumps(dict(status='REGISTERED',plan_sha256=sha(OUT/'PLAN.json'))),flush=True)

def array_snapshot(arrays):
    return {key:(value.dtype.str,value.shape,value.tobytes()) for key,value in arrays.items()}

def array_fingerprints(arrays):
    return {key:dict(dtype=value.dtype.str,shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key,value in arrays.items()}

def contract():
    for path,digest in ((CONTEXT,CONTEXT_SHA),(CLARIFICATIONS,CLARIFICATIONS_SHA),(COVERAGE,COVERAGE_SHA)):
        require(sha(path)==digest,'Prospective definition changed: '+str(path))
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,clarifications_sha256=CLARIFICATIONS_SHA,
        coverage_sha256=COVERAGE_SHA,arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],
        period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,dose=.01,max_absolute_delta=.005,
        model_prefixes=list(MODEL_PREFIXES),physical_model_prefixes=True,all123_order_prefixes=True,
        selection_paths=['same_close_zero','next_open_zero'],next_open_cost_diagnostic_only=True,
        model_fit_accounting=context['formation']['accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 zero-cost paths select; later34/full repeated research diagnostic, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,new_model_fits=0,predictive_model_fits=0,
        no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True,
        stored_volume_proxy=True,volume_units_certified=False,strict_pit=False)


def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_volume_slope232_host.py','test_monthly_volume_slope232_host.py',
        'monthly_volume_slope232_kernel.py','test_monthly_volume_slope232_kernel.py',
        'monthly_volume_slope232_independent.py','test_monthly_volume_slope232_independent.py',
        'monthly_volume_slope232_context.json','volume_slope232_CLARIFICATIONS.json','volume_slope232_COVERAGE.json',
        'volume_slope232_SOURCE_REVIEW.json','volume_slope232_source_metadata.py','ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_volume_slope232_independent as independent
    return direct+sorted(set(independent.dependency_files())-set(direct))


def state_check(state):
    require(len(state['completed'])==227 and sum(row['trials'] for row in state['completed'])==1198,'Expected227 completed/1198trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round231 must be predecessor')


def price_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Source metadata review changed')
    review=read(SOURCE_REVIEW)
    for key in ('passed','source_only','metadata_only','conditional_proxy_admission'):
        require(review.get(key) is True,'Missing source review '+key)
    for key in ('volume_units_certified','strict_pit','source_npz_decoded','actual_support_computed','actual_features_computed','actual_returns_computed','observers_modified'):
        require(review.get(key) is False,'Source scope differs '+key)
    require(review.get('context_sha256')==CONTEXT_SHA and review.get('coverage_sha256')==COVERAGE_SHA,'Source prospective binding differs')
    require(review.get('raw_source_count')==64 and review.get('raw_source_hashes')==raw and len(raw)==64,'Original64 raw map differs')
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['input_sha256'].items():bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        conditional_proxy_admission=True,volume_units_certified=False,strict_pit=False)


def scale_and_delta(values,eligible,n):
    import monthly_volume_slope232_kernel as kernel
    require(set(values)=={'candidate','control','available'},'Three feature arrays required')
    available=np.asarray(values['available']);eligible=np.asarray(eligible)
    require(available.dtype==eligible.dtype==bool and available.ndim==2 and available.shape==eligible.shape,
            'Exact boolean common eligible axes required')
    require(type(n) is int and 0<=n<=len(available) and not (available & ~eligible).any(),'Feature outside original eligible domain')
    ranks,deltas={},{}
    for name,key in RAW.items():
        raw=np.asarray(values[key]);require(raw.dtype==np.float64 and raw.shape==available.shape and
            not np.isinf(raw).any() and np.array_equal(np.isfinite(raw),available),'Exact raw common mask required')
        r=np.zeros((n,available.shape[1]),float)
        for i in range(n):r[i]=kernel.centered_rank(raw[i],available[i])
        delta=DOSE*r
        if n:delta[0]=0.
        require(np.isfinite(r).all() and (np.abs(r)<=.5).all() and not r[~available[:n]].any() and
            np.isfinite(delta).all() and (np.abs(delta)<=.005).all(),'Auxiliary rank/score budget exceeded')
        ranks[name],deltas[name]=r,delta
    return ranks,deltas


def decisions(meta,arrays,values,n,study=None):
    study=local() if study is None else study
    a=slice_arrays(arrays,n);m=dict(meta)
    for key in ('dates','ends','symbols'):m[key]=meta[key][:n]
    v={key:np.asarray(value)[:n].copy() for key,value in values.items()}
    policy=read(LOCAL/'INCUMBENT_POLICY.json')
    ch,sc=study.choices(study.configs()[4],a,m,n,policy=policy)
    choices={NAMES[0]:ch};scores={NAMES[0]:sc}
    ranks,deltas=scale_and_delta(v,a['eligible'],n)
    for name in NAMES[1:]:
        pre=study.modified_pre_score(study.configs()[4],a,n)
        nonzero=deltas[name]!=0.;pre[nonzero]+=deltas[name][nonzero]
        ch,sc=study.adapter.score_and_choose('incumbent66',0.,np.full(pre.shape,np.nan),a,m,n,
            pre_score=pre,effects=a['effects'],episode_policy=policy,boundary_selector=study.ep.select_boundary)
        choices[name]=ch;scores[name]=sc
    return {name:study.rp.make_orders(choices[name],scores[name],m) for name in NAMES},scores,choices


def model_counts(audit):
    names=('events','checked_cells','eligible_cells','attempted_cells','available_cells','unavailable_cells',
        'initial_partial_cells','ineligible_cells','source_price_observations','source_volume_observations',
        'volume_rank_observations','descriptive_time_slope_estimates','unweighted_slope_solves','weighted_slope_solves',
        'constant_price_cells','constant_volume_cells','constant_volume_copies','constant_price_shortcuts')
    result={key:audit[key] for key in names}
    require(all(type(v) is int and v>=0 for v in result.values()),'Descriptive nonnegative counts required')
    result['cross_sectional_rank_observations']=2*audit['available_cells']
    return result


def require_model_audit(check,values,audit,rows):
    import monthly_volume_slope232_independent as independent
    for key in independent.FEATURE_FLAGS:require(check.get(key) is True,'Independent feature audit missing '+key)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False and
        check.get('query_rows')==rows and check.get('numerical_tolerance')==0.,'Feature oracle scope differs')
    require(check.get('counts')=={k:audit[k] for k in independent.COUNT_FIELDS},'Independent feature counts differ')
    for key in ('new_model_fits','predictive_model_fits'):
        require(type(check.get(key)) is int and check[key]==0,'Independent feature fit accounting differs')
    require(audit.get('events')==rows and len(audit.get('event_history',[]))==rows,'Feature event coverage differs')
    require(audit.get('common_availability') is True and audit.get('trading_eligibility_modified') is False and
        audit.get('predictive_model_fits')==0 and audit.get('new_model_fits')==0 and
        audit.get('portfolio_returns_calculated') is False,'Feature scope differs')
    require(audit.get('window_price_count')==audit.get('window_volume_count')==253 and
        audit.get('span_sessions')==252 and audit.get('skip_sessions')==42 and
        audit.get('source_start_offset')==-294 and audit.get('source_end_offset')==-42,'Formation window differs')
    mask=np.asarray(values['available']);require(mask.dtype==bool and mask.shape==(rows,audit['symbols_per_event']),'Feature mask shape')
    c=model_counts(audit);A=int(mask.sum());require(c['available_cells']==A and
        c['checked_cells']==mask.size and c['unavailable_cells']==mask.size-A and
        c['source_price_observations']==c['source_volume_observations']==c['volume_rank_observations']==253*A and
        c['descriptive_time_slope_estimates']==2*A,'Feature accounting differs')
    require(c['unweighted_slope_solves']+c['constant_price_shortcuts']==A and
        c['weighted_slope_solves']+c['constant_volume_copies']+c['constant_price_shortcuts']==A,'Solve/copy accounting differs')
    for key in ('candidate','control'):
        raw=np.asarray(values[key]);require(raw.dtype==np.float64 and raw.shape==mask.shape and
            not np.isinf(raw).any() and np.array_equal(np.isfinite(raw),mask),'Raw support differs')


def require_selection_audit(check,rows):
    import monthly_volume_slope232_independent as independent
    for key in independent.SELECTION_FLAGS:require(check.get(key) is True,'Independent selection missing '+key)
    require(check.get('producer_modules_imported') is False and check.get('computes_returns') is False and
        check.get('numerical_tolerance')==0. and check.get('original_source_parity_tolerance')==1e-12,'Selection oracle scope differs')
    bound=check.get('max_absolute_delta')
    require(type(bound) in (float,int) and np.isfinite(bound) and 0.<=bound<=.005,'Selection score bound differs')
    require(check.get('ordered_choices_checked')==3*rows,'Selection rows differ')


def prefix_inputs(panel,meta,calendar,arrays,n):
    a=slice_arrays(arrays,n);m=dict(meta)
    for key in ('dates','ends','symbols'):m[key]=meta[key][:n]
    cutoff=m['dates'][-1] if n else None
    p={key:frame.loc[:cutoff].copy() if n else frame.iloc[:0].copy() for key,frame in panel.items()}
    return p,m,calendar,a


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_volume_slope232_kernel as kernel
    import monthly_volume_slope232_independent as independent
    study=local();meta,arrays,px,op=study.load()
    require(meta==read(LOCAL/'META.json') and len(meta['dates'])==122 and arrays['baseline'].shape==(122,49) and
        len(arrays)==27,'Original metadata/27arrays/122x49 required')
    before=array_snapshot(arrays)
    source,info=study.arch.old.panel_load(meta['etfs']);raw=study.raw_source_hashes(info)
    require(raw==plan['raw_source_hashes'],'Raw source lineage changed');verify(raw)
    execution=study.verify_panel_parity(source,px,op)
    panel={key:source[key] for key in ('adj','volume')};calendar=read(CALENDAR)['sessions']
    eligible=np.asarray(arrays['eligible'],dtype=bool)
    values,audit=kernel.build_features(panel,meta,calendar,eligible)
    check=independent.verify_features(panel,meta,calendar,eligible,values,audit)
    require_model_audit(check,values,audit,122)
    orders,scores,choices=decisions(meta,arrays,values,122,study)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original orders changed')
    selection=independent.verify_selection(arrays,meta,values,scores,choices);require_selection_audit(selection,122)
    prefixes=[]
    for n in MODEL_PREFIXES:
        pp,mm,cc,aa=prefix_inputs(panel,meta,calendar,arrays,n)
        vv,au=kernel.build_features(pp,mm,cc,aa['eligible'])
        ck=independent.verify_features(pp,mm,cc,aa['eligible'],vv,au);require_model_audit(ck,vv,au,n)
        for key in values:np.testing.assert_array_equal(vv[key],values[key][:n])
        require(au['event_history']==audit['event_history'][:n],'Physical feature prefix differs')
        oo,ss,ch=decisions(mm,aa,vv,n,study)
        cs=independent.verify_selection(aa,mm,vv,ss,ch);require_selection_audit(cs,n)
        require(all(oo[k]==orders[k][:n] and ch[k]==choices[k][:n] for k in NAMES),'Physical order prefix differs')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
        prefixes.append(dict(prefix=n,source_rows={k:len(v) for k,v in pp.items()},feature=ck,selection=cs,counts=model_counts(au)))
    for n in range(123):
        oo,ss,ch=decisions(meta,arrays,values,n,study)
        require(all(oo[k]==orders[k][:n] and ch[k]==choices[k][:n] for k in NAMES),'Decision prefix differs')
        for key in NAMES:np.testing.assert_array_equal(ss[key],scores[key][:n])
    require(before==array_snapshot(arrays),'Original27arrays modified')
    parity=dict(raw_source_hashes=raw,price_source_binding=plan['price_source_binding'],execution_prices=execution,
        calendar_sha256=sha(CALENDAR),original_array_fingerprints=array_fingerprints(arrays),
        original_arrays_unchanged=True,original_orders_exact=True,volume_units_certified=False,strict_pit=False)
    evidence=dict(model=check,selection=selection,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        physical_prefix_evidence=prefixes,original_orders_exact=True,original_arrays_unchanged=True,original_array_count=27,
        original_core_preserved=True,single_original_episode=True,additional_max_score_budget=.005,
        model_counts=model_counts(audit),selection_uses_zero_cost_paths_only=True)
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FEATURES.npz',**values);np.savez_compressed(dest/'SCORES.npz',**scores)
    dump(dest/'MODEL_AUDIT.json',audit);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',meta)
    dump(dest/'SOURCE_PARITY.json',parity);dump(dest/'INDEPENDENT_PREPARE.json',evidence)
    completion=dict(status='COMPLETE',created_utc=now(),plan_sha256=expected,
        artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},original_array_count=27,model_counts=model_counts(audit),
        new_returns_computed=False,new_model_fits=0,predictive_model_fits=0,model_fit_accounting=contract()['model_fit_accounting'])
    verify_prepared_evidence(completion,meta,evidence,parity,values,audit)
    dump(dest/'COMPLETION.json',completion)
    print(json.dumps(dict(status='PREPARED_INDEPENDENTLY_VERIFIED',completion_sha256=sha(dest/'COMPLETION.json'),counts=model_counts(audit))),flush=True)


def verify_prepared_evidence(completion,meta,evidence,parity,values,audit):
    require(set(completion['artifacts_sha256'])==PREPARED_FILES,'Exact7 prepared artifacts required')
    require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and
        all(len(row)==49 and len(set(row))==49 for row in meta['symbols']),'Full122x49 identity required')
    require_model_audit(evidence['model'],values,audit,122);require_selection_audit(evidence['selection'],122)
    require([e['date'] for e in audit['event_history']]==meta['dates'] and
        [e['symbols'] for e in audit['event_history']]==meta['symbols'],'Feature identity differs')
    for key in ('original_orders_exact','original_arrays_unchanged','original_core_preserved','single_original_episode','selection_uses_zero_cost_paths_only'):
        require(evidence.get(key) is True,'Prepared invariant missing '+key)
    require(evidence.get('original_array_count')==completion.get('original_array_count')==27 and
        evidence.get('additional_max_score_budget')==.005,'Original arrays/score dose differ')
    require(evidence.get('prefixes')==123 and evidence.get('model_prefixes')==list(MODEL_PREFIXES) and
        [p['prefix'] for p in evidence.get('physical_prefix_evidence',[])]==list(MODEL_PREFIXES),'Prefix receipts differ')
    for row in evidence['physical_prefix_evidence']:
        n=row['prefix'];require_selection_audit(row['selection'],n)
        import monthly_volume_slope232_independent as independent
        for key in independent.FEATURE_FLAGS:require(row['feature'].get(key) is True,'Physical independent feature check missing')
        require(row['counts']['events']==n and set(row['source_rows'])=={'adj','volume'},'Physical prefix count differs')
        require(all(type(v) is int and (v>0 if n else v==0) for v in row['source_rows'].values()),'Physical prefix source rows')
    require(evidence['model_counts']==completion['model_counts']==model_counts(audit),'Descriptive counts differ')
    require(completion.get('new_returns_computed') is False and completion.get('new_model_fits')==0 and
        completion.get('predictive_model_fits')==0 and completion['model_fit_accounting']==contract()['model_fit_accounting'],'Preparation scope differs')
    require(parity.get('calendar_sha256')==sha(CALENDAR) and parity.get('original_arrays_unchanged') is True and
        parity.get('original_orders_exact') is True and parity.get('volume_units_certified') is False and
        parity.get('strict_pit') is False,'Source parity scope differs')
    require(len(parity['original_array_fingerprints'])==27,'Original27 byte fingerprints required')
    for key in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact'):
        require(parity['execution_prices'].get(key) is True,'Execution parity missing '+key)


def prepared(expected):
    gate(expected);dest=OUT/'prepared';completion=read(dest/'COMPLETION.json')
    require(completion['status']=='COMPLETE' and completion['plan_sha256']==expected,'Preparation incomplete')
    require(set(completion['artifacts_sha256'])==PREPARED_FILES,'Exact prepared artifacts required')
    for name,digest in completion['artifacts_sha256'].items():require(sha(dest/name)==digest,'Changed prepared '+name)
    with np.load(dest/'FEATURES.npz',allow_pickle=False) as f:values={k:f[k] for k in f.files}
    meta=read(dest/'META.json')
    verify_prepared_evidence(completion,meta,read(dest/'INDEPENDENT_PREPARE.json'),read(dest/'SOURCE_PARITY.json'),values,read(dest/'MODEL_AUDIT.json'))
    return meta,read(dest/'ORDERS.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('test','register','prepare','select','diagnose','close'))
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(args.plan_sha256,'Explicit immutable PLAN SHA required')
        globals()[args.phase](args.plan_sha256)


if __name__=='__main__':main()
