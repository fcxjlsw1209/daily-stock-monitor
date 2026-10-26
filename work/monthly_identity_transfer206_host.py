"""Round206: matched-month, matched-mass identity-source residual transfer.

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
OUT = ROOT/'outputs/monthly_identity_transfer206'
LOCAL = ROOT/'outputs/monthly_local_linear70'
REBUILD = ROOT/'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild/control'
PRIOR = ROOT/'outputs/monthly_adjacent_risk205'
CENTRAL = ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
PROTECTED = ROOT/'work/trade_reconcile_20261015_close/NEXT_EXPECTED_PROTECTED.json'
NAMES = ('incumbent_local67','peer_residual_transfer','own_residual_control')
RAW = {NAMES[1]:'candidate', NAMES[2]:'control'}
RAW_KEYS = ('source_log','local_linear','local_constant','candidate','control')
TARGETS = ('simple_excess','log_excess','vol_scaled','loss_utility')
TRAIN = 88
MODEL_PREFIXES = (0,1,53,88,122)
MODEL_COUNT_FIELDS = ('source_target_fits','reference_pooled_log_refits','reference_local_evaluations','reference_local_solves','transfer_pooled_log_refits','own_constant_evaluations','peer_query_evaluations','peer_month_evaluations','peer_row_observations','actual_regression_solves')
CONTEXT = ROOT/'work/monthly_identity_transfer206_context.json'
CONTEXT_SHA = '29d7d503564aca5ac2ebe66a4180e519e93427e0497ba74ecebd15999ab7255b'
SOURCE_REVIEW = ROOT/'work/identity206_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = 'cb4ac9494e06ffab11b718930471c4469646a3df6d2801dc4458b532a4bbf8ae'
TEST_RECEIPT = ROOT/'work/monthly_identity_transfer206_TESTS.json'
EXTERNAL_RUNTIME = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PRIOR_EXPECTED = {
 'PLAN.json':'8c5e8c3b29961fc1b909059cd0acefec697b2d5c10cd376065c767221c87d40b',
 'COMPLETION.json':'e2755642efc529e5932d4bfae0d5ed9be7cf84934340dccb20cdde51625e545d',
 'results/COMPLETION.json':'9ea19847727cde355410a4656063d9d5d300d491b0bdf539064d210ebe6cd15e',
 'RESEARCH_REVIEW.json':'8f2f6886f8e3cfa74565ce9b87825c268aa899d29d329bb00904885f90d611c9'}
MODEL_FLAGS = ('passed','independent_augmented_least_squares','annual_prior_december_cutoffs_checked',
 'five_year_mature_cohort_checked','four_target_common_cohort_checked','original_source_and_local_parity',
 'availability_unchanged','own_history_threshold_checked','matched_month_identity_checked',
 'query_security_excluded','peer_kernel_checked','own_month_mass_preserved',
 'common_shrinkage_penalty_checked','no_future_or_outcome_inputs','audit_checked')
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
    return module('monthly_identity_transfer206_original', LOCAL / 'study.py')

def contract():
    require(sha(CONTEXT)==CONTEXT_SHA,'Changed prospective context')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),
        candidate=NAMES[1],control_nonpromotable=NAMES[2],period=context['evaluation']['period'],
        training_months=88,later_months=34,model_prefixes=list(MODEL_PREFIXES),all123_order_prefixes=True,
        model_fit_accounting=context['formation']['fit_accounting'],limitation=context['limitations'],
        historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',
        new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,
        original_pool_preserved=True,no_parameter_search=True,no_new_score_budget=True)


def code_files():
    direct=[ROOT/'work'/name for name in ('monthly_identity_transfer206_host.py',
        'test_monthly_identity_transfer206_host.py','monthly_identity_transfer206_model.py',
        'test_monthly_identity_transfer206_model.py','monthly_identity_transfer206_independent.py',
        'test_monthly_identity_transfer206_independent.py','monthly_identity_transfer206_context.json',
        'ptr169_rebuild_evaluate.py','ptr169_results_independent.py')]
    import monthly_identity_transfer206_model as model
    import monthly_identity_transfer206_independent as independent
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
    backup = CENTRAL.parent / ('PRE_MONTHLY_IDENTITY_TRANSFER206_' + tag + '_' + expected + '.json')
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
    path = ROOT / 'work/identity206_CODE_REVIEW.json'
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
        close_vs_own_control=guard(metrics[NAMES[1]]['same_close_zero'], metrics[NAMES[2]]['same_close_zero'], 0.),
        open_vs_own_control=guard(metrics[NAMES[1]]['next_open_cost'], metrics[NAMES[2]]['next_open_cost'], 0.),
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
    import monthly_identity_transfer206_independent as independent
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
    import monthly_identity_transfer206_independent as independent
    check = independent.audit_outcomes(px, op, orders, exports, results, meta); require_outcome_audit(check)
    verify_training_against_audit(read(OUT / 'selected/TRAIN_RESULTS.json'), check)
    dump(dest / 'INDEPENDENT_OUTCOME.json', check)
    dump(dest / 'RESULTS.json', dict(selection=frozen['selection'], arms=results, reused_sample=True,
        strict_pit=False, matched_month_identity_transfer=True, same_support_both_arms=True, observer_promotion=False))
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
    require(len(state['completed'])==203 and sum(row['trials'] for row in state['completed'])==1126,
            'Expected203 completed entries /1126 trials')
    require(state['best_training_selected_candidate']['directory']=='monthly_local_linear70','Incumbent changed')
    require(state['completed'][-1]['directory']==PRIOR.name,'Round205 must be the completed predecessor')


def model_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed model source review')
    review=read(SOURCE_REVIEW)
    require(review.get('passed') is True and review.get('context_sha256')==CONTEXT_SHA,'Source admission/context differs')
    require(review.get('raw_source_hashes')==raw,'Original raw source lineage changed')
    for key in ('actual_features_computed','actual_model_fits_computed','actual_orders_computed','actual_returns_computed','actual_source_arrays_loaded','new_data_acquired'):
        require(review.get(key) is False,'Source review crossed research boundary: '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    require(isinstance(review.get('source_sha256'),dict) and review['source_sha256'],'Missing immutable model sources')
    for name,digest in review['source_sha256'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        require(path.resolve()!=CENTRAL.resolve(),'Mutable central cannot be a persistent source input')
        bind(pins,path,digest)
    require(review.get('immutable_source_files')==len(review['source_sha256']) and review.get('prior_saved_training_rows')==8820 and review.get('prior_saved_query_rows')==5978 and review.get('prior_saved_feature_count')==10 and review.get('prior_parity_receipt_reverified') is True and review.get('strict_pit') is False,'Original model source coverage metadata differs')
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,
        original_cached_training_and_queries=True,new_labels=False,new_raw_features=False,strict_pit=False)


def tests():
    names=('test_monthly_identity_transfer206_host.py','test_monthly_identity_transfer206_model.py',
           'test_monthly_identity_transfer206_independent.py')
    modules=[module('test_identity206_'+str(i),ROOT/'work'/name) for i,name in enumerate(names)]
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
    plan=dict(schema='monthly_identity_transfer206_v1',created_utc=now(),runtime=ev.runtime(),
        contract=contract(),input_sha256=pins,raw_source_hashes=raw,predecessor_binding=previous,
        model_source_binding=source,numerical_code_review=review,
        prior_central_sha256=hashlib.sha256(before).hexdigest(),protected_files=len(protected),
        actual_new_feature_evaluations_before_registration=0,actual_new_model_fits_before_registration=0,
        actual_new_peer_corrections_before_registration=0,actual_new_orders_before_registration=0,
        actual_new_returns_before_registration=0)
    dump(OUT/'PLAN.json',plan)
    state.update(current=OUT.name,current_status='registered_before_new_identity_transfer_fits',
        registered_queue=[dict(directory=OUT.name,plan_sha256=sha(OUT/'PLAN.json'))],updated_utc=now(),
        last_goal_turn_classification='progress',next_step='Prepare matched-month identity transfer and independent audit, then freeze first88 selection.')
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


def require_model_audit(check,forecast):
    for key in MODEL_FLAGS:require(check.get(key) is True,'Independent model evidence missing '+key)
    require(check.get('atol')==1e-12 and check.get('computes_returns') is False,'Independent model scope/tolerance differs')
    require(check.get('residuals_are_mature_in_sample') is True and check.get('residuals_are_oof') is False,
            'Residual validation claim differs')
    require(set(forecast['raw'])==set(RAW_KEYS),'Raw forecast keys differ')
    available=np.asarray(forecast['available']);local_available=np.asarray(forecast['local_available'])
    require(available.shape==local_available.shape==(5978,) and available.dtype==local_available.dtype==np.dtype(bool),'Full query availability axes differ')
    require(not np.any(local_available&~available),'Local availability exceeds pooled availability')
    for key,value in forecast['raw'].items():
        value=np.asarray(value)
        require(value.shape==(5978,) and not np.isinf(value).any() and np.array_equal(np.isfinite(value),available),'Raw forecast support differs: '+key)
    np.testing.assert_array_equal(forecast['raw']['control'],forecast['raw']['local_constant'])
    audit=forecast['audit']
    require(check.get('annual_cohorts')==len(audit['annual'])==11,'Annual audit coverage')
    require(check.get('local_queries')==len(audit['local'])==int(np.count_nonzero(forecast['available'])),
            'Local-query audit coverage')
    counts=audit['counts']
    require(isinstance(counts,dict) and set(counts)==set(MODEL_COUNT_FIELDS) and all(type(v) is int and v>=0 for v in counts.values()),'Invalid model accounting')
    require(counts['actual_regression_solves']==sum(counts[key] for key in ('source_target_fits','reference_pooled_log_refits','reference_local_solves','transfer_pooled_log_refits')),'Regression solve accounting differs')
    require(counts['peer_query_evaluations']==counts['own_constant_evaluations']==sum(row.get('status')=='fit' for row in audit['local']),'Correction query counts differ')
    require(counts['peer_month_evaluations']==sum(len(row.get('peer_months',[])) for row in audit['local']),'Peer-month counts differ')
    require(counts['peer_row_observations']==sum(month['peer_count'] for row in audit['local'] for month in row.get('peer_months',[])),'Peer-row counts differ')
    require(isinstance(check.get('counts'),dict) and set(check['counts'])==set(counts),'Independent count schema differs')
    for key,value in counts.items():
        require(type(check['counts'][key]) is int and check['counts'][key]==value,'Independent model accounting differs: '+key)
    require(type(check.get('independent_pooled_fits')) is int and check['independent_pooled_fits']>=0,
            'Missing independent fit accounting')


def require_slot_audit(check,deltas):
    require(set(deltas)==set(RAW),'Delta arm keys differ')
    for value in deltas.values():
        require(np.asarray(value).shape==(122,49) and np.isfinite(value).all() and np.max(np.abs(value))<=.0125+1e-12 and np.array_equal(value[0],np.zeros(49)),'Delta axes/bounds/initial allocation differ')
    for key in SLOT_FLAGS:require(check.get(key) is True,'Independent slot evidence missing '+key)
    require(set(check.get('ready_rows',{}))==set(RAW),'Missing readiness coverage')
    require(all(type(v) is int and 0<=v<=122 for v in check['ready_rows'].values()) and
            len(set(check['ready_rows'].values()))==1,'Candidate readiness differs')
    require(check.get('changed_cells')=={key:int(np.count_nonzero(value)) for key,value in deltas.items()},'Changed-cell audit differs')


def prepare(expected):
    plan=gate(expected);require(not (OUT/'prepared').exists(),'Preparation is write-once')
    import monthly_identity_transfer206_model as model
    import monthly_identity_transfer206_independent as independent
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
    forecast=model.annual_forecasts(*model_inputs(train,meta,a))
    require(set(forecast['raw'])==set(RAW_KEYS),'Unexpected model raw schema')
    np.testing.assert_array_equal(forecast['raw']['control'],forecast['raw']['local_constant'])
    np.testing.assert_array_equal(forecast['raw']['control'],a['raw_local_constant'].reshape(-1))
    np.testing.assert_array_equal(forecast['available'],a['model_available'].reshape(-1))
    np.testing.assert_array_equal(forecast['local_available'],a['local_available'].reshape(-1))
    check=independent.check_prepare(train,meta,a,forecast);require_model_audit(check,forecast)
    orders,ranks,deltas=decisions(study,a,m,forecast,122)
    require(orders[NAMES[0]]==read(LOCAL/'local_linear_ORDERS.json'),'Original full order parity failed')
    slot=independent.verify_rank_slot(a,forecast,ranks,deltas);require_slot_audit(slot,deltas)
    prefix_evidence=[]
    for n in MODEL_PREFIXES:
        inputs=model_inputs(train,meta,a,limit=n)
        small=model.annual_forecasts(*inputs);q=n*49
        for key in RAW_KEYS:np.testing.assert_array_equal(small['raw'][key],forecast['raw'][key][:q])
        for key in ('available','local_available'):np.testing.assert_array_equal(small[key],forecast[key][:q])
        prefix_evidence.append(dict(prefix=n,training_rows=len(inputs[0]),query_rows=q,
            query_date=m['dates'][n-1] if n else None,
            cutoff='%04d-12-31'%(int(m['dates'][n-1][:4])-1) if n else None,
            latest_label_end=str(np.asarray(inputs[1],dtype='datetime64[D]').max()) if len(inputs[1]) else None,
            counts=small['audit']['counts']))
    for n in range(123):
        oo,rr,dd=decisions(study,a,m,forecast,n)
        require(all(oo[key]==orders[key][:n] for key in NAMES),'Prefix order parity failed')
        for key in RAW:
            np.testing.assert_array_equal(rr[key],ranks[key][:n]);np.testing.assert_array_equal(dd[key],deltas[key][:n])
    require(before=={key:(v.dtype.str,v.shape,v.tobytes()) for key,v in a.items()},'Original arrays modified')
    gate(expected);dest=OUT/'prepared';dest.mkdir()
    np.savez_compressed(dest/'FORECASTS.npz',**{'raw_'+key:value for key,value in forecast['raw'].items()},
        available=forecast['available'],local_available=forecast['local_available'])
    np.savez_compressed(dest/'RANKS_DELTAS.npz',**{'rank_'+key:v for key,v in ranks.items()},**{'delta_'+key:v for key,v in deltas.items()})
    dump(dest/'MODEL_AUDIT.json',forecast['audit']);dump(dest/'ORDERS.json',orders);dump(dest/'META.json',m)
    dump(dest/'SOURCE_PARITY.json',dict(original_training_cache_exact=True,original_identity_exact=True,
        raw_source_hashes=plan['raw_source_hashes'],model_source_binding=plan['model_source_binding'],
        new_labels=False,new_raw_features=False,original_control_constant_exact=True))
    evidence=dict(model=check,slot=slot,prefixes=123,model_prefixes=list(MODEL_PREFIXES),
        model_prefix_sources_truncated=True,model_prefix_evidence=prefix_evidence,
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
    for key in ('model_prefix_sources_truncated','original_orders_exact','original_arrays_unchanged',
                'original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget'):
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
        next_step='Review matched-month identity transfer and source limitations; preserve frozen observers.')
    require(len(state['completed']) == 204 and sum(r['trials'] for r in state['completed']) == 1129, 'Wrong final counters')
    commit_state(before, state, 'close'); print(json.dumps(receipt), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['test','register','prepare','select','diagnose','close'])
    parser.add_argument('--plan-sha256');args=parser.parse_args()
    if args.phase=='test':tests()
    elif args.phase=='register':register()
    else:
        require(bool(args.plan_sha256),'An explicit registered PLAN SHA is required')
        globals()[args.phase](args.plan_sha256)
